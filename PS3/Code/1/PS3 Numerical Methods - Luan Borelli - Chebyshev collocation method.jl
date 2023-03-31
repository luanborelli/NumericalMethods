#########################################
## Problem Set 3 - Numerical Methods   ##
## Chebyshev Collocation               ##
## Student: Luan Borelli               ##
#########################################

###############################
## Importing useful packages ##
###############################

using Plots, NLsolve, Distributions, Random, Base.Threads

#################################
## Defining Tauchen's function ##
#################################

function tauchen(μ,σsq,ρ,N,m)
  
    σ = sqrt(σsq); # Standard deviation of ε.
    μ_θ = μ; # Expected value of θ_t. Notice that in this particular specification, it is exactly μ. In other specifications, it could be e.g. μ/(1-ρ).
    σ_θ = σ/sqrt(1-ρ^2); # Standard deviation of θ_t.
      
    θ = range(μ_θ - m*σ_θ, μ_θ + m*σ_θ, N); # Generating grid points. 
    Δθ = θ[2]-θ[1]; # This is the width of the invervals. Notice that all the points are equidistant, by construction.
      

    # Now we calculate all the probability transition matrix elements "at once" through matrix operations. 
    # Idea: all we need to do this is to calculate a matrix of "all possible differences" of grid θ elements.
    # In general, letting 1 be a vector of ones, the matrix of all possible differentes between elements
    # of vectors  u and v is given by: D := 1u' - v1'.

    θ_j = ones(N,1)*θ';
    θ_i = θ*ones(N,1)';
    
    P_1 = cdf(Normal(),((θ_j - ρ*θ_i .+ Δθ/2) .- (1-ρ)*μ)/σ);
    P_2 = cdf(Normal(),((θ_j - ρ*θ_i .- Δθ/2) .- (1-ρ)*μ)/σ);
  
    P = P_1 - P_2;
    
    # Calculating corner transition probabilities:
    P[:,1] = P_1[:,1]; 
    P[:,N] = -P_2[:,N] .+ 1;

    return θ, P    
  end


###############################
## Defining model parameters ##
###############################

β = 0.987
μ = 2
α = 1/3 
δ = 0.012 
ρ = 0.95
σ = 0.007

k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1)) # Steady state value for capital. 

a = 0.75*k_ss # Lower bound for the capital domain.
b = 1.25*k_ss # Upper bound for the capital domain. 
k_grid = range(a, b, length = 500); # Generatng the capital domain. 

tauch = tauchen(0,σ^2,ρ,7,3); # Discretizing the state space for the parameterization of the problem. 
z_grid = exp.(tauch[1]); # Grid.
Π = tauch[2]; # Transition matrix.  

########################################################################
## Defining specific functions for the (Chebyshev) collocation method ##
########################################################################

# Chebyshev polynomials are defined in [-1, 1]. Thus we need to define the following two functions:

normalized_k(k) = 2*((k - a)/(b - a)) - 1 # This function translates values from k_grid into the [-1,1] domain. 
unnormalized_k(k) = ((b - a)/2)*(k + 1) + a # This function translates values from [-1,1] into the k_grid domain. 

# Defining Chebyshev polynomials and Chebyshev roots functions:

chebyshev_polynomial(x, degree) = cos(degree * acos(x)); # This function evaluates a Chebyshev polynomial of degree 'degree' at the point 'x'. 
chebyshev_root(i, degree) = -cos((2*i-1)/(2*degree)*pi); # This function obtains the 'i'-th root of the degree 'degree' Chebyshev polynomial. 
chebyshev_roots(degree) = [chebyshev_root(i, degree) for i in 1:degree]; # This function returns all the 'degree' degree Chebyshev polynomial.

# Here our basis functions are the Chebyshev polynomials. 
# Defining the "approximate policy function", c_hat: 

function c_hat(k, z_ind, γ) # "Approximate" policy function.
    fval = 0; 
    size_γ = size(γ)[1]
    for i in 0:(size_γ - 1) # Notice that length(γ) = polynomial order + 1. I.e., length(γ) - 1 is the polynomial order.
        fval = fval + γ[i+1, z_ind]*chebyshev_polynomial(normalized_k(k), i)
    end 
    return fval 
end

# Defining the residual function, R(k, z, γ):

function R(k, z_ind, γ) # Residual function. 
    kp = z_grid[z_ind] * k^α + (1-δ)*k - c_hat(k, z_ind, γ) # k'
    cps = [c_hat(kp, z, γ) for z in 1:length(z_grid)] # c(k', z')'s
    error = c_hat(k, z_ind, γ).^(-μ) - β * Π[z_ind,:]' * (cps .^(-μ) .* ( α*z_grid*kp.^(α - 1) .+ 1 .- δ))
    # error = c_hat(k,γ) - (β * Π[z_ind,:]' * (c_hat(kp, γ)^(-μ) * ( α*z_grid*kp^(α - 1) .+ 1 .- δ) ))^(-1/μ)
    return error
end

# Defining the system: 

function system(γ) # This function constructs the system to be solved. 
    size_γ = size(γ)[1]
    err = zeros(size_γ, length(z_grid))

    roots = chebyshev_roots(size_γ)
    collocation_points = unnormalized_k.(roots)

    for z in 1:length(z_grid)
        for i in 1:size_γ
            err[i, z] = R(collocation_points[i], z, γ)
        end
    end 

    return err
end

###########################################################
## Solving the RBC model by Chebyshev Collocation Method ##
###########################################################

deg = 5 # Maximum degree for the basis functions (Chebyshev polynomials). I set 5.

params_c = zeros(deg, length(z_grid)) # A vector that will allocate the final vector for the basis functions' coefficients.  
policy_c = zeros(length(k_grid), length(z_grid)) # A vector that will allocate the final consumption policy function.  

### Solving the model.
# Note that I use an initial guess "improvement trick". 
# It turns out that the collocation method is quite sensitive with respect to the initial guess considered and, depending on the guess, 
# the model may not converge properly to the appropriate solution. The problem is that the greater the chosen maximum degree 'deg',
# the greater will be the vector of coefficients of basis functions and, therefore, the greater will be the dimension of the initial 
# guess that we must elaborate. Therefore, the more difficult it will be to "predict" a reasonable guess. 
# To avoid this, the idea is to start with a low maximum degree 'deg' (say, one), solve the system using an arbitrary guess, 
# use the solution (concatenating an additional zero at the end of the vector to match dimensions) as an initial guess to re-solve the model, 
# now to ' deg'+1 (say, two) and repeat the process until you reach the final desired degree (say, five). With this successive improvement of guesses, 
# proper convergence of the solution becomes more likely. 

# Note that, to fully characterize the economy, in practice we solve "7 models", one for each possible value of z (that is, one for each state of nature).

@time begin # Solving the system.
        guess = ones(2, length(z_grid)) 
        for n in 1:deg 
            results = nlsolve(system, guess).zero
            if n < deg
                guess = vcat(results, zeros(length(z_grid))')
            else 
                guess = results
            end 
            # print("\n", guess, "\n")
        end 
        params_c = guess
end 

# Recovering the consumption policy function: 
policy_c = [c_hat(k, z, params_c) for k in k_grid, z in 1:length(z_grid)]

# Recovering the capital policy function: 
policy_k = [z_grid[z]*k_grid[k]^α + (1-δ)*k_grid[k] - policy_c[k, z] for k in 1:length(k_grid), z in 1:length(z_grid)]

# Recovering the value function:

# For this, we need to define some additional and values and functions: 

tol = 10^(-5); # Tolerance for the distance between elements of the value function.
# We will use this tolerance to define when to stop the iterative process.
# tol defines, therefore, the distance from which we consider that the elements of the value function
# are "close enough". Remember that the sequence of V's is a Cauchy sequence.

iter = 0;  # We will use this variable to count the number of iterations.
maxiter = 1000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.
V_prev = ones(length(k_grid), length(z_grid)); # Temporary vector for the value function iteration.

# Moll's initial guess for the value function: 
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # A kind of "consumption matrix", but setting k' = k. This matrix computes all possible consumptions for all possible combinations of k and z.  
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # Initial guess, constructed from c.

# "Estimating" the positions of k'(k, z) (capital  policy function) on the exogenous grid: 
pol_index = [argmin(abs.(k_grid .- policy_k[i,j])) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Obtaining the capital policy function on the exogenous grid: 
policy_exo = [k_grid[pol_index[i, j]] for i in eachindex(k_grid), j in eachindex(z_grid)]

# Defining preferences: 
function u(c) 
    if c > 0
        u = (c^(1-μ) - 1)/(1-μ); # Utility function
    else 
        u = -Inf # This is necessary to avoid negative consumption. If c < 0, utility = -∞.
    end
    return u
end

# Obtaining the value function: 

while maximum(abs.(V_prev - V)) > tol && iter < maxiter
    V_prev = copy(V)
    for j in 1:length(z_grid)
        for i in 1:length(k_grid)
            V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_exo[i,j]) + β * Π[j,:]' * V_prev[pol_index[i,j],:] 
            # print("\n", V[i, j], "\n")
        end
    end

    # iter += 1;
    # print("\n", "Iter: ", iter)
    # print("\n", "Distance: ", maximum(abs.(V_prev - V)))

end

#######################
## Results and plots ##
#######################

# Plotting the value function: 

plot(k_grid, V, 
    xlab = "k",
    ylab = "V", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(k_grid,1,7), repeat(z_grid',length(k_grid)), V,
    xlab = "k", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot

# Plotting the capital policy function:

plot(k_grid, policy_k,
    xlab = "k",
    ylab = "k'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(k_grid,1,7), repeat(z_grid',length(k_grid)), policy_k,
    xlab = "k", 
    ylab = "z", 
    zlab = "k'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.
 
# Plotting the consumption policy function: 

plot(k_grid, policy_c,
    xlab = "k",
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(k_grid,1,7), repeat(z_grid',length(k_grid)), policy_c,
    xlab = "k", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

###########################
## Euler Equation Errors ##
###########################

cp = [c_hat(policy_k[k,z1], z2, params_c) for k in 1:length(k_grid), z1 in 1:length(z_grid), z2 in 1:length(z_grid)] # Generate an array of C primes; an array of C primes for each state.

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (cp[k_ind, z_ind, :].^(-μ) .* (α*z_grid*policy_k[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors:
plot(k_grid, EEEs,
    xlab = "k",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

###########################################
## Problem Set 3 - Numerical Methods     ##
## Finite Elements Method + Galerkin     ##
## Student: Luan Borelli                 ##
###########################################

###############################
## Importing useful packages ##
###############################

using Plots, NLsolve, Distributions, Random, FastGaussQuadrature, Base.Threads

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
k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))
k_grid = range(0.75*k_ss, 1.25*k_ss, length = 15); # For finite elements methods we don't need a very "fine" grid to represent the capital domain. 
# A few points are enough for a good approximation. For this reason, I generate a grid of size 15 here.

tauch = tauchen(0,σ^2,ρ,7,3)
z_grid = exp.(tauch[1]) 
Π = tauch[2]

#########################################################
## Defining specific functions for the Galerkin Method ##
#########################################################

## I consider Gauss-Legendre quadrature for the numerical integration function:

function integral_gl(f, a, b, n)
    nodes, weights = gausslegendre(n);
    result = weights' * f.((b-a)/2 * nodes .+ (a+b)/2)*(b-a)/2
    return result
end

#= But another possible alternative would be to use the Gauss-Chebyshev quadrature:

function integral_gc(f, a, b, n)
   nodes, weights = gausschebyshev(n); 
   result = pi*(b-a)/(2*n) * sum(f.((nodes .+ 1)*(b-a)/2 .+ a) .* sqrt.(1 .- nodes.^2))
   return result  
end

=# 

# Defining the basis function: 

function basis(k, i)
    if i == 1 
        if k >= k_grid[i] && k <= k_grid[i+1]
            value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
        else 
            value = 0
        end 
    elseif i == length(k_grid)
       if k >= k_grid[i-1] && k <= k_grid[i]
            value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
       else 
            value = 0
       end 
    elseif k >= k_grid[i-1] && k <= k_grid[i]
        value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
    elseif k >= k_grid[i] && k <= k_grid[i+1]
        value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
    else 
        value = 0
    end 
    return value 
end  

# Defining the "approximate policy function", c_hat: 

function c_hat(k, z_ind, γ)
    fval = 0; 
    size_γ = size(γ)[1]
    for i in 1:size_γ
        fval = fval + γ[i, z_ind]*basis(k, i)
    end 
    return fval 
end

# Defining the residual function, R(k, z, γ):

function R(k, z_ind, γ) 
    kp = - c_hat(k, z_ind, γ) + z_grid[z_ind]*k^α + (1-δ)*k # k'
    cps = [c_hat(kp, z, γ) for z in 1:length(z_grid)] # c(k', z')'s 
    error = c_hat(k, z_ind, γ).^(-μ) - β * Π[z_ind,:]' * (cps .^(-μ) .* ( α*z_grid*kp.^(α - 1) .+ 1 .- δ))
    return error
end

# Defining the system: 

function system(γ) 
    size_γ = size(γ)[1]
    err = zeros(size_γ, length(z_grid))

    for z in 1:length(z_grid)
        for i in 1:size_γ

            if i > 1
                term_1(k) = R(k, z, γ).*((k-k_grid[i-1])/(k_grid[i] - k_grid[i-1]))
                err[i, z] = err[i, z] + integral_gl(term_1, k_grid[i-1], k_grid[i], length(k_grid))
            end 
            
            if i < size_γ
                term_2(k) = R(k, z, γ).*((k_grid[i+1]-k)/(k_grid[i+1] - k_grid[i]))
                err[i, z] = err[i, z] + integral_gl(term_2, k_grid[i], k_grid[i+1], length(k_grid))
            end 

        end
    end 

    return err
end


################################################################
## Solving the RBC model by Finite Elements Method + Galerkin ##
################################################################

params_c = zeros(length(k_grid), length(z_grid)) # A vector that will allocate the final vector for the basis functions' coefficients.
policy_c = zeros(length(k_grid), length(z_grid)) # A vector that will allocate the final consumption policy function.
init_guess = repeat([i for i in range(2,4, length(k_grid))]', length(z_grid))' # Convergence requires a minimally intelligent initial guess. This is why I consider this specific initial guess.

@time begin 
    params_c = nlsolve(system, init_guess).zero # Solving the system. 
end 

# Recovering the consumption policy function: 
policy_c = [c_hat(k, z, params_c) for k in k_grid, z in 1:length(z_grid)]

# But note that the initial grid we used to solve the model had only 15 points. 
# What if we want to evaluate the policy function in a "richer" grid? 
# Well, we can simply generate a new grid of the desired size and recalculate the policy function 
# using the coefficients obtained by the solution of the original system (which considered only 15 grid points).

# Now we create a richer grid (say with 500 points) and get the consumption policy for that grid by 
# simply calculating the above function on this grid: 
function estimated_c(k, z)
    fval = 0; 
        for i in 1:length(params_c[:, z])
            fval = fval + params_c[:, z][i]*basis(k, i)
        end 
    return fval 
end

# Now we create a richer grid (say with 500 points) and get the consumption policy for that grid by 
# simply calculating the above function on this grid: 
new_k_grid = range(0.75*k_ss, 1.25*k_ss, length = 500);
policy_c_500 = hcat([estimated_c.(new_k_grid, z) for z in 1:length(z_grid)]...)

# Recovering the capital policy function (for 15 grid points): 
policy_k = [z_grid[z]*k_grid[k]^α + (1-δ)*k_grid[k] - policy_c[k, z] for k in 1:length(k_grid), z in 1:length(z_grid)]

# Now, for 500 grid points: 
policy_k_500 = [z_grid[z]*new_k_grid[k]^α + (1-δ)*new_k_grid[k] - policy_c_500[k, z] for k in 1:length(new_k_grid), z in 1:length(z_grid)]

# Recovering the value function (for 500 grid points): 

tol = 10^(-5); # Tolerance for the distance between elements of the value function.
# We will use this tolerance to define when to stop the iterative process.
# tol defines, therefore, the distance from which we consider that the elements of the value function
# are "close enough". Remember that the sequence of V's is a Cauchy sequence.

iter = 0; # We will use this variable to count the number of iterations.
maxiter = 1000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.
V_prev = ones(length(new_k_grid), length(z_grid)); # Temporary vector for the value function iteration.
c = [z_grid[i]*(new_k_grid[j]^α).-new_k_grid[j].+(1-δ)*new_k_grid[j] for j in 1:length(new_k_grid), i in 1:length(z_grid)] # A kind of "consumption matrix", but setting k' = k. This matrix computes all possible consumptions for all possible combinations of k and z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # Moll's initial guess, constructed from c.

# "Estimating" the positions of k'(k, z) (capital  policy function) on the exogenous grid: 
pol_index = [argmin(abs.(new_k_grid .- policy_k_500[i,j])) for i in 1:length(new_k_grid), j in 1:length(z_grid)]

# Obtaining the capital policy function on the exogenous grid: 
policy_exo = [new_k_grid[pol_index[i, j]] for i in eachindex(new_k_grid), j in eachindex(z_grid)]

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
        for i in 1:length(new_k_grid)
            V[i, j] = u(z_grid[j]*new_k_grid[i]^α + (1-δ)*new_k_grid[i] - policy_exo[i,j]) + β * Π[j,:]' * V_prev[pol_index[i,j],:] 
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

plot(new_k_grid, V, 
    xlab = "k",
    ylab = "V", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(new_k_grid,1,7), repeat(z_grid',length(new_k_grid)), V,
    xlab = "k", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot

# Plotting the capital policy function:

plot(new_k_grid, policy_k_500,
    xlab = "k",
    ylab = "k'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(new_k_grid,1,7), repeat(z_grid',length(new_k_grid)), policy_k_500,
    xlab = "k", 
    ylab = "z", 
    zlab = "k'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# Plotting the consumption policy function: 

plot(new_k_grid, policy_c_500,
    xlab = "k",
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(new_k_grid,1,7), repeat(z_grid',length(new_k_grid)), policy_c_500,
    xlab = "k", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

##########
## EEEs ##   
##########

cp = [c_hat(policy_k_500[k,z1], z2, params_c) for k in 1:length(new_k_grid), z1 in 1:length(z_grid), z2 in 1:length(z_grid)] # Generate an array of C primes; an array of C primes for each state. 

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (cp[k_ind, z_ind, :].^(-μ) .* (α*z_grid*policy_k_500[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c_500[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(new_k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(new_k_grid, EEEs,
    xlab = "k",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

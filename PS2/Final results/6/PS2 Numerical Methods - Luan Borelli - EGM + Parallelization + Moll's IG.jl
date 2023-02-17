############################################################################
## Problem Set 2 - Numerical Methods                                      ##
## Endogenous Grid Method (EGM) + Parallelization + Moll's initial guess  ##
## Student: Luan Borelli                                                  ##
############################################################################

###############################
## Importing useful packages ##
###############################

using Distributions, Random, Plots, Base.Threads, Interpolations, Roots

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

## Model parameters 

μ = 2;
β = 0.987;
α = 1/3;
δ = 0.012;
k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))

## Technical values

tol = 10^(-5); # Tolerance for the distance between elements of the value function.
# We will use this tolerance to define when to stop the iterative process.
# tol defines, therefore, the distance from which we consider that the elements of the value function
# are "close enough". Remember that the sequence of V's is a Cauchy sequence.

iter = 0; # We will use this variable to count the number of iterations.
maxiter = 100000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.

# Defining the utility function: 

function u(c) 
    if c > 0
        u = (c^(1-μ) - 1)/(1-μ); # Utility function.
    else 
        u = -Inf # This is necessary to avoid negative consumption. If c < 0, utility = -∞.
    end
    return u
end

k_grid = range(0.75*k_ss, 1.25*k_ss, length = 500); # A grid for capital values. Required to discretize the domain.
tauch = tauchen(0,0.007^2,0.95,7,3) # Tauchen discretization of the AR(1) process.
z_grid = exp.(tauch[1]) # Grid for shocks. Note that since the process is log'd, we need to exponentiate the grid returned by the tauch(.) function.
Π = tauch[2] # Transition probabilities matrix for shocks.

V_prev = ones(length(k_grid), length(z_grid)); # Temporary vector for the value function iteration.
policy = zeros(length(k_grid), length(z_grid)); # Vector that will store the policy function.
policy_index = Int.(zeros(length(k_grid), length(z_grid))); # Vector that will store the indexes of the policy function.
values = zeros(length(k_grid), length(z_grid)); # Vector that will store the Bellman equation values during the brute-force grid-search. 

# V = zeros(length(k_grid), length(z_grid)); # Initial guess for the value function. 
# Usually zero. But there are better guesses. For example, Moll's initial guess: 
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # A king of "consumption matrix", but setting k' = k. This matrix computes all possible consumptions for all possible combinations of k and z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # Initial guess, constructed from c.

###########################################################################
## Endogenous Grid Method (EGM) + Parallelization + Moll's initial guess ##
###########################################################################

policy_c_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função política.  
policy_c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z.  # ones(length(k_grid), length(z_grid)); # Vetor que armazenará a policy function 
policy_k = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_c[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)]

# Defining a function for the FOC expression that we need to find the root:
foc(k, z_index, kp_index) = z_grid[z_index]*k^α + (1-δ)*k - β^(-1/μ) * (Π[z_index,:]' * (policy_c[kp_index,:].^(-μ).*(α*z_grid*k_grid[kp_index].^(α-1).+(1-δ)))).^(-1/μ) .- k_grid[kp_index] 

egm_grid = zeros(length(k_grid), length(z_grid)) # Defining the variable that will store the EGM k grid.
egm_policy_k = zeros(length(k_grid), length(z_grid)) # Defining the variable that will store the EGM policy function. 

# Solving: 
@time begin
    
    while maximum(abs.(policy_c_prev - policy_c)) > tol && iter < maxiter
        
        policy_c_prev = copy(policy_c)

        # Constructing the endogenous grid: 
        @threads for i in 1:length(k_grid) 
            @threads for j in 1:length(z_grid) 
                egm_grid[i,j] = find_zero((k -> foc(k, j, i)), 0) # Finding the root; i.e., solving for k.
            end 
        end
        
        # Recovering the capital policy on the exogenous grid by interpolation: 
        @threads for i in 1:length(k_grid)
            @threads for j in 1:length(z_grid)
                if k_grid[i] > egm_grid[1,j] && k_grid[i] < egm_grid[length(k_grid), j]
                    egm_policy_k[i,j] = linear_interpolation(egm_grid[:,j],k_grid)(k_grid[i]) # Interpolating. 
                elseif k_grid[i] < egm_grid[1,j] # Following Gordon's slides, at points that fall outside the grid boundaries (below or above) we need to set this way...
                    egm_policy_k[i, j] = k_grid[1]
                elseif k_grid[i] > egm_grid[length(k_grid), j]
                    egm_policy_k[i, j] = k_grid[length(k_grid)]
                end 
            end 
        end 

        # Updating consumption policy function:
        policy_c = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - egm_policy_k[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)] 

        iter += 1; 
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(policy_c_prev - policy_c)), "\n")
    end 

end

# Recovering capital policy function: 
policy = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_c[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)]

# "Estimating" the positions of k'(k, z) (capital  policy function) on the exogenous grid: 
policy_index = [argmin(abs.(k_grid .- policy[i,j])) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Obtaining the capital policy function on the exogenous grid: 
policy_exo = [k_grid[policy_index[i, j]] for i in eachindex(k_grid), j in eachindex(z_grid)]

# Recovering the value function:

while maximum(abs.(V_prev - V)) > tol && iter < maxiter
    V_prev = copy(V)
    for j in 1:length(z_grid)
        for i in 1:length(k_grid)
            V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_exo[i,j]) + β * Π[j,:]' * V_prev[policy_index[i,j],:] 
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

# Plotting value function: 
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
    camera=(20,40)) # 3D plot.

# Plotting capital policy function: 
plot(k_grid, policy,
    xlab = "k",
    ylab = "k'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

plot(repeat(k_grid,1,7), repeat(z_grid',length(k_grid)), policy,
    xlab = "k", 
    ylab = "z", 
    zlab = "k'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# Plotting consumption policy function: 
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

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (policy_c[policy_index[k_ind,z_ind],:].^(-μ) .* (α*z_grid*policy[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plt = plot(k_grid, EEEs,
    xlab = "k",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)
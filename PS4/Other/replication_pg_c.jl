#########################################
## Problem Set 4 - Numerical Methods   ##
## Incomplete Markets                  ##
## Student: Luan Borelli               ##
#########################################

###############################
## Importing useful packages ##
###############################

using Plots, Distributions, Random, Base.Threads, LinearAlgebra, SparseArrays, Roots, NLsolve #,  Arpack

#################################
## Defining Tauchen's function ##
#################################

#######
## a ##
#######
 
function tauchen(μ,σsq,ρ,N,m)

    # This code calculates and returns states vector (grid points) and transition matrix for the discrete markov process approximation 
    # of AR(1) process specified as θ_t = μ*(1-ρ) + ρ*z_{t-1} + ε_t, ε_t ~ N(0,σ^2), by Tauchen's method.
  
    # m stands for the maximum number of standard deviatons from mean;  
    # N stands for the number of states in discretization of θ (must be an odd number).
  
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

β = 0.96;
γ = 1.0001;
ρ = 0.9;
σ = 0.01;

z_grid, Π = tauchen(0, σ^2, ρ, 9, 3);

# Defining preferences:

function u(c) 
    if c > 0
        u = (c^(1-γ) - 1)/(1-γ); # Utility function
    else 
        u = -Inf # This is necessary to avoid negative consumption. If c < 0, utility = -∞.
    end
    return u
end

#######
## b ##
#######

z_min = z_grid[1]
z_max = z_grid[length(z_grid)]

rf = 1/β - 1; # Risk-free rate. 
ϕ = exp(z_min)/rf; # Natural debt limit.

# Generating the asset grid: 

grid_size = 201;             
a_min = -1;
a_max = +4;
a_grid = range(a_min, a_max, grid_size);


function blockmatrix_to_matrix(A::Array{Array{T,2},2}) where T
    nblocks = size(A)
    nrows, ncols = size(A[1,1])
    nrows_total = nrows * nblocks[1]
    ncols_total = ncols * nblocks[2]
    B = zeros(T, nrows_total, ncols_total)
    for i in 1:nblocks[1]
        for j in 1:nblocks[2]
            rowstart = (i-1) * nrows + 1
            rowend = rowstart + nrows - 1
            colstart = (j-1) * ncols + 1
            colend = colstart + ncols - 1
            B[rowstart:rowend, colstart:colend] = A[i,j]
        end
    end
    return B
end

function solve_individuals_problem(r)

    #######
    # VFI #
    #######

    tol = 10^(-5); # Tolerance for the distance between elements of the value function.
    V_prev = ones(length(a_grid), length(z_grid)); # Temporary vector for the value function iteration.
    policy = zeros(length(a_grid), length(z_grid)); # Vector that will store the policy function.
    policy_index = Int.(zeros(length(a_grid), length(z_grid))); # Vector that will store the indexes of the policy function.
    values = zeros(length(a_grid), length(z_grid)); # Vector that will store the Bellman equation values during the brute-force grid-search. 
    V = zeros(length(a_grid), length(z_grid)); # Initial guess for the value function.
    iter = 0; # We will use this variable to count the number of iterations.
    maxiter = 100000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.

    c_matrix = [exp(z_grid[i]) .+ (1+r)*a_grid[j] .- a_grid[k] for j in 1:length(a_grid), i in 1:length(z_grid), k in 1:length(a_grid)] # A three-dimensional array of consumptions, containing all possible consumption values, for all possible combinations of k, z and k'.
    u_matrix = u.(c_matrix) # A three-dimensional array of utility values. All possible values for the utility function, for all possible combinations of k, z and k'. 
    
    @time begin 
        
        while maximum(abs.(V_prev - V)) > tol && iter < maxiter 
                
            V_prev = copy(V); 
    
            EV = [Π[j,:]' * V_prev[n,:] for n in 1:length(a_grid), j in 1:length(z_grid)] # A matrix of expected values for each possible combination of k' and z.
    
            # We start the iterative process.
            @threads for j in 1:length(z_grid) # Given z...
                @threads for i in 1:length(a_grid) # Given k...
                    value = u_matrix[i,j,:] + β * EV[:,j] # All possible values for the Bellman equation, given k and z.
                    V[i,j] = maximum(value); # Takes the maximum value. 
                    policy_index[i,j] = argmax(value) # Capital policy function indexes. Will be necessary for calculating Euler Equation Errors in the future. 
                    policy[i,j] = a_grid[policy_index[i,j]] # Capital policy function.
                end
            end 
    
            iter += 1; # Adds one to the iteration counter.
            # print("\n", "Iter: ", iter)
            # print("\n", "Distance: ", maximum(abs.(V_prev - V)))
        end
        # print("Total iterations: ", iter, "\n")
    end

    unstacked_x = [[a_grid[i], z_grid[h]] for h in eachindex(z_grid), i in eachindex(a_grid)] # Unstacked
    x = vec(unstacked_x) # Stacked 

    # Induced Markov Chain: 
    indicator(ap, a, s) = 1*(a_grid[ap] == policy[a,s])
    print("Computing the induced Markov chain... \n")
    Π_x = dropzeros(sparse(blockmatrix_to_matrix([[indicator(ap, a, s)*Π[s, sp] for a in eachindex(a_grid), ap in eachindex(a_grid)] for s in eachindex(z_grid), sp in eachindex(z_grid)])))

    # Obtaining stationary distribution: 
    # print("Obtaining the eigenvalues and eigenvectors... \n")
    # λ,v = eigs(Π_x', maxiter=100000000) # Eigenvalues and eigenvectors
    # unit_eigv = argmin(abs.(real(λ) .- 1)) # Unit eigenvalue position
    print("Obtaining the eigenvector... \n")
    v = [1; (I - Π_x'[2:end, 2:end]) \ Vector(Π_x'[2:end, 1])] # Directly solving 
    # norm_π = real.(v[:,unit_eigv])./sum(real.(v[:,unit_eigv])) # Normalized distribution
    norm_π = v./sum(v)
    π_∞ = reshape(norm_π, length(a_grid), length(z_grid)) # stationary distribution.
    # print("Computing the marginal asset distribution... \n")
    # Marginal asset distribution: 
    mg_π = zeros(length(a_grid))

    for i in eachindex(z_grid)
        mg_π = mg_π + π_∞[:, i]
    end 

    agg_savings = norm_π'vec(policy);
    print("Aggregate savings: ", agg_savings, "\n")
    return agg_savings, V, policy, policy_index, π_∞, mg_π
end

# Testing usign the risk-free rate:
solve_individuals_problem(0.0416)
solve_individuals_problem(0.0415233) # 0.0415233 -> 4.964730415707952e-5 

@time begin
    eq_r = find_zero(solve_individuals_problem, (0.04, 1/β - 1), Bisection())
end  # Não sai do lugar... Não para de rodar... Acredito que o problema esteja sendo na obteção dos autovalores em determinado momento.

# @time begin
#    eq_r = find_zero(solve_individuals_problem, (0.04, 0.0416), Bisection())
# end 

eq_r

# 1508 seconds with brackets (0.04, 0.0416), super close to zero. Converged to: 0.0415231584280692
# 1618.502569 seconds. Using full brackets and directly solving for the eigenvector, converged to r = 0.04152402868135286.

# Recovering equilibrium objects: 

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(eq_r)

# Value function plot 

plot(a_grid, V)

# Asset policy function plot 

plot(a_grid, policy)

# Consumption policy function plot 

# Recovering the consumption policy function: 
policy_c = [exp(z_grid[i]) + (1+eq_r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)] 

plot(a_grid, policy_c) 

# Invariant distribution: 

plot(repeat(a_grid,1,9), repeat(z_grid',201), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface,)

# Marginal asset distribution: 

plot(a_grid, π_∞)

plot(a_grid, mg_π) # Marginal asset distribution plot.
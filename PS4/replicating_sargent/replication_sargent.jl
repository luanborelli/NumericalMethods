#########################################################
## Replicating Sargent's 18.7. computed examples       ##
#########################################################

# Figs 18.6.3 and 18.7.2 approximately


###############################
## Importing useful packages ##
###############################

using Plots, Distributions, Random, Base.Threads, LinearAlgebra, Roots, SparseArrays

##################################################################
## Defining functions to be used in the entire rest of the code ## 
##################################################################

#######################################
## Tauchen's discretization function ##
#######################################

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

# The following function takes a block matrix as input and returns the equivalent non-partioned matrix: 

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

# Given an interest rate 'r' as input, the following function solves the individuals' problem and returns
# i) 'agg_savings': the aggregate savings of the economy;
# ii) 'V': the value function;
# iii) 'policy': the asset policy function; 
# iv) 'policy_index': the argmins associated with the asset policy function; 
# v) 'π_∞': the stationary distribution; 
# vi) 'mg_π': the marginal asset distribution.

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

    #= Vectorized
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
    =# 

    n_h = 10 # Number of "additional iterations", without maximization, that will be performed using the approximated
    # policy function, after each brute force maximization. I set n_h = 10, as suggested by the exercise. 
    # Setting n_h = 10 can be understood as "only performing the maximization part for 10% of the iterations".
    
    @time begin
        iter = 0
        while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Note that we measure the distance using the sup norm.
            # The code will stop when the distance is less than the defined tolerance, tol.
            # Additionally, we enforce an iteration limit to avoid the possibility of infinite loops.
    
            V_prev = copy(V); # Sets the value function to be the value function obtained in the last iteration.
            # Note that in the first iteration V_prev will therefore be the initial guess for V. 
    
            @threads for j in 1:length(z_grid) # The iterative process is initiated.
                k = 1
                for i in 1:length(a_grid) 
                        v_max = u(exp(z_grid[j]) + (1+r)*a_grid[i] - a_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(exp(z_grid[j]) + (1+r)*a_grid[i] - a_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:] 
                        while next_v_max > v_max && k < length(a_grid) - 1
                            k += 1  
                            v_max = u(exp(z_grid[j]) + (1+r)*a_grid[i] - a_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                            next_v_max = u(exp(z_grid[j]) + (1+r)*a_grid[i] - a_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                        end  
                        V[i, j] = v_max
                        policy[i,j] = a_grid[k]
                        policy_index[i,j] = k
                end
            end # End of the iterative process. 
    
            iter += 1; 
            # print("\n", "Iter: ", iter)
            # print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    
            for h in 1:n_h # Acceleration starts here. We perform n_h additional iterations *without maximization*, using the previously approximated policy function.
                V_prev = copy(V)
                @threads for j in 1:length(z_grid) 
                    @threads for i in 1:length(a_grid)
                        V[i, j] = u(exp(z_grid[j]) + (1+r)*a_grid[i] - policy[i,j]) + β * Π[j,:]' * V_prev[trunc(Int, policy_index[i,j]),:] 
                    end
                end
                iter += 1; 
                # print("\n", "Iter: ", iter)
                # print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
            end
    
        end
        # print("Total iterations: ", iter, "\n")
    end

    unstacked_x = [[a_grid[i], z_grid[h]] for h in eachindex(z_grid), i in eachindex(a_grid)] # Unstacked
    x = vec(unstacked_x) # Stacked 

    # Induced Markov Chain: 
    indicator(ap, a, s) = 1*(a_grid[ap] == policy[a,s])
    print("Computing the induced Markov chain... \n")
    Π_x = dropzeros(sparse(blockmatrix_to_matrix([[indicator(ap, a, s)*Π[s, sp] for a in eachindex(a_grid), ap in eachindex(a_grid)] for s in eachindex(z_grid), sp in eachindex(z_grid)])))

    print("Obtaining the eigenvector... \n")
    v = [1; (I - Π_x'[2:end, 2:end]) \ Vector(Π_x'[2:end, 1])] # Directly solving 
    norm_π = v./sum(v)

    print("Computing the stationary distribution... \n")
    π_∞ = reshape(norm_π, length(a_grid), length(z_grid)) # stationary distribution.

    # Marginal asset distribution: 
    mg_π = zeros(length(a_grid))

    for i in eachindex(z_grid)
        mg_π = mg_π + π_∞[:, i]
    end 

    agg_savings = norm_π'vec(policy);
    print("Aggregate savings: ", agg_savings, "\n")
    return agg_savings, V, policy, policy_index, π_∞, mg_π
end

# Parameters: 

β = 0.96;
γ = 3;
ρ = 0.2;
σ = 0.4;
N_z = 7; 
m = 3;

# Discretizing z: 
z_grid, Π = tauchen(0, σ^2, ρ, N_z, m);

# Utility function: 

function u(c) 
    if c > 0
        u = (c^(1-γ) - 1)/(1-γ); # Utility function
    else 
        u = -Inf # This is necessary to avoid negative consumption. If c < 0, utility = -∞.
    end
    return u
end

z_min = z_grid[1]
z_max = z_grid[N_z]

r = 1/β - 1; # Risk-free rate. 
ϕ = exp(z_min)/r; # Natural debt limit.

# Generating the asset grid: 

grid_size = 500;
a_min = -3;
a_max = 16;
a_grid = range(a_min, a_max, 200); # Note that a small positive perturbation to the lower bound is required so that agents are not able to borrow infinitely.

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1, 300)

@time begin 
    Ears = solve_individuals_problem.(grid_r)
end 

@time begin 
    Ears_2 = solve_individuals_problem.(grid_r)
end 

Ear = [Ears[i][1] for i in eachindex(grid_r)]
Ear_2 = [Ears_2[i][1] for i in eachindex(grid_r)]
b1 = [-3 for i in eachindex(grid_r)]
b2 = [-6 for i in eachindex(grid_r)]
plot(Ear, grid_r)
plot!(Ear_2, grid_r)
plot!(b1, grid_r, color="black", linestyle=:dash)
plot!(b2, grid_r, color="black", linestyle=:dash)
hline!([1/β - 1], color="red", linestyle=:dash)
vline!([0], color="black", linestyle=:dash)
# savefig("fig1863.png")

########## Eq.:

@time begin
    r = find_zero(solve_individuals_problem, (0, 1/β - 1), Bisection())
end  


r

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r)

# Value function plot 
plot(a_grid, V)

# Asset policy function plot 
plot(a_grid, policy)

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function
plot(a_grid, policy_c) 

# Invariant distribution: 

plot(repeat(a_grid,1,7), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface,)


# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution
plot(a_grid, mg_π) # Marginal asset distribution plot.
savefig("fig1872.png")
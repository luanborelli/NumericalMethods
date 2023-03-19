#########################################
## Problem Set 4 - Numerical Methods   ##
## Incomplete Markets                  ##
## Student: Luan Borelli               ##
#########################################

###############################
## Importing useful packages ##
###############################

using Plots, Distributions, Random, Base.Threads

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

r = 1/β - 1; # Risk-free rate. 
ϕ = exp(z_min)/r; # Natural debt limit.

# Generating the asset grid: 

grid_size = 500;             
a_min = -ϕ;
a_max = +ϕ;
a_grid = range(a_min + 10e-5, a_max, 500);

## Technical values

tol = 10^(-5); # Tolerance for the distance between elements of the value function.
# We will use this tolerance to define when to stop the iterative process.
# tol defines, therefore, the distance from which we consider that the elements of the value function
# are "close enough". Remember that the sequence of V's is a Cauchy sequence.

iter = 0; # We will use this variable to count the number of iterations.
maxiter = 100000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.
V_prev = ones(length(a_grid), length(z_grid)); # Temporary vector for the value function iteration.
policy = zeros(length(a_grid), length(z_grid)); # Vector that will store the policy function.
policy_index = Int.(zeros(length(a_grid), length(z_grid))); # Vector that will store the indexes of the policy function.
values = zeros(length(a_grid), length(z_grid)); # Vector that will store the Bellman equation values during the brute-force grid-search. 

V = zeros(length(a_grid), length(z_grid)); # Initial guess for the value function. 

# Provavelmente eu tenho que chutar uma taxa de juros, resolver, checar market clearing,
# ajustar taxa de juros, resolver dnv, checar... até dar mkt clearing.

@time begin # To compute the running time.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Note that we measure the distance using the sup norm.
    # The code will stop when the distance is less than the defined tolerance, tol.
    # Additionally, we enforce an iteration limit to avoid the possibility of infinite loops.
            
        V_prev = copy(V); # Sets the value function to be the value function obtained in the last iteration.
        # Note that in the first iteration V_prev will therefore be the initial guess for V.

        @threads for j in 1:length(z_grid) # The iterative process is initiated.
            for i in 1:length(a_grid) 
                @threads for n in 1:length(a_grid)
                    values[n,j] = u(exp(z_grid[j]) + (1+r)*a_grid[i] - a_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                end
                V[i, j] = maximum(values[:, j])
                policy_index[i, j] = argmax(values[:, j])
                pos = argmax(values[:, j])
                policy[i,j] = a_grid[pos]
            end
        end # End of the iterative process. 

        iter += 1; # Adds one to the iteration counter.
        print("\n", "Iter: ", iter) # Prints the current iteration. 
        print("\n", "Distance: ", maximum(abs.(V_prev - V))) # Prints the distance between the current and the previous value function.
    end
end


# Recovering the consumption policy function: 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)] 


#######
## c ##
####### 

# See pg. 788 (827 pdf), recursive methods. 
# See pg. 33 (pdf 72), recursive methods. 
# Notice that theorems 2.2.1 and 2.2.2 doesn't hold here. 
# We have 0 entries... 

# x: 
x = vec([[a_grid[i], z_grid[h]] for h in eachindex(z_grid), i in eachindex(a_grid)])

# Induced Markov Chain for x: 

indicator = [a_grid[i] == policy[n,s] for i in eachindex(a_grid), n in eachindex(a_grid), s in eachindex(z_grid)]

# Markov chain for x: 
 # Construir... 

 # Then obtain the stationary distribution following section 2.2.1 (Recursive methods...)

# Having the stationary distribution, we obtain the aggregate savings 


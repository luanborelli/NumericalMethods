#########################################
## Problem Set 4 - Numerical Methods   ##
## Incomplete Markets                  ##
## Student: Luan Borelli               ##
#########################################

###############################
## Importing useful packages ##
###############################

using Plots, Distributions, Random, Base.Threads, LinearAlgebra, Roots, SparseArrays, Latexify

###################################################################
## Defining functions to be used throughout the rest of the code ## 
###################################################################

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


# Given an interest rate 'r' as input, the following function solves the individuals' problem and returns
# i) 'agg_savings': the aggregate savings of the economy;
# ii) 'V': the value function;
# iii) 'policy': the asset policy function; 
# iv) 'policy_index': the argmins associated with the asset policy function; 
# v) 'π_∞': the stationary distribution; 
# vi) 'mg_π': the marginal asset distribution.

function solve_individuals_problem(r)

    ######################################################################
    # VFI: Exploiting monotonicity + concavity and using the accelerator #
    ######################################################################

    tol = 10^(-5); # Tolerance for the distance between elements of the value function.
    V_prev = ones(length(a_grid), length(z_grid)); # Temporary vector for the value function iteration.
    policy = zeros(length(a_grid), length(z_grid)); # Vector that will store the policy function.
    policy_index = Int.(zeros(length(a_grid), length(z_grid))); # Vector that will store the indexes of the policy function.
    values = zeros(length(a_grid), length(z_grid)); # Vector that will store the Bellman equation values during the brute-force grid-search. 
    # V = zeros(length(a_grid), length(z_grid)); # Initial guess for the value function.
    V = [u(r*a_grid[i] + exp(z_grid[j]))./(1-β) for i in eachindex(a_grid), j in eachindex(z_grid)] # Moll's initial guess

    iter = 0; # We will use this variable to count the number of iterations.
    maxiter = 100000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.

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
        print("Total iterations: ", iter, "\n")
    end
 
    # Finding the stationary distribution. See L&S RMT pg. 788 for details.

    # Mapping (a, s) states into a single state vector, x. In practice, it will not be necessary. Just replicating what is written in the book for a better understanding.
    # unstacked_x = [[a_grid[i], z_grid[h]] for h in eachindex(z_grid), i in eachindex(a_grid)]; # Unstacked
    # x = vec(unstacked_x); # Stacked 

    # Induced Markov Chain (the optimal policy function a' = g(a, s) and the Markov chain P on s induce a Markov chain for x via the formula I(a', a, s)Π(s, s'). 
    print("Computing the induced Markov chain... \n")
    Πs = kron(Π,ones(length(a_grid),1));
    indic = vcat([(a_grid .== policy[i]) for i in 1:4500]'...) # Indicator 
    Π_x = sparse(vcat([kron(Πs[s,:], indic[s,:]) for s in 1:4500]'...)) # Induced Markov chain

    # The stationary distribution is given by the eigenvector of Π_x associated with its unit eigenvalue. There is a quick way to get this eigenvector directly: 
    print("Obtaining the eigenvector... \n")
    v = [1; (I - Π_x'[2:end, 2:end]) \ Vector(Π_x'[2:end, 1])]; # Directly solving 
    norm_π = v./sum(v); # Normalizing to add 1.

    print("Computing the stationary distribution... \n")
    π_∞ = reshape(norm_π, length(a_grid), length(z_grid)); # Stationary distribution.

    # Marginal asset distribution: 
    mg_π = zeros(length(a_grid));

    for i in eachindex(z_grid)
        mg_π = mg_π + π_∞[:, i];
    end 

    # Aggregate savings: 
    agg_savings = norm_π'vec(policy);
    print("Aggregate savings: ", agg_savings, "\n")

    return agg_savings, V, policy, policy_index, π_∞, mg_π
end


########################################################################################################################################## 
##########################################################################################################################################

#######
## a ##
#######
 
# Parameters: 
ρ = 0.9;
σ = 0.01;
N_z = 9; 
m = 3;

# Discretizing z: 
z_grid, Π = tauchen(0, σ^2, ρ, N_z, m);

# latexify(round.(Π, digits = 3)) |> print
# latexify(round.(z_grid, digits = 3)) |> print

########################################################################################################################################## 
##########################################################################################################################################

#######
## b ##
#######

# Parameters: 

β = 0.96;
γ = 1.0001;
ρ = 0.9;
σ = 0.01;
N_z = 9; 
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
a_min = -ϕ;
a_max = +ϕ;
a_grid = range(a_min + 10e-9, a_max, 500); # Note that a small positive perturbation to the lower bound is required so that agents are not able to borrow infinitely.

@time begin 
    agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r);
end 

# Value function plot
V_2d_b = plot(a_grid[2:end,:], V[2:end,:], 
        xlab = "a", 
        ylab = "V", 
        label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
        legend = :outertopright) # 2D plot.

# savefig(V_2d_b, "V_2d_b.pdf")

V_3d_b = plot(repeat(a_grid[2:end],1,length(z_grid)), repeat(z_grid',length(a_grid)-1), V[2:end, :],
    xlab = "a", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(35,35)) # 3D plot.

# savefig(V_3d_b, "V_3d_b.pdf")

# Asset policy function plot 
a_2d_b = plot(a_grid, policy, 
    xlab = "a", 
    ylab = "V", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright)

# savefig(a_2d_b, "a_2d_b.pdf")

a_3d_b = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy,
    xlab = "a", 
    ylab = "z", 
    zlab = "a'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# savefig(a_3d_b, "a_3d_b.pdf")

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function

c_2d_b = plot(a_grid, policy_c,
    xlab = "a", 
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) 

# savefig(c_2d_b, "c_2d_b.pdf")

c_3d_b = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy_c,
    xlab = "a", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

# savefig(c_3d_b, "c_3d_b.pdf")

# Invariant distribution: 

pi_3d_b = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface) # 3D plot.

# savefig(pi_3d_b, "pi_3d_b.pdf")

# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution

mg_pi_b = plot(a_grid, mg_π,
    xlab = "a",
    ylab = "Marginal asset distribution",
    legend = false) # Marginal asset distribution plot.

# savefig(mg_pi_b, "mg_pi_b.pdf")
 
## Euler Equation Errors

# This function calculates the Euler Error for a given a (index) and a given z (index): 
EEE(a_ind, z_ind) = log10(abs(1-((1+r)*β*(Π[z_ind,:]' * (policy_c[policy_index[a_ind,z_ind],:].^(-γ))))^(-1/γ)/(policy_c[a_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of a's and z's:
EEEs = [EEE(i,j) for i in 1:length(a_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(a_grid, EEEs,
    xlab = "a",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) # 2D plot.

# savefig(eees_b, "eees_b.pdf")

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1 - 10^-9, 500)

@time begin 
    Ears_b = solve_individuals_problem.(grid_r)
end 

Ear_b = [Ears_b[i][1] for i in eachindex(grid_r)]
Ear_plt_b = plot(Ear_b, grid_r, ylims=(0.035, 0.042))


########################################################################################################################################## 
##########################################################################################################################################

#######
## c ##
####### 

# @time begin # Wide brackets.
#     r = find_zero(solve_individuals_problem, (0, 1/β - 1), Bisection())
# end  

@time begin # Close brackets.
  r = find_zero(solve_individuals_problem, (0.04152, 0.04153), Bisection())
end 

# 678.477271 seconds. r = 0.04152402867818983. Using brackets (0.0415, 0.0416).
# 1618.502569 seconds. Using full brackets and directly solving for the eigenvector, converged to r = 0.04152402868135286.
# 82 seconds. Using brackets (0.04152, 0.04153). Converged to 0.04152388631194748.

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r)

# Value function plot
V_2d_c = plot(a_grid[2:end,:], V[2:end,:], 
        xlab = "a", 
        ylab = "V", 
        label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
        legend = :outertopright) # 2D plot.

# savefig(V_2d_c, "V_2d_c.pdf")

V_3d_c = plot(repeat(a_grid[2:end],1,length(z_grid)), repeat(z_grid',length(a_grid)-1), V[2:end, :],
    xlab = "a", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(35,35)) # 3D plot.

# savefig(V_3d_c, "V_3d_c.pdf")

# Asset policy function plot 
a_2d_c = plot(a_grid, policy, 
    xlab = "a", 
    ylab = "a'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright)

# savefig(a_2d_c, "a_2d_c.pdf")

a_3d_c = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy,
    xlab = "a", 
    ylab = "z", 
    zlab = "a'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# savefig(a_3d_c, "a_3d_c.pdf")

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function

c_2d_c = plot(a_grid, policy_c,
    xlab = "a", 
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) 

# savefig(c_2d_c, "c_2d_c.pdf")

c_3d_c = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy_c,
    xlab = "a", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

# savefig(c_3d_c, "c_3d_c.pdf")

# Invariant distribution: 

pi_3d_c = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface) # 3D plot.

# savefig(pi_3d_c, "pi_3d_c.pdf")

# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution

mg_pi_c = plot(a_grid, mg_π,
    xlab = "a",
    ylab = "Marginal asset distribution",
    legend = false) # Marginal asset distribution plot.

# savefig(mg_pi_c, "mg_pi_c.pdf")

## Euler Equation Errors

# This function calculates the Euler Error for a given a (index) and a given z (index): 
EEE(a_ind, z_ind) = log10(abs(1-((1+r)*β*(Π[z_ind,:]' * (policy_c[policy_index[a_ind,z_ind],:].^(-γ))))^(-1/γ)/(policy_c[a_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of a's and z's:
EEEs = [EEE(i,j) for i in 1:length(a_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(a_grid, EEEs,
    xlab = "a",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) # 2D plot.

# savefig(eees_c, "eees_c.pdf")

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1 - 10^-9, 500)

@time begin 
    Ears_c = solve_individuals_problem.(grid_r)
end 

Ear_c = [Ears_c[i][1] for i in eachindex(grid_r)]
Ear_plt_c = plot(Ear_c, grid_r, ylims=(0.035, 0.042))

########################################################################################################################################## 
##########################################################################################################################################

#######
## d ##
####### 

# Parameters: 

β = 0.96;
γ = 1.0001;
ρ = 0.97;
σ = 0.01;
N_z = 9; 
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
a_min = -ϕ;
a_max = +ϕ;
a_grid = range(a_min + 10e-9, a_max, 500); # Note that a small positive perturbation to the lower bound is required so that agents are not able to borrow infinitely.

# @time begin
#     r = find_zero(solve_individuals_problem, (0, 1/β - 1), Bisection())
# end  

@time begin
   r = find_zero(solve_individuals_problem, (0.0414, 0.0415), Bisection())
end  

# Converged in 1922.523141 seconds. r = 0.04140454634421464
# Acelerei para... 1624 segundos, calculando autovetor direto.
# 681.599172 seconds. r = 0.04140454634878677. Brackets: (0.0414, 0.0415)
# 121 seconds. r = 0.04140437689344434. Brackets: (0.0414, 0.0415)

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r)

# Value function plot
V_2d_d = plot(a_grid[2:end,:], V[2:end,:], 
        xlab = "a", 
        ylab = "V", 
        label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
        legend = :outertopright) # 2D plot.

# savefig(V_2d_d, "V_2d_d.pdf")

V_3d_d = plot(repeat(a_grid[2:end],1,length(z_grid)), repeat(z_grid',length(a_grid)-1), V[2:end, :],
    xlab = "a", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(35,35)) # 3D plot.

# savefig(V_3d_d, "V_3d_d.pdf")

# Asset policy function plot 
a_2d_d = plot(a_grid, policy, 
    xlab = "a", 
    ylab = "a'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright)

# savefig(a_2d_d, "a_2d_d.pdf")

a_3d_d = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy,
    xlab = "a", 
    ylab = "z", 
    zlab = "a'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# savefig(a_3d_d, "a_3d_d.pdf")

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function

c_2d_d = plot(a_grid, policy_c,
    xlab = "a", 
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) 

# savefig(c_2d_d, "c_2d_d.pdf")

c_3d_d = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy_c,
    xlab = "a", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

# savefig(c_3d_d, "c_3d_d.pdf")

# Invariant distribution: 

pi_3d_d = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface) # 3D plot.

# savefig(pi_3d_d, "pi_3d_d.pdf")

# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution

mg_pi_d = plot(a_grid, mg_π,
    xlab = "a",
    ylab = "Marginal asset distribution",
    legend = false) # Marginal asset distribution plot.

# savefig(mg_pi_d, "mg_pi_d.pdf")

## Euler Equation Errors

# This function calculates the Euler Error for a given a (index) and a given z (index): 
EEE(a_ind, z_ind) = log10(abs(1-((1+r)*β*(Π[z_ind,:]' * (policy_c[policy_index[a_ind,z_ind],:].^(-γ))))^(-1/γ)/(policy_c[a_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of a's and z's:
EEEs = [EEE(i,j) for i in 1:length(a_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
eees_d = plot(a_grid, EEEs,
    xlab = "a",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) # 2D plot.

# savefig(eees_d, "eees_d.pdf")

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1 - 10^-9, 500)

@time begin 
    Ears_c = solve_individuals_problem.(grid_r)
end 

Ear_c = [Ears_c[i][1] for i in eachindex(grid_r)]
Ear_plt_c = plot(Ear_c, grid_r, ylims=(0.035, 0.042))

########################################################################################################################################## 
##########################################################################################################################################

#######
## e ##
####### 

# Parameters: 

β = 0.96;
γ = 5;
ρ = 0.9;
σ = 0.01;
N_z = 9; 
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
a_min = -ϕ;
a_max = +ϕ;
a_grid = range(a_min + 10e-5, a_max, 500); # Note that a small positive perturbation to the lower bound is required so that agents are not able to borrow infinitely.

# @time begin
#    r = find_zero(solve_individuals_problem, (0, 1/β - 1), Bisection())
# end

@time begin
    r = find_zero(solve_individuals_problem, (0.0405, 0.041), Bisection())
end

# 2161.348826 seconds. Converged to r = 0.04090359551256151.
# 255 seconds. Brackets: (0.0405, 0.041). Converged to r = 0.04090359512950348. 

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r)

# Value function plot
V_2d_e = plot(a_grid[50:end,:], V[50:end,:], 
        xlab = "a", 
        ylab = "V", 
        label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
        legend = :outertopright) # 2D plot.

# savefig(V_2d_e, "V_2d_e.pdf")

V_3d_e = plot(repeat(a_grid[50:end],1,length(z_grid)), repeat(z_grid',length(a_grid)-49), V[50:end, :],
    xlab = "a", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(35,35)) # 3D plot.

# savefig(V_3d_e, "V_3d_e.pdf")

# Asset policy function plot 
a_2d_e = plot(a_grid, policy, 
    xlab = "a", 
    ylab = "a'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright)

# savefig(a_2d_e, "a_2d_e.pdf")

a_3d_e = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy,
    xlab = "a", 
    ylab = "z", 
    zlab = "a'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# savefig(a_3d_e, "a_3d_e.pdf")

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function

c_2d_e = plot(a_grid, policy_c,
    xlab = "a", 
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) 

# savefig(c_2d_e, "c_2d_e.pdf")

c_3d_e = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy_c,
    xlab = "a", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

# savefig(c_3d_e, "c_3d_e.pdf")

# Invariant distribution: 

pi_3d_e = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface) # 3D plot.

# savefig(pi_3d_e, "pi_3d_e.pdf")

# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution

mg_pi_e = plot(a_grid, mg_π,
    xlab = "a",
    ylab = "Marginal asset distribution",
    legend = false) # Marginal asset distribution plot.

# savefig(mg_pi_e, "mg_pi_e.pdf")

## Euler Equation Errors

# This function calculates the Euler Error for a given a (index) and a given z (index): 
EEE(a_ind, z_ind) = log10(abs(1-((1+r)*β*(Π[z_ind,:]' * (policy_c[policy_index[a_ind,z_ind],:].^(-γ))))^(-1/γ)/(policy_c[a_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of a's and z's:
EEEs = [EEE(i,j) for i in 1:length(a_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
eees_e = plot(a_grid, EEEs,
    xlab = "a",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) # 2D plot.

# savefig(eees_e, "eees_e.pdf")

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1 - 10^-9, 500)

@time begin 
    Ears_e = solve_individuals_problem.(grid_r)
end 

Ear_e = [Ears_e[i][1] for i in eachindex(grid_r)]
Ear_plt_e = plot(Ear_e, grid_r, ylims=(0.035, 0.042))

########################################################################################################################################## 
##########################################################################################################################################

#######
## f ##
####### 

# Parameters: 

β = 0.96;
γ = 1.0001;
ρ = 0.9;
σ = 0.05;
N_z = 9; 
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
a_min = -ϕ;
a_max = +ϕ;
a_grid = range(a_min + 10e-5, a_max, 500); # Note that a small positive perturbation to the lower bound is required so that agents are not able to borrow infinitely.

# @time begin
#     r = find_zero(solve_individuals_problem, (0, 1/β - 1), Bisection())
# end

@time begin
   r = find_zero(solve_individuals_problem, (0.04, 0.041), Bisection())
end

# 1679.900198 seconds. Converged to r = 0.04088091638701418.
# 1104.687970 seconds. Converged to r = 0.040880916387370125. Brackets: (0.04, 0.041)
# 131 seconds. Converged to r = 0.04088093653876845. Brackets: (0.04, 0.041)

agg_savings, V, policy, policy_index, π_∞, mg_π = solve_individuals_problem(r)

# Value function plot
V_2d_f = plot(a_grid[1:end,:], V[1:end,:], 
        xlab = "a", 
        ylab = "V", 
        label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
        legend = :outertopright) # 2D plot.

# savefig(V_2d_f, "V_2d_f.pdf")

V_3d_f = plot(repeat(a_grid[1:end],1,length(z_grid)), repeat(z_grid',length(a_grid)), V[1:end, :],
    xlab = "a", 
    ylab = "z", 
    zlab = "V",
    seriestype=:surface, 
    camera=(35,35)) # 3D plot.

# savefig(V_3d_f, "V_3d_f.pdf")

# Asset policy function plot 
a_2d_f = plot(a_grid, policy, 
    xlab = "a", 
    ylab = "a'", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright)

# savefig(a_2d_f, "a_2d_f.pdf")

a_3d_f = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy,
    xlab = "a", 
    ylab = "z", 
    zlab = "a'",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot.

# savefig(a_3d_f, "a_3d_f.pdf")

# Consumption policy function plot 
policy_c = [exp(z_grid[i]) + (1+r)*a_grid[j] - policy[j,i] for j in 1:length(a_grid), i in 1:length(z_grid)]  # Recovering the consumption policy function

c_2d_f = plot(a_grid, policy_c,
    xlab = "a", 
    ylab = "c", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) 

# savefig(c_2d_f, "c_2d_f.pdf")

c_3d_f = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), policy_c,
    xlab = "a", 
    ylab = "z", 
    zlab = "c",
    seriestype=:surface, 
    camera=(20,40)) # 3D plot. 

# savefig(c_3d_f, "c_3d_f.pdf")

# Invariant distribution: 

pi_3d_f = plot(repeat(a_grid,1,length(z_grid)), repeat(z_grid',length(a_grid)), π_∞,
    xlab = "a", 
    ylab = "z", 
    zlab = "π",
    seriestype=:surface) # 3D plot.

# savefig(pi_3d_f, "pi_3d_f.pdf")

# Marginal asset distribution: 
plot(a_grid, π_∞) # Stationary distribution

mg_pi_f = plot(a_grid, mg_π,
    xlab = "a",
    ylab = "Marginal asset distribution",
    legend = false) # Marginal asset distribution plot.

# savefig(mg_pi_f, "mg_pi_f.pdf")

## Euler Equation Errors

# This function calculates the Euler Error for a given a (index) and a given z (index): 
EEE(a_ind, z_ind) = log10(abs(1-((1+r)*β*(Π[z_ind,:]' * (policy_c[policy_index[a_ind,z_ind],:].^(-γ))))^(-1/γ)/(policy_c[a_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of a's and z's:
EEEs = [EEE(i,j) for i in 1:length(a_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
eees_f = plot(a_grid, EEEs,
    xlab = "a",
    ylab = "Euler Equation Error (EEE)", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7" "State 8" "State 9"], 
    legend = :outertopright) # 2D plot.

# savefig(eees_f, "eees_f.pdf")

# Some statistics on the EEEs: 
avg_EEE = mean(EEEs)
median_EEE = median(EEEs)
max_EEE = maximum(EEEs)
min_EEE = minimum(EEEs)

# Aggregate savings plot (E[a(r)]):
grid_r = range(0, 1/β - 1 - 10^-9, 500)

@time begin 
    Ears_f = solve_individuals_problem.(grid_r)
end 

Ear_f = [Ears_f[i][1] for i in eachindex(grid_r)]
Ear_plt_f = plot(Ear_f, grid_r, ylims=(0.035, 0.042))
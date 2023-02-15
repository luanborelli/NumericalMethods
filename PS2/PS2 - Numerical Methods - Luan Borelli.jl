#########################################
## Problem Set 2 - Numerical Methods   ##
## Student: Luan Borelli               ##
#########################################

###############################
## Importing useful packages ##
###############################

using Distributions, Random, Plots, Base.Threads 

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
maxiter = 1000; # Maximum number of iterations. We will use this to avoid the possibility of an infinite loop.

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

# V = zeros(length(k_grid), length(z_grid)); # Initial guess for the value function. 
# Usually zero. But there are better guesses. For example, Moll's initial guess: 
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # A king of "consumption matrix", but setting k' = k. This matrix computes all possible consumptions for all possible combinations of k and z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # The guess, constructed from c.

#######
## 1 ##
#######

## Below I present several different methods for solving the model.
## All you need to do in order to obtain the results reported in the 
## LaTeX document is to uncomment (removing the '#=' and '=#') one 
## of the blocks below and then run the entire code.


##############################################################
# Solution #1: brute-force grid search without parallelizing #
##############################################################

#=
@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Note that we measure the distance using the sup norm.
    # The code will stop when the distance is less than the defined tolerance, tol.
    # Additionally, we enforce an iteration limit to avoid the possibility of infinite loops.
            
        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 
            
        for (j, z) in enumerate(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            for (i, k) in enumerate(k_grid) 
                for (n, kp) in enumerate(k_grid)
                    v_max = u(z*k^α + (1-δ)*k - kp) + β * Π[j,:]' * V_prev[n,:] 
                    if v_max > V[i, j]
                        V[i, j] = v_max
                        policy[i, j] = kp
                    end 
                end
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(V_prev - V)))
    end
end
=# 

#######################################################
# Solution #2: brute-force grid search, parallelizing #
#######################################################

# To parallelize for-loops, we cannot use multiple iterators.
# Therefore, we must rewrite the code without using enumerate(.).

# Obs.: note that for the parallelization to work it is necessary to run Julia
# declaring the number of processor threads to be used:
#
# E.g.: julia --threads=auto
# "auto" sets the number of threads to the number of local CPU threads.
#
# In VSCode: go to Julia extension settings for VSCode and set
# in settings.json:
#
# "julia.NumThreads": "auto"
#
# This will make VSCode always run Julia using the maximum number of processor threads.


#= 

@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Note that we measure the distance using the sup norm.
    # The code will stop when the distance is less than the defined tolerance, tol.
    # Additionally, we enforce an iteration limit to avoid the possibility of infinite loops.
            
        V_prev = copy(V); # Sets the value function to be the value function obtained in the last iteration.
        # Note that in the first iteration V_prev will therefore be the initial guess for V.

        @threads for j in 1:length(z_grid) # The iterative process is initiated.
            @threads for i in 1:length(k_grid) 
                @threads for n in 1:length(k_grid)
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                    if v_max > V[i, j]
                        V[i, j] = v_max
                        policy[i, j] = k_grid[n]
                    end 
                end
            end
        end # End of the iterative process. 

        #iter += 1; # Adds one to the iteration counter.
        #print("\n", "Iter: ", iter) # Prints the current iteration. 
        #print("\n", "Distance: ", maximum(abs.(V_prev - V))) # Prints the distance between the current and the previous value function.
    end
end

=# 

####################################################
## Solution #3: vectorized, without parallelizing ##
####################################################

#= 

# Matriz 3D de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k, z e k'.
c_matrix = [z_grid[i]*(k_grid[j]^α).-k_grid[k].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid), k in 1:length(k_grid)]
u_matrix = u.(c_matrix) # Matriz 3D de utilidades 

@time begin 
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter 
            
        V_prev = copy(V); 

        EV = [Π[j,:]' * V_prev[n,:] for n in 1:length(k_grid), j in 1:length(z_grid)] # Matriz de valores esperados, para cada combinação possível de k' e z.

        for j in 1:length(z_grid) # Iniciamos o processo iterativo.
            for i in 1:length(k_grid) 
                value = u_matrix[i,j,:] + β * EV[:,j]
                V[i,j] = maximum(value); # Maximiza a função valor usando V_prev.
                policy_index[i,j] = argmax(value) # Para calcular as Euler Equation Errors futuramente. 
                policy[i,j] = k_grid[policy_index[i,j]]
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        # print("\n", "Iter: ", iter)
        # print("\n", "Distance: ", maximum(abs.(V_prev - V)))
    end
    print("Total iterations: ", iter, "\n")
end

=# 

#############################################
## Solution #4: vectorized,  parallelizing ##
#############################################

#= 
c_matrix = [z_grid[i]*(k_grid[j]^α).-k_grid[k].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid), k in 1:length(k_grid)]
u_matrix = u.(c_matrix) # Matriz 3D de utilidades 

@time begin 
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter 
            
        V_prev = copy(V); 

        EV = [Π[j,:]' * V_prev[n,:] for n in 1:length(k_grid), j in 1:length(z_grid)] # Matriz de valores esperados, para cada combinação possível de k' e z.

        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo.
            @threads for i in 1:length(k_grid) 
                value = u_matrix[i,j,:] + β * EV[:,j]
                V[i,j] = maximum(value); # Maximiza a função valor usando V_prev.
                policy_index[i,j] = argmax(value) # Para calcular as Euler Equation Errors futuramente. 
                policy[i,j] = k_grid[policy_index[i,j]] # Capital policy function.
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        # print("\n", "Iter: ", iter)
        # print("\n", "Distance: ", maximum(abs.(V_prev - V)))
    end
    print("Total iterations: ", iter, "\n")
end

=# 

#############################
## EXPLOITING MONOTONICITY ##
#############################

@time begin

    pos = 1
    iter = 0


    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.

        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração. 
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 

        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            pos = 1
            @threads for i in 1:length(k_grid) 
                @threads for n in pos:length(k_grid)
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                    if v_max > V[i, j]
                        V[i, j] = v_max
                        pos = n
                        policy[i, j] = k_grid[pos]
                    end 
                end
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        #print("\n", "Iter: ", iter)
        #print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    end
    print("Total iterations: ", iter, "\n")
end

##########################
## EXPLOITING CONCAVITY ##
##########################

#= 

@time begin
    pos = 1
    iter = 0
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração. 
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 
        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            pos = 1
            @threads for i in 1:length(k_grid)

                    k = 1 
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                    next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:] 

                    while next_v_max > v_max && k < length(k_grid) - 1  
                        v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                        k += 1
                    end  

                    V[i, j] = v_max
                    policy_index[i,j] = k
                    policy[i,j] = k_grid[k]
            end
        end 
        iter += 1; # Soma um ao contador de iterações
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    end
    print("Total iterations: ", iter, "\n")
end

=# 

###########
## Plots ##
###########

# Recovering consumption policy function: 
policy_c = [z_grid[i]*(k_grid[j]^α).-policy[j,i].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] 

# Plotting value function: 
plot(k_grid, V) # 2D plot.
plot(repeat(k_grid,1,7), repeat(z_grid',500), V, seriestype=:surface, camera=(20,40)) # 3D plot.

# Plotting capital policy function: 
plot(k_grid, policy) # 2D plot.
plot(repeat(k_grid,1,7), repeat(z_grid',500), policy, seriestype=:surface, camera=(20,40)) # 3D plot.

# Plotting consumption policy function: 
plot(k_grid, policy_c) # 2D plot.
plot(repeat(k_grid,1,7), repeat(z_grid',500), policy_c, seriestype=:surface, camera=(20,40)) # 3D plot. 

###########################
## Euler Equation Errors ##
###########################

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (policy_c[policy_index[k_ind,z_ind],:].^(-μ) .* (α*z_grid*policy[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(k_grid, EEEs)

#= To-do: 
[ ] Traduzir comentários;  
[ ] Formatar plots. 
=#
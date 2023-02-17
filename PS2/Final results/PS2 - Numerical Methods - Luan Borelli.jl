#########################################
## Problem Set 2 - Numerical Methods   ##
## Student: Luan Borelli               ##
#########################################

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
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # The guess, constructed from c.

#######
## 3 ##
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

##################################################
## Solution #5: Exploiting monotonicity (alone) ##
##################################################

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
            starting_pos = 1
            for i in 1:length(k_grid) 
                @threads for n in starting_pos:length(k_grid)
                    values[n,j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                end
                V[i, j] = maximum(values[:, j])
                policy_index[i, j] = argmax(values[:, j])
                starting_pos = argmax(values[:, j])
                policy[i,j] = k_grid[starting_pos]
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    end
    print("Total iterations: ", iter, "\n")
end

=# 

###############################################
## Solution #6: Exploiting concavity (alone) ##
###############################################

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

####################################################################
# Solution #7: Exploiting both concavity and monotonicity together #
####################################################################

# Notice that in order to exploit monotonicity together with concavity we only need to change one line of 
# code in the concavity algorithm. Indeed, this one line is the line that redefines k to 1. 
# Now we redefine it to 1 only at the z_grid loop, so that "k" counting is being accumulated during the k_grid for-loop. 
# This makes sense: in this way, for each k in 1:length(k_grid) the algorithm will start to search for the optimal k' 
# beginning at the position of the optimal k' associated with the previous capital. This is exactly the idea of 
# exploiting monotonicity! 

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
            k = 1
            for i in 1:length(k_grid) 
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                    next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:] 
                    while next_v_max > v_max && k < length(k_grid) - 1
                        k += 1  
                        v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                    end  
                    V[i, j] = v_max
                    policy_index[i,j] = k
                    policy[i,j] = k_grid[k]
            end
        end 

        iter += 1; # Soma um ao contador de iterações
        # print("\n", "Iter: ", iter)
        # print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    end
    print("Total iterations: ", iter, "\n")
end
=# 

#######
## 4 ##
#######

n_h = 10 # Number of iterations using the existing policy function we will perform after each policy function update.

#################
## Accelerator ##
#################

#= 

@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
            
        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 
        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            for i in 1:length(k_grid) 
                @threads for n in 1:length(k_grid)
                    values[n,j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                end
                V[i, j] = maximum(values[:, j])
                pos = argmax(values[:, j])
                policy_index[i, j] = pos 
                policy[i,j] = k_grid[pos]
            end
        end 

       for h in 1:n_h
            V_prev = copy(V)
            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                @threads for i in 1:length(k_grid)
                    V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy[i,j]) + β * Π[j,:]' * V_prev[trunc(Int, policy_index[i,j]),:] 
                end
            end
        end
        iter += 1 + n_h;
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(V_prev - V)))
    end
end

=# 

##########################################
# Accelerator + Monotonicity + Concavity # 
##########################################

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
            k = 1
            for i in 1:length(k_grid) 
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                    next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:] 
                    while next_v_max > v_max && k < length(k_grid) - 1
                        k += 1  
                        v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                    end  
                    V[i, j] = v_max
                    policy[i,j] = k_grid[k]
                    policy_index[i,j] = k
            end
        end 

        @threads for h in 1:n_h
            V_prev = copy(V)
            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                @threads for i in 1:length(k_grid)
                    V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy[i,j]) + β * Π[j,:]' * V_prev[trunc(Int, policy_index[i,j]),:] 
                end
            end
        end
        iter += 1; # Soma um ao contador de iterações
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
    end
    print("Total iterations: ", iter, "\n")
end

=# 

#######
## 5 ##
#######

#=

grid_sizes = [100, 500, 5000] # A vector containing the grid sizes that will be considered in the multigrid method. 
k_grids = [range(0.75*k_ss, 1.25*k_ss, length = s) for s in grid_sizes] # Generating the vector of grids that will be considered in the multigrid method.
k_grid = k_grids[1]; # Initial grid for k. 

V_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função valor.
policy = zeros(length(k_grid), length(z_grid)); # Vetor que armazenará a policy function.

c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β) # Initial guess. 

##########################################
## Multigrid + Monotonicity + Concavity ##
##########################################

@time begin

    pos = 1
    iter = 0

    for g in 1:length(k_grids)
    
        if g > 1 
            V = mapreduce(permutedims, vcat, [linear_interpolation(k_grids[g-1],V[:,i])(range(0.75*k_ss, 1.25*k_ss, length=length(k_grids[g]))) for i in 1:length(z_grid)])'
            V_prev = ones(length(k_grids[g]), length(z_grid));
            policy = zeros(length(k_grids[g]), length(z_grid)); # Vetor que armazenará a policy function
            policy_index = Int.(zeros(length(k_grids[g]), length(z_grid))); # Vetor que armazenará o índice da policy function  
        end

        while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
        # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
        # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.

            V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração. 
            # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 

            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                k = 1
                for i in 1:length(k_grids[g]) 
                        v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k+1]) + β * Π[j,:]' * V_prev[k+1,:] 
                        while next_v_max > v_max && k < length(k_grids[g]) - 1
                            k += 1  
                            v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k]) + β * Π[j,:]' * V_prev[k,:]
                            next_v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                        end  
                        V[i, j] = v_max
                        policy_index[i,j] = k
                        policy[i,j] = k_grids[g][k]
                        #print("Valor k: ", k, "\n")
                end
            end 

            iter += 1; # Soma um ao contador de iterações
            print("\n", "Iter: ", iter)
            print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
        end
    end 
    print("Total iterations: ", iter, "\n")
end

k_grid = k_grids[length(k_grids)]

=# 

#############################################
## Multigrid, with vectorized maximization ##
#############################################

#= 

@time begin # Computaremos o tempo de execução do processo de iteração.
    for g in 1:length(k_grids)
        # Matriz 3D de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k, z e k'.
        c_matrix = [z_grid[i]*(k_grids[g][j]^α).-k_grids[g][k].+(1-δ)*k_grids[g][j] for j in 1:length(k_grids[g]), i in 1:length(z_grid), k in 1:length(k_grids[g])]
        u_matrix = u.(c_matrix) # Matriz 3D de utilidades 
        
        if g > 1 
            V = mapreduce(permutedims, vcat, [linear_interpolation(k_grids[g-1],V[:,i])(range(0.75*k_ss, 1.25*k_ss, length=length(k_grids[g]))) for i in 1:length(z_grid)])'
            V_prev = ones(length(k_grids[g]), length(z_grid));
            policy = zeros(length(k_grids[g]), length(z_grid)); # Vetor que armazenará a policy function 
            policy_index = zeros(length(k_grids[g]), length(z_grid)); # Vetor que armazenará a policy function 
        end
        # print("Initial distance: ", maximum(abs.(V_prev - V)), " | Grid: ", g, "\n")
        while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
        # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
        # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
                
            V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
            # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 
            EV = [Π[j,:]' * V_prev[n,:] for n in 1:length(k_grids[g]), j in 1:length(z_grid)] # Matriz de valores esperados, para cada combinação possível de k' e z.
            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                @threads for i in 1:length(k_grids[g]) 
                    value = u_matrix[i,j,:] + β * EV[:,j]
                    V[i,j] = maximum(value); # Maximiza a função valor usando V_prev.
                    policy_index[i,j] = argmax(value)
                    policy[i,j] = k_grids[g][argmax(value)]
                end
            end 
            if iter == 0 
                print("Initial distance: ", maximum(abs.(V_prev - V)), " | Grid: ", g, "\n")
            end 
            iter += 1; # Soma um ao contador de iterações
            print("\n", "Iter: ", iter, " | Grid: ", g)
            print("\n", "Distance: ", maximum(abs.(V_prev - V)))
        end
    end 
    print("Total iterations: ", iter, "\n")
end
k_grid = k_grids[length(k_grids)]

=#

#######
## 6 ##
#######

#####################
## Endogenous Grid ##
#####################

#= 

policy_c_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função política.  
policy_c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z.  # ones(length(k_grid), length(z_grid)); # Vetor que armazenará a policy function 
policy_k = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_c[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)]

# Defining a function for the FOC expression that we need to find the root:
foc(k, z_index, kp_index) = z_grid[z_index]*k^α + (1-δ)*k - β^(-1/μ) * (Π[z_index,:]' * (policy_c[kp_index,:].^(-μ).*(α*z_grid*k_grid[kp_index].^(α-1).+(1-δ)))).^(-1/μ) .- k_grid[kp_index] 

egm_grid = zeros(length(k_grid), length(z_grid)) # Defining the variable that will store the EGM k grid.
egm_policy_k = zeros(length(k_grid), length(z_grid)) # Defining the variable that will store the EGM policy function. 

# Generating the EGM k grid: 
@time begin
    
    while maximum(abs.(policy_c_prev - policy_c)) > tol && iter < maxiter
        
        policy_c_prev = copy(policy_c)

        @threads for i in 1:length(k_grid) 
            @threads for j in 1:length(z_grid) 
                egm_grid[i,j] = find_zero((k -> foc(k, j, i)), 0) # Finding the root; i.e., solving for k.
            end 
        end
        
        # Recovering the capital policy on the exogenous grid by interpolation: 

        @threads for i in 1:length(k_grid)
            @threads for j in 1:length(z_grid)
                if k_grid[i] > egm_grid[1,j] && k_grid[i] < egm_grid[length(k_grid), j]
                    egm_policy_k[i,j] = linear_interpolation(egm_grid[:,j],k_grid)(k_grid[i])
                elseif k_grid[i] < egm_grid[1,j] # Seguindo Gordon, nos pontos que caem fora dos limites do grid (abaixo ou acima), precisamos definir desta forma...
                    egm_policy_k[i, j] = k_grid[1]
                elseif k_grid[i] > egm_grid[length(k_grid), j]
                    egm_policy_k[i, j] = k_grid[length(k_grid)]
                end 
            end 
        end 

        # Updating consumption policy function:
        policy_c = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - egm_policy_k[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)] 

        #iter += 1; # Soma um ao contador de iterações
        #print("\n", "Iter: ", iter)
        #print("\n", "Distance: ", maximum(abs.(policy_c_prev - policy_c)), "\n")
    end 

end

# Recovering capital policy function: 
policy = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_c[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)]

# "Estimando" as posições de k'(k, z) no grid de k: 
policy_index = [argmin(abs.(k_grid .- policy[i,j])) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Obtendo k' no grid exógeno
policy_exo = [k_grid[policy_index[i, j]] for i in eachindex(k_grid), j in eachindex(z_grid)]

# Recovering value function 

while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    V_prev = copy(V)
    for j in 1:length(z_grid)
        for i in 1:length(k_grid)
            V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_exo[i,j]) + β * Π[j,:]' * V_prev[policy_index[i,j],:] 
            # print("\n", V[i, j], "\n")
        end
    end

    # iter += 1; # Soma um ao contador de iterações
    # print("\n", "Iter: ", iter)
    # print("\n", "Distance: ", maximum(abs.(V_prev - V)))

end

=# 

#######################
## Results and plots ##
#######################

# Recovering consumption policy function: 
policy_c = [z_grid[i]*(k_grid[j]^α).-policy[j,i].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] 

# Plotting value function: 
plot(k_grid, V, 
    xlab = "k",
    ylab = "V", 
    label = ["State 1" "State 2" "State 3" "State 4" "State 5" "State 6" "State 7"], 
    legend = :outertopright) # 2D plot.
    
plot(repeat(k_grid,1,7), repeat(z_grid',500), V,
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

plot(repeat(k_grid,1,7), repeat(z_grid',500), policy,
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

plot(repeat(k_grid,1,7), repeat(z_grid',500), policy_c,
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
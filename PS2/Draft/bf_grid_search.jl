using Distributions, Random
using Plots
using BenchmarkTools
using Base.Threads

# Defining Tauchen function.

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

 
# Parameters
μ = 2;
β = 0.987;
α = 1/3;
δ = 0.012;
k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))

# Technical values
tol = 10^(-5); # Tolerância para a distância entre elementos da função valor. 
# Utilizaremos essa tolerância para definir quando interromper o processo iterativo. 
# tol define, portanto, a distância a partir da qualconsideramos que os elementos da função valor 
# estão "suficientemente próximo". Lembre que a sequência de V's é uma sequência de Cauchy.  

iter = 0; # Utilizarei essa variável para contar o número de iterações. 
maxiter = 1000; # Limite de iterações. Utilizaremos isso para evitar um possível loop infinito.

# Defining preferences: 
function u(c) 
    if c > 0
        u = (c^(1-μ) - 1)/(1-μ); # Utility function
    else 
        u = -Inf # Necessário para evitar consumo negativo. Se c < 0, utilidade = -infinito.
    end
    return u
end

k_grid = range(0.75*k_ss, 1.25*k_ss, length = 500); # Grid para o k. Necessário para discretizar o domínio.
tauch = tauchen(0,0.007^2,0.95,7,3)
z_grid = exp.(tauch[1]) 
Π = tauch[2]

V_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função valor. 
policy = zeros(length(k_grid), length(z_grid)); # Vetor que armazenará a policy function.
policy_index = Int.(zeros(length(k_grid), length(z_grid))); # Vetor que armazenará os índices da policy function.

V = zeros(length(k_grid), length(z_grid)); # Initial guess para a função valor. 
# Geralmente, zero. Mas existem chutes melhores. Por exemplo:
# V = repeat(ones(size(k_grid)).*u(k_ss)/(1-β), 1, 3)  # Judd's suggested initial guess.

# Moll's initial guess adaptado do caso determinístico: 
# V = repeat(u.(k_grid.^α.-k_grid.+(1-δ)*k_grid)./(1-β), 1, length(z_grid)) # Moll's initial guess.

# Moll's initial guess, mais adequado: 

# c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
# V = ((c.^(1-μ).-1)./(1-μ))./(1-β)


####################
# SEM PARALELIZAR: #
####################

#=
@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
            
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

#################
# PARALELIZANDO #
#################

# Para paralelizar os for-loops, não podemos utilizar múltiplos iteradores. 
# Logo, devo reescrever o código sem usar enumerate(.).

# Obs.: note que para que a paralelização funcione é preciso executar o Julia 
# declarando o número de threads do processador que serão utilizadas: 
# julia --threads=auto
# "auto" sets the number of threads to the number of local CPU threads.
# No VSCode: vá nas configurações da extensão do Julia para VSCode e defina 
# em settings.json:
# "julia.NumThreads": "auto" 
# Isso fará com que o VSCode sempre execute o Julia utilizando o número máximo
# de threads do processador. 

#= 

@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
            
        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 

        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            @threads for i in 1:length(k_grid) 
                @threads for n in 1:length(k_grid)
                    v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β * Π[j,:]' * V_prev[n,:] 
                    if v_max > V[i, j]
                        V[i, j] = v_max
                        policy[i, j] = k_grid[n]
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


# Forma alternativa: em vez de produto interno, mais um for. 
# Piorou bastante!

#= 

v_max = zeros(length(k_grid))
ev = 0

@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
        # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
        # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
                
            V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
            # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 
    
            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                @threads for i in 1:length(k_grid) 
                    @threads for n in 1:length(k_grid)
                        
                        ev = 0
                        for k in 1:length(z_grid)
                            ev = ev + Π[j,k] * V_prev[n, k] # Expected value
                        end  

                        v_max = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - k_grid[n]) + β*ev 
                        if v_max > V[i, j]
                            V[i, j] = v_max
                            policy[i, j] = k_grid[n]
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

###############
# Vectorizing #
###############

# Matriz 3D de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k, z e k'.
c_matrix = [z_grid[i]*(k_grid[j]^α).-k_grid[k].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid), k in 1:length(k_grid)]
u_matrix = u.(c_matrix) # Matriz 3D de utilidades 

@time begin # Computaremos o tempo de execução do processo de iteração.
    
    while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
    # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.
            
        V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração.
        # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 

        EV = [Π[j,:]' * V_prev[n,:] for n in 1:length(k_grid), j in 1:length(z_grid)] # Matriz de valores esperados, para cada combinação possível de k' e z.

        @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
            @threads for i in 1:length(k_grid) 
                value = u_matrix[i,j,:] + β * EV[:,j]
                V[i,j] = maximum(value); # Maximiza a função valor usando V_prev.
                #policy[i,j] = k_grid[argmax(value)]
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

policy_c = [z_grid[i]*(k_grid[j]^α).-policy[j,i].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)]

# Plotting value functions for each state:
plot(k_grid, V[:,1])
plot!(k_grid, V[:,2])
plot!(k_grid, V[:,3])
plot!(k_grid, V[:,4])
plot!(k_grid, V[:,5])
plot!(k_grid, V[:,6])
plot!(k_grid, V[:,7])

# Value function 3-D plot: 
plot(repeat(k_grid,1,7), repeat(z_grid',500), V, seriestype=:surface, camera=(10,50))


# Plotting policy function (k') for each state:

plot(k_grid, policy[:,1])
plot!(k_grid, policy[:,2])
plot!(k_grid, policy[:,3])
plot!(k_grid, policy[:,4])
plot!(k_grid, policy[:,5])
plot!(k_grid, policy[:,6])
plot!(k_grid, policy[:,7])

# k' policy function 3-D plot: 
plot(repeat(k_grid,1,7), repeat(z_grid',500), policy, seriestype=:surface, camera=(10,50))

# Plotting policy
plot(k_grid, policy_c[:,1])
plot!(k_grid, policy_c[:,2])
plot!(k_grid, policy_c[:,3])
plot!(k_grid, policy_c[:,4])
plot!(k_grid, policy_c[:,5])
plot!(k_grid, policy_c[:,6])
plot!(k_grid, policy_c[:,7])


# c policy function 3-D plot: 
plot(repeat(k_grid,1,7), repeat(z_grid',500), policy_c, seriestype=:surface, camera=(10,50))


###########################
## Euler Equation Errors ##
###########################

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (policy_c[policy_index[k_ind,z_ind],:].^(-μ) .* (α*z_grid*policy[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
EEE_plt = plot(k_grid, EEEs[:,4])
for i in 2:length(z_grid)
    plot!(k_grid, EEEs[:,i])
end

display(EEE_plt)

using Distributions, Random
using Plots
using BenchmarkTools
using Base.Threads
using Interpolations

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
maxiter = 100000; # Limite de iterações. Utilizaremos isso para evitar um possível loop infinito.

# Defining preferences: 
function u(c) 
    if c > 0
        u = (c^(1-μ) - 1)/(1-μ); # Utility function
    else 
        u = -Inf # Necessário para evitar consumo negativo. Se c < 0, utilidade = -infinito.
    end
    return u
end

###############
# Vectorizing #
###############

grid_sizes = [100, 200, 500]
k_grids = [range(0.75*k_ss, 1.25*k_ss, length = s) for s in grid_sizes]

k_grid = k_grids[1]; # Grid para o k. Necessário para discretizar o domínio.
tauch = tauchen(0,0.007^2,0.95,7,3)
z_grid = exp.(tauch[1]) 
Π = tauch[2]

V_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função valor 
policy = zeros(length(k_grid), length(z_grid)); # Vetor que armazenará a policy function 

# V = zeros(length(k_grid), length(z_grid)); # Initial guess para a função valor. 
# Geralmente, zero. Mas existem chutes melhores. Por exemplo:
# V = repeat(ones(size(k_grid)).*u(k_ss)/(1-β), 1, 3)  # Judd's suggested initial guess.

# Moll's initial guess adaptado do caso determinístico: 
# V = repeat(u.(k_grid.^α.-k_grid.+(1-δ)*k_grid)./(1-β), 1, length(z_grid)) # Moll's initial guess.

# Moll's initial guess, mais adequado: 

c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β)

################
## Vectorized ##
################


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
                    policy[i,j] = k_grids[g][argmax(value)]
                end
            end 
            if iter == 0 
                print("Initial distance: ", maximum(abs.(V_prev - V)), " | Grid: ", g, "\n")
            end 
            iter += 1; # Soma um ao contador de iterações
            # print("\n", "Iter: ", iter, " | Grid: ", g)
            # print("\n", "Distance: ", maximum(abs.(V_prev - V)))
        end
    end 
    print("Total iterations: ", iter, "\n")
end

k_grid = k_grids[length(k_grids)]
policy_c = [z_grid[i]*(k_grid[j]^α).-policy[j,i].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)]

=# 

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
        end

        while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
        # O código será interrompido quando a distância for menor que a tolerância definida anteriormente. 
        # Adicionalmente, impomos um limite de iterações para evitar a possibilidade de loops infinitos.

            V_prev = copy(V); # Define a função valor como a função valor obtida na última iteração. 
            # Note que na primeira iteração V_prev será, portanto, a initial guess para V. 

            @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
                k = 1
                @threads for i in 1:length(k_grids[g]) 
                        v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k]) + β * Π[j,:]' * V_prev[k,:]
                        next_v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k+1]) + β * Π[j,:]' * V_prev[k+1,:] 
                        while next_v_max > v_max && k < length(k_grids[g]) - 1
                            k += 1  
                            v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k]) + β * Π[j,:]' * V_prev[k,:]
                            next_v_max = u(z_grid[j]*k_grids[g][i]^α + (1-δ)*k_grids[g][i] - k_grids[g][k+1]) + β * Π[j,:]' * V_prev[k+1,:]
                        end  
                        V[i, j] = v_max
                        policy[i,j] = k_grids[g][k]
                        #print("Valor k: ", k, "\n")
                end
            end 

            iter += 1; # Soma um ao contador de iterações
            # print("\n", "Iter: ", iter)
            # print("\n", "Distance: ", maximum(abs.(V_prev - V)), "\n")
        end
    end 
    print("Total iterations: ", iter, "\n")
end

k_grid = k_grids[length(k_grids)]
policy_c = [z_grid[i]*(k_grid[j]^α).-policy[j,i].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)]

###########
## Plots ##
###########

# Plotting value functions for each state:
plot(k_grid, V[:,1])
plot!(k_grid, V[:,2])
plot!(k_grid, V[:,3])
plot!(k_grid, V[:,4])
plot!(k_grid, V[:,5])
plot!(k_grid, V[:,6])
plot!(k_grid, V[:,7])

# Value function 3-D plot: 
plot(repeat(k_grid,1,length(z_grid)), repeat(z_grid',length(k_grid)), V, seriestype=:surface, camera=(10,50))


# Plotting policy function (k') for each state:

plot(k_grid, policy[:,1])
plot!(k_grid, policy[:,2])
plot!(k_grid, policy[:,3])
plot!(k_grid, policy[:,4])
plot!(k_grid, policy[:,5])
plot!(k_grid, policy[:,6])
plot!(k_grid, policy[:,7])

# k' policy function 3-D plot: 
plot(repeat(k_grid,1,length(z_grid)), repeat(z_grid',length(k_grid)), policy, seriestype=:surface, camera=(10,50))

# Plotting policy
plot(k_grid, policy_c[:,1])
plot!(k_grid, policy_c[:,2])
plot!(k_grid, policy_c[:,3])
plot!(k_grid, policy_c[:,4])
plot!(k_grid, policy_c[:,5])
plot!(k_grid, policy_c[:,6])
plot!(k_grid, policy_c[:,7])


# c policy function 3-D plot: 
plot(repeat(k_grid,1,length(z_grid)), repeat(z_grid',length(k_grid)), policy_c, seriestype=:surface, camera=(10,50))

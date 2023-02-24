##################################
## Endogenous Grid Method (EGM) ##
##################################

using Distributions, Random
using Plots
using BenchmarkTools
using Base.Threads
using Roots 
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

        @threads for i in 1:length(k_grid) # CUIDADO: TALVEZ EU POSSA TER PROBLEMA COM VARIAVEL LOCAL/GLOBAL AQUI. NOTE QUE ISSO PRECISA FICAR DENTRO DO LOOP PQ A policy_c ATUALIZOU, E ELA ESTÁ DENTRO DA FUNÇÃO foc... Não sei o que vai acontecer. Vamos ver. 
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

        iter += 1; # Soma um ao contador de iterações
        print("\n", "Iter: ", iter)
        print("\n", "Distance: ", maximum(abs.(policy_c_prev - policy_c)), "\n")
    end 

end

# Recovering capital policy function: 
policy_k = [z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_c[i,j] for i in 1:length(k_grid), j in 1:length(z_grid)]

# Recovering Value Function 
### COMO FAZER?  

while maximum(abs.(V_prev - V_c)) > tol && iter < maxiter
    V_prev = copy(V)
    @threads for j in 1:length(z_grid) # Iniciamos o processo iterativo. Enumerate pega o índice e o valor do vetor, respectivamente.
        @threads for i in 1:length(k_grids[g])
            V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_k[i,j]) + β * Π[j,:]' * V_prev[trunc(Int, pol_index[i,j]),:] 
        end
    end
end


plot(k_grid, V_test[:,1])
plot!(k_grid, V_test[:,2])
plot!(k_grid, V_test[:,3])
plot!(k_grid, V_test[:,4])
plot!(k_grid, V_test[:,5])
plot!(k_grid, V_test[:,6])
plot!(k_grid, V_test[:,7])

# Consumption policy function plot. Super smooth... É isso mesmo? 
plot(k_grid, policy_c[:,1])
plot!(k_grid, policy_c[:,2])
plot!(k_grid, policy_c[:,3])
plot!(k_grid, policy_c[:,4])
plot!(k_grid, policy_c[:,5])
plot!(k_grid, policy_c[:,6])
plot!(k_grid, policy_c[:,7])

# Capital policy function plot 
plot(k_grid, policy_k[:,1])
plot!(k_grid, policy_k[:,2])
plot!(k_grid, policy_k[:,3])
plot!(k_grid, policy_k[:,4])
plot!(k_grid, policy_k[:,5])
plot!(k_grid, policy_k[:,6])
plot!(k_grid, policy_k[:,7])

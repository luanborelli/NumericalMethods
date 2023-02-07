using Plots
using NLsolve
using Distributions, Random
using FastGaussQuadrature
using Base.Threads

##################################
## RBC by Chebyshev Collocation ##
##################################

# Defining model parameters: 

β = 0.987
μ = 2
α = 1/3 
δ = 0.012 
ρ = 0.95
σ = 0.007

k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))

a = 0.75*k_ss
b = 1.25*k_ss
k_grid = range(a, b, length = 500);
normalized_k(k) = 2*((k - a)/(b - a)) - 1
unnormalized_k(k) = ((b - a)/2)*(k + 1) + a

# Defining Chebyshev polynomials and Chebyshev roots functions:

chebyshev_polynomial(x, degree) = cos(degree * acos(x))
chebyshev_root(i, degree) = -cos((2*i-1)/(2*degree)*pi);
chebyshev_roots(degree) = [chebyshev_root(i, degree) for i in 1:degree]

# Discretizing: 

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


tauch = tauchen(0,σ^2,ρ,7,3)
z_grid = exp.(tauch[1]) 
Π = tauch[2]

# Here our basis functions are the Chebyshev polynomials. 
# Defining the "approximate policy function", c_hat: 

function c_hat(k, z_ind, γ)
    fval = 0; 
    size_γ = size(γ)[1]
    for i in 0:(size_γ - 1) # Note que length(γ) = Ordem do polinomio + 1. Ou seja, length(γ) - 1 é a ordem do polinômio.
        fval = fval + γ[i+1, z_ind]*chebyshev_polynomial(normalized_k(k), i)
    end 
    return fval 
end

function R(k, z_ind, γ) 
    kp = z_grid[z_ind] * k^α + (1-δ)*k - c_hat(k, z_ind, γ)
    cps = [c_hat(kp, z, γ) for z in 1:length(z_grid)] # c(k', z')'s
    error = c_hat(k, z_ind, γ).^(-μ) - β * Π[z_ind,:]' * (cps .^(-μ) .* ( α*z_grid*kp.^(α - 1) .+ 1 .- δ))
    # error = c_hat(k,γ) - (β * Π[z_ind,:]' * (c_hat(kp, γ)^(-μ) * ( α*z_grid*kp^(α - 1) .+ 1 .- δ) ))^(-1/μ)
    return error
end

function system(γ) 
    size_γ = size(γ)[1]
    err = zeros(size_γ, length(z_grid))

    roots = chebyshev_roots(size_γ)
    collocation_points = unnormalized_k.(roots)

    for z in 1:length(z_grid)
        for i in 1:size_γ
            err[i, z] = R(collocation_points[i], z, γ)
        end
    end 

    return err
end

#= 

#init_guess = [i for i in range(0,4,5)]
init_guess = ones(2, length(z_grid))
params_c = zeros(length(init_guess[:,1]), length(z_grid))
policy_c = zeros(length(k_grid), length(z_grid))


# Resolvendo sem aprimoramento de chute: 
@time begin 
    @threads for i in 1:length(z_grid)  
        system_at_z = x -> system(x, i)
        params_c[:,i] = nlsolve(system_at_z, init_guess[:,i]).zero
        policy_c[:,i] = c_hat.(k_grid, repeat([params_c[:,i]], length(k_grid)))
    end
end 

plot(k_grid, policy_c)
=# 

deg = 5

params_c = zeros(deg, length(z_grid))
policy_c = zeros(length(k_grid), length(z_grid))

### Resolvendo com aprimoramento de chute:
@time begin
    for i in 1:length(z_grid)
        guess = ones(2, length(z_grid)) 
        for n in 1:deg ## 1 ou 2? Resultados iguais o.O
            results = nlsolve(system, guess).zero
            if n < deg
                guess = vcat(results, zeros(length(z_grid))')
            else 
                guess = results
            end 
            print("\n", guess, "\n")
        end 
        params_c = guess
    end
end 

policy_c = [c_hat(k, z, params_c) for k in k_grid, z in 1:length(z_grid)]

plot(k_grid, policy_c)

# Recovering capital policy function: 
policy_k = [z_grid[z]*k_grid[k]^α + (1-δ)*k_grid[k] - policy_c[k, z] for k in 1:length(k_grid), z in 1:length(z_grid)]
plot(k_grid, policy_k)
#plot(k_grid, policy_k, xlims = (47, 48), ylims =(47, 48)) # Zooming in for a better visualization. 


# Recovering value function:  # ESTÁ ESTRANHO... (!!!!!!!!!!!!)

tol = 10^(-5); # Tolerância para a distância entre elementos da função valor.
iter = 0; # Utilizarei essa variável para contar o número de iterações. 
maxiter = 1000; # Limite de iterações. Utilizaremos isso para evitar um possível loop infinito.
V_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função valor. 
# V = zeros(length(k_grid), length(z_grid)) # Initial guess 
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β)

# "Estimando" as posições de k'(k, z) no grid de k: 
pol_index = [findfirst(isequal(minimum(abs.(k_grid .- policy_k[i,j]))), abs.(k_grid .- policy_k[i,j])) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Defining preferences: 
function u(c) 
    if c > 0
        u = (c^(1-μ) - 1)/(1-μ); # Utility function
    else 
        u = -Inf # Necessário para evitar consumo negativo. Se c < 0, utilidade = -infinito.
    end
    return u
end

while maximum(abs.(V_prev - V)) > tol && iter < maxiter # Utilizamos a sup norm. 
    V_prev = copy(V)
    for j in 1:length(z_grid)
        for i in 1:length(k_grid)
            V[i, j] = u(z_grid[j]*k_grid[i]^α + (1-δ)*k_grid[i] - policy_k[i,j]) + β * Π[j,:]' * V_prev[pol_index[i,j],:] 
            # print("\n", V[i, j], "\n")
        end
    end

    iter += 1; # Soma um ao contador de iterações
    print("\n", "Iter: ", iter)
    print("\n", "Distance: ", maximum(abs.(V_prev - V)))

end

V

plot(k_grid, V)

# Euler Errors #

# No grid de 500: (no grid de 11 fica horrível)
cp = [c_hat(policy_k[k,z1], z2, params_c) for k in 1:length(k_grid), z1 in 1:length(z_grid), z2 in 1:length(z_grid)] # Cria um array de C primes, "uma matriz de C primes para cada estado". 

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (cp[k_ind, z_ind, :].^(-μ) .* (α*z_grid*policy_k[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(k_grid, EEEs)

# EEE máximo para cada choque: 
max_EEEs = [maximum(EEEs[:, s]) for s in eachindex(z_grid)]




# Tem algo errado... Pra ordens baixas (até 5 +-) roda, 
# mas policy functions não fazem sentido! SOLUÇÃO: aprimoramento de chutes. 


# Verificar resultados com diferentes especificações da função de residuo: 
## Tem a que eu usei, tem a "mais simples", sem tirar expoente nem nada... #
## Tem a que divide... OBS.: Testei com galerkin trocar a que usei por mais simplse e aparentemente resultados não mudaram. 

# Testar código do Greco sem aprimoramento para ver a robustez dos resultados. 

#####################
## RBC by Galerkin ##
#####################

using Plots
using NLsolve
using Distributions, Random
using FastGaussQuadrature
using Base.Threads

# Defining model parameters: 

β = 0.987
μ = 2
α = 1/3 
δ = 0.012 
ρ = 0.95
σ = 0.007
k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))
k_grid = range(0.75*k_ss, 1.25*k_ss, length = 11);

# Numerical integration function:

function integral_gl(f, a, b, n)
    nodes, weights = gausslegendre(n);
    result = weights' * f.((b-a)/2 * nodes .+ (a+b)/2)*(b-a)/2
    return result
end

#= 
########################
## Deterministic case ##
########################

# Defining the "approximate policy function", c_hat: 

function basis(k, i)
    if i == 1 
        if k >= k_grid[i] && k <= k_grid[i+1]
            value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
        else 
            value = 0
        end 
    elseif i == length(k_grid)
       if k >= k_grid[i-1] && k <= k_grid[i]
            value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
       else 
            value = 0
       end 
    elseif k >= k_grid[i-1] && k <= k_grid[i]
        value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
    elseif k >= k_grid[i] && k <= k_grid[i+1]
        value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
    else 
        value = 0
    end 
    return value 
end  

function c_hat(k, γ)
    fval = 0; 
    for i in 1:length(γ)
        fval = fval + γ[i]*basis(k, i)
    end 
    return fval 
end

function R_det(k, γ) 
    # kp = (- c_hat(k, γ) + k^α + (1-δ)*k) 
    error = c_hat(k,γ) - ( β * c_hat((- c_hat(k, γ) + k^α + (1-δ)*k), γ)^(-μ) * ( α*(- c_hat(k, γ) + k^α + (1-δ)*k)^(α - 1) + 1 - δ) )^(-1/μ)
    return error
end

function res(γ) 
    err = zeros(length(γ))

    for i in 1:length(γ)

        if i > 1
            restmp_det_1(k) = R_det(k, γ).*((k-k_grid[i-1])/(k_grid[i] - k_grid[i-1]))
            err[i] = err[i] + integral_gl(restmp_det_1, k_grid[i-1], k_grid[i], 1000)
            #print(err[i], "\n")
        end 
        
        if i < length(γ)
            restmp_det_2(k) = R_det(k, γ).*((k_grid[i+1]-k)/(k_grid[i+1] - k_grid[i]))
            err[i] = err[i] + integral_gl(restmp_det_2, k_grid[i], k_grid[i+1], 1000)
            #print(err[i], "\n")
        end 
    end

    return err
end

init_guess = [i for i in range(1,10, length(k_grid))] 

@time begin
    params_det = nlsolve(res, init_guess).zero
end

approx_det = c_hat.(k_grid, repeat([params_det], length(k_grid)))
# approx_det = c_hat.(range(0.75*k_ss, 1.25*k_ss, length = 500), repeat([params_det], 500))

plot(k_grid, approx_det)
# plot(range(0.75*k_ss, 1.25*k_ss, length = 500), approx_det)
approx_det

=# 

#####################
## Stochastic case ##
#####################


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

# Defining the basis function: 

function basis(k, i)
    if i == 1 
        if k >= k_grid[i] && k <= k_grid[i+1]
            value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
        else 
            value = 0
        end 
    elseif i == length(k_grid)
       if k >= k_grid[i-1] && k <= k_grid[i]
            value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
       else 
            value = 0
       end 
    elseif k >= k_grid[i-1] && k <= k_grid[i]
        value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
    elseif k >= k_grid[i] && k <= k_grid[i+1]
        value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
    else 
        value = 0
    end 
    return value 
end  


# Defining the "approximate policy function", c_hat: 

function c_hat(k, γ)
    fval = 0; 
    for i in 1:length(γ)
        fval = fval + γ[i]*basis(k, i)
    end 
    return fval 
end

function R(k, z_ind, γ) 
    kp = - c_hat(k, γ) + z_grid[z_ind]*k^α + (1-δ)*k 
    # error = c_hat(k,γ) - (β * Π[z_ind,:]' * (c_hat(kp, γ)^(-μ) * ( α*z_grid*kp^(α - 1) .+ 1 .- δ) ))^(-1/μ)
    error = c_hat(k, γ)^(-μ) - β * Π[z_ind,:]' * (c_hat(kp, γ)^(-μ) * ( α*z_grid*kp^(α - 1) .+ 1 .- δ))
    return error
end

function res_st(γ, z) 
    err = zeros(length(γ))

    for i in 1:length(γ)

        if i > 1
            restmp_det_1(k) = R(k, z, γ).*((k-k_grid[i-1])/(k_grid[i] - k_grid[i-1]))
            err[i] = err[i] + integral_gl(restmp_det_1, k_grid[i-1], k_grid[i], 11)
        end 
        
        if i < length(γ)
            restmp_det_2(k) = R(k, z, γ).*((k_grid[i+1]-k)/(k_grid[i+1] - k_grid[i]))
            err[i] = err[i] + integral_gl(restmp_det_2, k_grid[i], k_grid[i+1], 11)
        end 
    end

    return err
end


# Solving: 

init_guess = [i for i in range(2,4, length(k_grid))]
params_c = zeros(length(k_grid), length(z_grid))
policy_c = zeros(length(k_grid), length(z_grid))

@time begin 
    @threads for i in 1:length(z_grid)  
        res_shock = x -> res_st(x, i)
        params_c[:,i] = nlsolve(res_shock, init_guess).zero
        policy_c[:,i] = c_hat.(k_grid, repeat([params_c[:,i]], length(k_grid)))
    end
end 

plot(k_grid, policy_c)

# Criando função para a "policy function estimada", usando parâmetros obtidos: 
function estimated_c(k, z)
    fval = 0; 
        for i in 1:length(params_c[:, z])
            fval = fval + params_c[:, z][i]*basis(k, i)
        end 
    return fval 
end

# Obtendo consumption policy para grid de 500 pontos: 
new_k_grid = range(0.75*k_ss, 1.25*k_ss, length = 500);
policy_c_500 = hcat([estimated_c.(new_k_grid, z) for z in 1:length(z_grid)]...)

# Recuperando função política do capital:
policy_k = [z_grid[z]*k_grid[k]^α + (1-δ)*k_grid[k] - policy_c[k, z] for k in 1:length(k_grid), z in 1:length(z_grid)]
plot(k_grid, policy_k)
plot(k_grid, policy_k, xlims = (47, 48), ylims =(47, 48)) # Zooming in for a better visualization. 

# Com 500 pontos: 
policy_k_500 = [z_grid[z]*new_k_grid[k]^α + (1-δ)*new_k_grid[k] - policy_c_500[k, z] for k in 1:length(new_k_grid), z in 1:length(z_grid)]

####################
## VALUE FUNCTION ## (!!!!!!) CAUTION
####################

# Recovering value function: 
tol = 10^(-5); # Tolerância para a distância entre elementos da função valor.
iter = 0; # Utilizarei essa variável para contar o número de iterações. 
maxiter = 1000; # Limite de iterações. Utilizaremos isso para evitar um possível loop infinito.
V_prev = ones(length(k_grid), length(z_grid)); # Vetor temporário para a iteração da função valor. 
# V = zeros(length(k_grid), length(z_grid)) # Initial guess 
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β)

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

##########
## EEEs ## (!!!!!!!!!!) Ainda não consegui. 
##########

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (policy_c[pol_index[k_ind,z_ind],:].^(-μ) .* (α*z_grid*policy_k[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
EEE_plt = plot(k_grid, EEEs)

### Testando com 500 pontos: 


pol_index_500 = [findfirst(isequal(minimum(abs.(new_k_grid .- policy_k_500[i,j]))), abs.(new_k_grid .- policy_k_500[i,j])) for i in 1:length(new_k_grid), j in 1:length(z_grid)]

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE_500(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (policy_c_500[pol_index_500[k_ind,z_ind],:].^(-μ) .* (α*z_grid*policy_k_500[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c_500[k_ind, z_ind])))

# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs_500 = [EEE_500(i,j) for i in 1:length(new_k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
EEE_plt_500 = plot(new_k_grid, EEEs_500)


# Tentando de forma alternativa: ## RESULTADO TOTALMENTE NONSENSE... 

cp = [c_hat(policy_k[k,z], params_c[:,z]) for k in 1:length(k_grid), z in 1:length(z_grid)] # matrix of c'

# This function calculates the Euler Error for a given k (index) and a given z (index): 
EEE(k_ind, z_ind) = log10(abs(1-((β*(Π[z_ind,:]' * (cp[k_ind,:].^(-μ) .* (α*z_grid*policy_k[k_ind,z_ind]^(α-1) .+ 1 .- δ))))^(-1/μ))/(policy_c[k_ind, z_ind])))
# policy_c[pol_index[k_ind,z_ind],:]
# Generating a matrix of EEEs for every possible combination of k's and z's:
EEEs = [EEE(i,j) for i in 1:length(k_grid), j in 1:length(z_grid)]

# Plotting Euler Equation Errors: 
plot(k_grid, EEEs)



################## 

# Aqui eu resolvi choque-a-choque... Será que não existe uma forma alternativa de resolver que seja mais rápida? 
# Outra coisa: usei 11 grid points. Eu poderia usar menor? O quão menos? 

# A missão agora é melhorar o código. 

# Multi-dimensional projection? 


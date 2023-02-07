#####################
## RBC by Galerkin ##
#####################

using Plots
using NLsolve
# using QuadGK # Tried to use, didn't work.
using Distributions, Random
# using MultiQuad # Tried to use, didn't work.
using FastGaussQuadrature # I ended up defining my own numerical integration function using this Quadrature package.

# Defining model parameters: 

β = 0.987
μ = 2
α = 1/3 
δ = 0.012 
ρ = 0.95
σ = 0.007
k_ss = ((1-β*(1-δ))/(α*β))^(1/(α-1))
k_grid = range(0.75*k_ss, 1.25*k_ss, length = 500);

# Numerical integration function:

function integral_gl(f, a, b, n)
    nodes, weights = gausslegendre(n);
    result = weights' * f.((b-a)/2 * nodes .+ (a+b)/2)*(b-a)/2
    return result
end

########################
## Deterministic case ##
########################

# Defining the "approximate policy function", c_hat: 

function basis(k, i)
    if i == 1 || i == length(k_grid)
        value = 0
    elseif k >= k_grid[i-1] && k <= k_grid[i]
        value = (k - k_grid[i-1])/(k_grid[i] - k_grid[i-1])
    elseif k >= k_grid[i] && k <= k_grid[i+1]
        value = (k_grid[i+1] - k)/(k_grid[i+1] - k_grid[i])
    else 
        value = 0
    end 
    return value 
end 

#= Monomials 
 function c_hat(k, γ)
    fval = 1; 
    for i in 1:length(γ)
        fval = fval + γ[i]*k.^i
    end 
    return fval 
end
=# 

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
        ftmp_det(k) = (c_hat(k,γ) - ( β * c_hat((- c_hat(k, γ) + k^α + (1-δ)*k), γ)^(-μ) * ( α*(- c_hat(k, γ) + k^α + (1-δ)*k)^(α - 1) + 1 - δ) )^(-1/μ)).*basis.(k,i)
        err[i] = integral_gl(ftmp_det, k_grid[1], k_grid[length(k_grid)], 1000)
    end 
    return err
end

params_det = nlsolve(res, zeros(length(k_grid),1)).zero
#approx_det = c_hat.(k_grid, repeat([params_det], length(k_grid)))
approx_det = c_hat.(range(1,2, 100), repeat([params_det], 100))

plot(range(1,2, 100), approx_det)
approx_det

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


tauch = tauchen(0,σ^2,ρ,10,3)
z_grid = exp.(tauch[1]) 
Π = tauch[2]

# Defining the "approximate policy function", c_hat: 

function c_hat(k, γ)
    fval = 1; 
    for i in 1:length(γ)
        fval = fval + γ[i]*k.^i
    end 
    return fval 
end

function R(k, z_ind, γ) 
    kp = - c_hat(k, γ) + z_grid[z_ind]*k^α + (1-δ)*k 
    error = c_hat(k,γ) - (β * Π[z_ind,:]' * (c_hat(kp, γ)^(-μ) * ( α*z_grid*kp^(α - 1) .+ 1 .- δ) ))^(-1/μ)
    return error
end

function res(γ, z) 
    err = zeros(length(γ))
    for i in length(γ)
        ftmp(k) = R(k, z, γ).*k.^i
        err[i] = integral_gl(ftmp, k_grid[1], k_grid[length(k_grid)], 100000)
    end 
    return err
end

R_1 = x -> res(x, 1)

params = nlsolve(R_1, zeros(4,1)).zero
approx = c_hat.(k_grid, repeat([params], length(k_grid)))

plot(k_grid, approx)
# plot!(k_grid, policy[:,1])


# Tá errado...
# Potencialidades: ou escrevi a função errada, ou defini a função aproximada de forma errada...

# Talvez eu precise de projeção multi-dimensional...
# Mas primeiro vou tentar fazer funcionar no caso deterministico, em que a função política é univariada (topo do código). 
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

function c_hat(k, γ)
    fval = zeros(length(k), length(z_grid)); 
    for j in 1:length(z_grid)
        for i in 1:size(γ)[1]
            fval[:,j] = fval[:,j] .+ γ[i, j].*basis.(k,i)
        end 
    end 
    return fval 
end

function R(k, z_ind, γ) 
    error = zeros(length(k), length(z_grid))
    kp = [z_grid[j]*k[i].^α + (1-δ)*k[i] .- c_hat(k, γ)[i,j] for i in 1:length(k), j in 1:length(z_grid)]

    for i in 1:length(k)
        for j in 1:length(z_grid)
            error[i,j] = c_hat(k, γ)[i,j]^(-μ) - β * Π[j,:]' * (c_hat(kp[:,j], γ)[i,:].^(-μ) .* ( α*z_grid*kp[i,j]^(α - 1) .+ 1 .- δ))
        end     
    end

    return error[:, z_ind]
end

# E AGORA? COMO MONTO ESSE SISTEMA VETORIALMENTE? 

function res_st(γ, z) 
    err = zeros(length(γ), length(z_grid))

    for i in 1:length(γ)

        if i > 1
            for n in length(k_grid)
                for j in length(z_grid) # EU ACREDITO QUE ISSO VÁ FUNCIONAR, CONTANTO QUE A FÇ R(.) QND RECEBER UM SINGLETON RETORNE UM NÚMERO!
                    restmp_det_1(k) = R(k, j, γ).*((k.-k_grid[i-1])./(k_grid[i] - k_grid[i-1]))
                    err[n, j] = err[n, j] + integral_gl(restmp_det_1, k_grid[i-1], k_grid[i], 11)
                end           
            end 

        end 
        
        if i < length(γ)
            restmp_det_2(k) = R(k, z, γ).*((k_grid[i+1]-k)/(k_grid[i+1] - k_grid[i]))
            err[i] = err[i] + integral_gl(restmp_det_2, k_grid[i], k_grid[i+1], 11)
        end 
    end

    return err
end

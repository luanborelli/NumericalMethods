using Random, Distributions
using Plots

function rouwenhorst(μ,σ,ρ,N)
    θ_1 = μ - sqrt(N-1)*σ/sqrt(1-ρ^2)
    θ_N = μ + sqrt(N-1)*σ/sqrt(1-ρ^2)
    θ_grid = range(θ_1, θ_N, N) 
    p = (1+ρ)/2
    P_2 = [p 1-p; 1-p p]
    P_prev = P_2 

    for i in 3:N 
        P_N = p*[P_prev zeros(i-1, 1); zeros(i-1, 1)' 0] + (1-p)*[zeros(i-1, 1) P_prev; 0 zeros(i-1, 1)'] + (1-p)*[zeros(i-1, 1)' 0; P_prev zeros(i-1, 1)] + p*[0 zeros(i-1, 1)'; zeros(i-1, 1) P_prev]
        P_prev = P_N 
        normalizer = repeat(sum(P_prev, dims=2), 1, i) # Matrix used to normalize the sum of each row. 
        P_prev = P_prev./normalizer # Normalizing the sum of each row to 1.
    end 

    return θ_grid, P_prev
end 

function ar(mu,sigma,rho,n)
    errors = rand(Normal(0,sigma), n)
    θ = mu*ones(n)
  
    for i in 2:length(errors) 
      θ[i] = (1-rho)*mu + rho*θ[i-1] + errors[i] 
    end
  
    return θ, errors
  end 

function rouw_discretized_ar(mu,sigma_sq,rho,N, errors)

    sigma = sqrt(sigma_sq)
    n = length(errors)

    rouw = rouwenhorst(mu,sigma,rho,N)
    theta_grid = rouw[1]
    Pi = rouw[2]
  
    θ_rouw = zeros(n) 
    Π_cdf = hcat([accumulate(+, Pi[i,:]) for i in 1:size(Pi,1)]...)'
  
    # θ_rouw[1] = theta_grid[argmin(abs.(errors[1].-theta_grid))] # Defining the first observation of the discretized time series
    θ_rouw[1] = mu
  
    for t in 2:n # Generating the discretized time series
      if θ_rouw[t-1] <= theta_grid[1]
        i = 0
      else 
        i = length(theta_grid[theta_grid .< θ_rouw[t-1]])
      end 
      
      print(i)
      
      cdf_vector = Π_cdf[i+1,:]
  
      if cdf(Normal(0,sigma),errors[t]) <= cdf_vector[1] ### É NORMAL 0,1 MESMO? TENHO A IMPRESSÃO DE QUE NA VERDADE DEVERIA SER A DISTRIBUIÇÃO DO AR E COMPARAR COM A REALIZAÇÃO DO AR CONTINUO, NAO COM O ERRO...
        j = 0
      else 
        j = length(theta_grid[cdf_vector .< cdf(Normal(0,sigma), errors[t])])
      end  
  
      print(j)
  
      θ_rouw[t] = theta_grid[j+1]
    end
    return θ_rouw 
end

# Example: 
# rouwenhorst(1, sqrt(0.05), 0.9, 7)


### Generating the discretized process 


# Defining process' parameters:

μ = 0 
σ = 0.007
ρ = 0.99
n = 1000

# Generating the continuous AR(1) process:

continuous_ar = ar(μ, σ, ρ, n)
ts = continuous_ar[1] # AR(1) time series. 
ε = continuous_ar[2] # AR(1) errors. 

# Generating the discretized AR(1) process: 

discrete_ar = rouw_discretized_ar(μ, σ^2, ρ, 9, ε)
 
# θ_tauchen = discretized_process(θ_grid, Π, ε)

plot(1:length(ts), ts)
plot!(1:length(discrete_ar), discrete_ar)

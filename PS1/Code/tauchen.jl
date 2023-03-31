using Random, Distributions
using Plots

# This code calculates and returns states vector (grid points) and transition matrix for the discrete markov process approximation 
# of AR(1) process specified as θ_t = μ*(1-ρ) + ρ*z_{t-1} + ε_t, ε_t ~ N(0,σ^2), by Tauchen's method.
  
# m stands for the maximum number of standard deviatons from mean;
# N stands for the number of states in discretization of θ (must be an odd number).

function tauchen(mu,σsq,rho,N,m)
  
    σ = sqrt(σsq); # Standard deviation of ε.
    μ_θ = mu; # Expected value of θ_t. Notice that in this particular specification, it is exactly μ. In other specifications, it could be e.g. μ/(1-ρ).
    σ_θ = σ/sqrt(1-rho^2); # Standard deviation of θ_t.
      
    θ = range(μ_θ - m*σ_θ, μ_θ + m*σ_θ, N); # Generating grid points. 
    Δθ = θ[2]-θ[1]; # This is the width of the invervals. Notice that all the points are equidistant, by construction.
      

    # Now we calculate all the probability transition matrix elements "at once" through matrix operations. 
    # Idea: all we need to do this is to calculate a matrix of "all possible differences" of grid θ elements.
    # In general, letting 1 be a vector of ones, the matrix of all possible differences between elements
    # of vectors  u and v is given by: D := 1u' - v1'.

    θ_j = ones(N,1)*θ';
    θ_i = θ*ones(N,1)';
    
    P_1 = cdf(Normal(),((θ_j - rho*θ_i .+ Δθ/2) .- (1-rho)*mu)/σ);
    P_2 = cdf(Normal(),((θ_j - rho*θ_i .- Δθ/2) .- (1-rho)*mu)/σ);
  
    P = P_1 - P_2;
    
    # Calculating corner transition probabilities:
    P[:,1] = P_1[:,1]; 
    P[:,N] = -P_2[:,N] .+ 1;

    return θ, P    
end 

function ar(mu,sigma,rho,n)
  errors = rand(Normal(0,sigma), n)
  θ = mu*ones(n)

  for i in 2:length(errors) 
    θ[i] = (1-rho)*mu + rho*θ[i-1] + errors[i] 
  end

  return θ, errors
end 

function tauch_discretized_ar(mu, sigma_sq, rho, N, m, errors)

  sigma = sqrt(sigma_sq)
  tauch = tauchen(mu,sigma_sq,rho,N,m)
  theta_grid = tauch[1]
  Pi = tauch[2]

  θ_tauchen = zeros(length(errors)) 
  Π_cdf = hcat([accumulate(+, Pi[i,:]) for i in 1:size(Pi,1)]...)'

  # θ_tauchen[1] = theta_grid[argmin(abs.(errors[1].-theta_grid))] # Defining the first observation of the discretized time series
  θ_tauchen[1] = mu

  for t in 2:length(errors) # Generating the discretized time series
    if θ_tauchen[t-1] <= theta_grid[1]
      i = 0
    else 
      i = length(theta_grid[theta_grid .< θ_tauchen[t-1]])
    end 
    
    print(i)
    
    cdf_vector = Π_cdf[i+1,:]

    if cdf(Normal(0,sigma),errors[t]) <= cdf_vector[1] ### É NORMAL 0,1 MESMO? TENHO A IMPRESSÃO DE QUE NA VERDADE DEVERIA SER A DISTRIBUIÇÃO DO AR E COMPARAR COM A REALIZAÇÃO DO AR CONTINUO, NAO COM O ERRO...
      j = 0
    else 
      j = length(theta_grid[cdf_vector .< cdf(Normal(0,sigma), errors[t])])
    end  

    print(j)

    θ_tauchen[t] = theta_grid[j+1]
  end
  return θ_tauchen 
end

####
## Here's an example, from Cezar's slides:
####

#= 

results = tauchen(1.0,0.05,0.9,3,3)

print("Grid points: \n")
print("\n")
print(results[1][1], "\n", results[1][2], "\n", results[1][3], "\n")

print("\n")

print("Transition matrix: \n")
print("\n")
display(results[2])

=# 

### Checking using QuantEcon: 

#= 
using QuantEcon 
QuantEcon.tauchen(3, 0.9, sqrt(0.05), 1*(1-0.9), 3).state_values # State values
QuantEcon.tauchen(3, 0.9, sqrt(0.05), 1*(1-0.9), 3).p # Transition matrix
=# 

### Generating the discretized process 


# Defining process' parameters:

μ = 1 
σ = sqrt(0.05)
ρ = 0.9 
n = 100

# Generating the continuous AR(1) process:

continuous_ar = ar(μ, σ, ρ, n)
ts = continuous_ar[1] # AR(1) time series. 
ε = continuous_ar[2] # AR(1) errors. 

# Generating the discretized AR(1) process: 

discrete_ar = tauch_discretized_ar(μ, σ^2, ρ, 4, 3, ε)
 
# θ_tauchen = discretized_process(θ_grid, Π, ε)

plot(1:length(ts), ts)
plot!(1:length(discrete_ar), discrete_ar)

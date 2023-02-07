######################################
## Numerical Integration Playground ##
######################################

# import Pkg; Pkg.add("QuadGK")
# using QuadGK
using Optim
using Plots
using Roots

# import Pkg; Pkg.add("FastGaussQuadrature")
using FastGaussQuadrature

# import Pkg; Pkg.add("SymPy")
# using SymPy
# import Pkg; Pkg.add("Calculus")
# using Calculus 

# import Pkg; Pkg.add("NLsolve")
using NLsolve

# A solução para a equação diferencial é: 
f(t) = exp(-t)

R(t, γ_1, γ_2) = γ_1 + 2*γ_2*t + 1 + γ_1 * t + γ_2 * t^2


quadgk(R, 0, 2)

p(γ) = quadgk(γ[1] + 2*γ[2]*t + 1 + γ[1]*t + γ[2] + t^2, 0, 2)

optimize(p)


# Acho que não tem jeito. Tenho que tomar a integral analítica "na mão" mesmo  
# e seguir com a otimização. I surrender. Tentar fazer assim.

##########################
## Least squares method ##
##########################

##
# We want to solve min_{γ_1, γ_2} int_0^2 R^2 dt. 
#
# Notice that: 
# dR/dγ_1 = 1 + t 
# dR/dγ_2 = 2t + 1 
#
# The FOCs are then 
# int_0^2 (γ_1 + 2*γ_2*t + 1 + γ_1 * t + γ_2 * t^2)(1+t) dt = 0
# int_0^2 (γ_1 + 2*γ_2*t + 1 + γ_1 * t + γ_2 * t^2)(2t + 1) dt = 0
#
# or (according to slides... but I did not get this by myself...)
# 
# 13*\gamma_1 + 24*\gamma_2 + 6 = 0
# 60*\gamma_1 + 124*\gamma_2 + 25 = 0
# 
##

ls_sol = inv([13 24; 60 124]) * [-6; -25]
x_ls(t) = 1 + ls_sol[1]*t + ls_sol[2]*t^2

# Plot: 
grid = range(0, 1, length=10)
y1 = f.(grid) 


y2_ls = x_ls.(grid)
plot(grid, y1)
plot!(grid, y2_ls)

######################
## Galerkin Method ##
######################

# FOCs: 
# 7*γ_1 + 14*γ_2 = -3
# 25*γ_1 + 54*γ_1 = -10

# Solution: 
galerkin_sol = inv([7 14; 25 54]) * [-3; -10]

# Resulting approximation: 
x_gal(t) = 1 + galerkin_sol[1]*t + galerkin_sol[2]*t^2

# Plot: 
y2_gal = x_gal.(grid)
plot(grid, y1, label="Real solution")
plot!(grid, y2_ls, label = "Least Squares approximation")
plot!(grid, y2, label = "Galerkin approximation")

# Note que a aproximação é razoavelmente boa para valores pequenos de x.
# Para valores grandes, é horrível. 

grid_10 = range(0, 10, length=10)
y1_10 = f.(grid_10) 
y2_10 = x.(grid_10)
plot(grid, y1_10)
plot!(grid, y2_10)

#################
## Collocation ##
#################


# Recall that 
# R(t, γ_1, γ_2) = γ_1 + 2*γ_2*t + 1 + γ_1 * t + γ_2 * t^2
# or 
# R(t, γ_1, γ_2) = γ_1(1 + t) + γ_2(2*t + t^2) + 1
# 
# We can define the coefficients of each γ as functions of t:

gamma_1(t) = 1 + t 
gamma_2(t) = 2*t + t^2

# Now we define the (finite) set of predetermined points: 
T = [1, 2]
# (???) How to define theses points? If we decide to use more than 2, we have an overidentified system...

# And we define a system of T equations: 

A = vcat([[gamma_1(i) gamma_2(i)]  for i in T]...)
b = repeat([-1], length(T))

col_sol = inv(A)*b
x_col(t) = 1 + col_sol[1]*t + col_sol[2]*t^2

y2_col = x_col.(grid)

plot(grid, y1, label="Real solution")
plot!(grid, y2_ls, label = "Least Squares approximation")
plot!(grid, y2, label = "Galerkin approximation")
plot!(grid, y2_col, label = "Collocation approximation")




#####################################
## Galerkin (from Gordon's slides) ##
#####################################

function f(t, γ)
    fval = 1; 
    for i in 1:length(γ)
        fval = fval + γ[i]*t.^i
    end 
    return fval 
end 

function fp(t, γ)
    fval = 0 
    for i in 1:length(γ) 
        fval = fval + i*γ[i]*t.^(i-1);
    end 
    return fval 
end

function res(γ) 
    err = zeros(length(γ))
    for i in 1:length(γ)
        ftmp(t) = (f(t,γ) - fp(t,γ)).*t.^i
        err[i] = quadgk(ftmp, 0, 1)
    end 
    return err
end 

solution = find_zero(res, 0)

###########################
## Fast Gauss Quadrature ##
###########################

# I've tried to use quadgk but it uses an adaptive method that ended up being problematic. 
# After some research I then decided to use the package FastGaussQuadrature to calculate quadratures 
# and, with this package, I defined my own numerical integration function, as follows. 

# So far the package includes gausschebyshev(), gausslegendre(), gaussjacobi(), gaussradau(), gausslobatto(), gausslaguerre(), and gausshermite().

# Function to calculate numerically the integral of a given function f from a to b,
# using the Gauss-Legendre method. 

function integral_gl(f, a, b, n)
    nodes, weights = gausslegendre(n);
    result = weights' * f.((b-a)/2 * nodes .+ (a+b)/2)*(b-a)/2
    return result
end 

integral_gl(g, 1, 2, 100000)


# Trying Galerkin now using this function: 

function res(γ) 
    err = zeros(length(γ))
    for i in 1:length(γ)
        ftmp(t) = (f(t,γ) - fp(t,γ)).*t.^i
        err[i] = integral_gl(ftmp, 0, 1, 100000)
    end 
    return err
end 

# solution = find_zero(res, zeros(5,1))
# Aparentemente find_zero não consegue lidar com sistemas de equações não lineares... Tive que apelar para o pacote NLSolve: 

solution = nlsolve(res, zeros(10,1)).zero

# Funcionou!!! FastGaussQuadrature + NLSolve FTW!!!

grid = range(0,1, length=100)

# Real solution: 
real = exp.(grid)

# Approximate solution:
approx = f.(grid, repeat([solution], length(grid)))

plot(grid, real, label = "Actual function")
plot!(grid, approx, label = "Approximation (n=5)")

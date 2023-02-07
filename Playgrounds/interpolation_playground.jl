############################
# INTERPOLATION PLAYGROUND #
############################

using Interpolations
using Plots 
# import Pkg; Pkg.add("Dierckx")
using Dierckx

grid_sizes = [90, 95, 500]
k_grids = [range(0.75*k_ss, 1.25*k_ss, length = s) for s in grid_sizes]
k_grid = k_grids[2]; # Grid para o k. Necessário para discretizar o domínio.
c = [z_grid[i]*(k_grid[j]^α).-k_grid[j].+(1-δ)*k_grid[j] for j in 1:length(k_grid), i in 1:length(z_grid)] # Matriz de consumos. Esta matriz computa todos os consumos possíveis para todas as combinações possíveis de k e z. 
V = ((c.^(1-μ).-1)./(1-μ))./(1-β)

plot(k_grids[2], V)

# Linear Interpolation

linterp_V = mapreduce(permutedims, vcat, [linear_interpolation(k_grids[2],V[:,i])(range(0.75*k_ss, 1.25*k_ss, length=length(k_grids[3]))) for i in 1:length(z_grid)])'

plot(k_grids[3], linterp_V)

# Spline Interpolation using Dierckx 

spl_interp_V = mapreduce(permutedims, vcat, [Spline1D(k_grids[2], V[:,i])(range(0.75*k_ss, 1.25*k_ss, length=length(k_grids[3]))) for i in 1:length(z_grid)])'


# Comparação: 

plot(k_grids[3], spl_interp_V)
plot!(k_grids[3], linterp_V)

# Virtualmente a mesma cois! 
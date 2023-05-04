#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, LinearAlgebra
using Plots, Test, Random

"""
Cahn-Hilliard equation

∂ₜu = MΔf(u) - κΔ²u 

x ∈ [0, L)ᵈ (periodic)

# https://doi.org/10.1016/S0898-1221(02)00142-6
"""

Nx = Ny = 32
Lx = Ly = 10.0
p = nothing

""" space discr """
domain = IntervalDomain(0, Lx) ⊗ IntervalDomain(0, Ly)
space  = FourierSpace(Nx, Ny; domain = domain)
tspace = transform(space)
discr  = Collocation()

(x, y,) = points(space)
iftr = transformOp(tspace)
ftr  = transformOp(space)

###
# parameters
###

κ = 2.0
M = 5.0

ρ = 5.0
ua = 0.3
ub = 0.7

function uic(x, y)
    c0 = 0.5
    ep = 0.01

    c0 + ep * (
        cos(0.05x) * cos(0.11y) + (cos(0.13x) * cos(0.087y))^2.0
            + cos(0.025x - 0.15y) * cos(0.07x - 0.02y)
    )
end

u0 = @. uic(x, y)

"""
F(u) = ρ * (u^2 - ua) * (u^2 - ub)
     = ρ * ((u^4 - u^2 (ua + ub) + ua*ub)

df(u) = ρ * (4u^3 - 2u (ua + ub))
"""
function dfdu(u, p, t)
    @. ρ * (4 * u^3 - 2 * (ua + ub) * u)
end

function dfdu(v, u, p, t)
    v .= dfdu(u, p, t)
end

# ∂ₜu = MΔf(u) - κΔ²u 
A = -laplaceOp(space, discr)
B = biharmonicOp(space, discr)
F = FunctionOperator(dfdu, u0;)

L = cache_operator(M*A*F - κ*B, u0)
N = cache_operator(F, u0)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = SplitODEProblem(L, N, u0, tspan, p)

odecb = begin
    function affect!(int)
        if int.iter % 100 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end
@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-4, callback=odecb)

""" analysis """
pred = Array(sol)
#

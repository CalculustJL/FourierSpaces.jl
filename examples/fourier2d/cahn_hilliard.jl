#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, Test, Random

"""
Cahn-Hilliard equation

∂ₜu = MΔf(u) - κΔ²u 

x ∈ [0, L)ᵈ (periodic)

numerical method
https://doi.org/10.1016/S0898-1221(02)00142-6

model
https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
"""

Nx = Ny = 64
Lx = Ly = 50.0
p = nothing

""" space discr """
domain = IntervalDomain(0, Lx) × IntervalDomain(0, Ly)
V  = FourierSpace(Nx, Ny; domain = domain)
Vh = transform(V)
discr  = Collocation()

(x, y,) = points(V)
iftr = transformOp(Vh)
ftr  = transformOp(V)

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
u0 = ftr \ (ftr * u0)

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
A = -laplaceOp(V, discr)
B = biharmonicOp(V, discr)
F = FunctionOperator(dfdu, u0;)

L = cache_operator(-κ*B, u0)
N = cache_operator(M * A * F, u0)

""" time discr """
tspan = (0.0, 2.0)
tsave = range(tspan...; length=10)

op = cache_operator(L + F, u0)

odefun = ODEFunction(op; jac_prototype = op)
# odefun = SplitFunction(L, N; jac_prototype = L)

odealg = SSPRK43()
# odealg = TRBDF2(autodiff = false, linsolve = KrylovJL_GMRES())
# odealg = TRBDF2(autodiff = false, linsolve = IterativeSolversJL_GMRES())

odeprob = ODEProblem(odefun, u0, tspan, p)
odecb = begin
    affect!(int) = int.iter % 100 == 0 && println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

@time sol = solve(odeprob, odealg, saveat=tsave, abstol=1e-4, callback=odecb)
pred = Array(sol)

# err = open("error.txt", "w")
# try
#     sol = solve(odeprob, odealg)
# catch e
#     showerror(err, e, catch_backtrace())
# end
nothing
#

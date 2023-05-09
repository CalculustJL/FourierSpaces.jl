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
Kuramoto-Sivashinsky equation

∂ₜu + Δu + Δ²u + 1/2 |∇u|² = 0

x ∈ [0, L)ᵈ (periodic)

TODO: Compute Lyapunov exponent (maybe sensitivities) in 1D/ 2D

https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2014.0932
"""

N = 128
L = 10pi
p = nothing

""" space discr """
domain = IntervalDomain(0, L)
space = FourierSpace(N; domain = domain)
tspace = transform(space)
discr  = Collocation()

(x,) = points(space)
iftr = transformOp(tspace)
ftr  = transformOp(space)

Nic = 5
Random.seed!(1234)
u0 = rand(N)
û0 = ftr * u0
û0[Nic+1:end] .= 0
û0[1] = 0

function convect!(v, u, p, t)
    copy!(v, u)
end

Â = laplaceOp(tspace, discr) # -Δ
B̂ = biharmonicOp(tspace, discr) # Δ²
Ĉ = advectionOp((zero(û0),), tspace, discr; vel_update_funcs=(convect!,)) # uuₓ
F̂ = SciMLOperators.NullOperator(tspace) # F = 0

L = cache_operator(Â - B̂, û0)
N = cache_operator(-Ĉ + F̂, û0)

""" time discr """
tspan = (0.0, 100)
tsave = range(tspan...; length=100)
odealg = Tsit5()
odealg = SSPRK43()
prob = SplitODEProblem(L, N, û0, tspan, p)

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
pred = iftr.(sol.u, nothing, 0)
pred = hcat(pred...)

# plot(pred, label=nothing)

anim = animate(pred, space, sol.t)
gif(anim, joinpath(dirname(@__FILE__), "ks.gif"), fps=10)
#

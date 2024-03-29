#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, LinearAlgebra
using Plots, Test

"""
Kuramoto-Sivashinsky equation

∂ₜu + Δu + Δ²u + 1/2 |∇u|² = 0

x ∈ [0, L)ᵈ (periodic)

TODO: Compute Lyapunov exponent (maybe sensitivities) in 1D/ 2D

https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2014.0932
"""

N = 256
T = Float32
L = T(5pi)
tspan = (0.0, 10)
p = nothing

function uIC(x; μ=zero(T), σ=T(0.1))
    u = @. exp(-1f0/2f0 * ((x-μ)/σ)^2)
    # reshape(u, :, 1)
end

""" space discr """
domain = IntervalDomain(-L, L)
V = FourierSpace(N; domain = domain)
Vh = transform(V)
discr  = Collocation()

(x,) = points(V)
iftr = transformOp(Vh)
ftr  = transformOp(V)

û0 = ftr * uIC(x)

function convect!(v, u, p, t)
    copy!(v, u)
end

Â = laplaceOp(Vh, discr) # -Δ
B̂ = biharmonicOp(Vh, discr) # Δ²
Ĉ = advectionOp((zero(û0),), Vh, discr; vel_update_funcs! =(convect!,)) # uuₓ
F̂ = SciMLOperators.NullOperator(Vh) # F = 0

L = cache_operator(Â - B̂, û0)
N = cache_operator(-Ĉ + F̂, û0)

""" time discr """
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

anim = animate(pred, V, sol.t, title = "Kuramoto-Sivashinsky 1D")
gif(anim, joinpath(dirname(@__FILE__), "ks1.gif"), fps=25)
#

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

# Kuramoto-Sivashinsky equation
#
# ∂ₜu + Δu + Δ²u + 1/2 |∇u|^2 = 0
#
# x ∈ [0, L)ᵈ (periodic)
#
# Compute Lyapunov exponent (maybe sensitivities) in 1D/ 2D
#
# https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
# https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2014.0932
#

N = 128
p = nothing

""" space discr """
space  = FourierSpace(N)
tspace = transform(space)
discr  = Collocation()

(x,) = points(space)
iftr = transformOp(tspace)
ftr  = transformOp(space)

α = 1
u0 = @. sin(α*x)
û0 = ftr * u0

function convection!(v, u, p, t)
    copy!(v, u)
end

Â = -laplaceOp(tspace, discr)
B̂ = biharmonicOp(tspace, discr)
Ĉ = advectionOp((zero(û0),), tspace, discr; vel_update_funcs=(convection!,))
F̂ = SciMLOperators.NullOperator(tspace)

L̂ = cache_operator(Â + B̂, û0)
Ĝ = cache_operator(-Ĉ+F̂, û0)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
odealg = SSPRK43()
prob = SplitODEProblem(L̂, Ĝ, û0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-4)

""" analysis """
pred = iftr.(sol.u, nothing, 0)
pred = hcat(pred...)

anim = animate(pred, space, sol.t)
gif(anim, joinpath(dirname(@__FILE__), "ks.gif"), fps=20)
#

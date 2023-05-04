#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, ComponentArrays, LinearAlgebra
using Plots, Test

"""
Cahn-Hilliard equation

∂ₜu + Δu + Δ²u + = 0

x ∈ [0, L)ᵈ (periodic)

"""

nx = 32
ny = 32
p = nothing

Lx = 10.0
Ly = 10.0

""" space discr """
domain = IntervalDomain(0, Lx) ⊗ IntervalDomain(0, Ly)
space  = FourierSpace(N, domain = domain)
tspace = transform(space)
discr  = Collocation()

(x,) = points(space)
iftr = transformOp(tspace)
ftr  = transformOp(space)

A = -laplaceOp(tspace, discr)
B = biharmonicOp(tspace, discr)
C = advectionOp((zero(û0),), tspace, discr; vel_update_funcs=(convection!,))
F = SciMLOperators.NullOperator(tspace)

L = cache_operator(Â + B̂, û0)
N = cache_operator(-Ĉ+F̂, û0)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = ODEProblem(odefunc, û0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-8, reltol=1e-8)

""" analysis """
pred = [F,] .\ sol.u
pred = hcat(pred...)

utrue(t) = @. u0 * (exp(-ν*(α^2+β^2)*t))
ut = utrue(sol.t[1])
for i=2:length(sol.t)
    utt = utrue(sol.t[i])
    global ut = hcat(ut, utt)
end

err = norm(pred .- ut, Inf)
println("frobenius norm of error across time", err)
@test err < 1e-7
#

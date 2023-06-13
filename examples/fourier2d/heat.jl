#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, Test

nx = 32
ny = 32
ν = 1e-2
p = nothing

""" space discr """
domain = IntervalDomain(0, 2pi) × IntervalDomain(pi, 5pi)
V = FourierSpace(nx, ny; domain = domain)
discr = Collocation()

x, y = points(V)
ftr  = transformOp(V)

""" IC """
α = 5
β = 3
u0 = @. sin(α*x) * sin(β*y)

""" operators """
A = -diffusionOp(ν, V, discr)
F = SciMLOperators.NullOperator(V)

A = cache_operator(A, x)
F = cache_operator(F, x)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = SplitODEProblem(A, F, u0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, reltol=1e-8)

""" analysis """
pred = Array(sol)

utrue(t) = @. u0 * (exp(-ν*(α^2+β^2)*t))
ut = utrue(sol.t[1])
for i=2:length(sol.t)
    utt = utrue(sol.t[i])
    global ut = hcat(ut, utt)
end

err = norm(pred .- ut, Inf)
println("Frobenius norm of error across time ", err)
@test err < 1e-7

# anim = animate(pred, V)
# filename = joinpath(dirname(@__FILE__), "heat" * ".gif")
# gif(anim, filename, fps=5)
#

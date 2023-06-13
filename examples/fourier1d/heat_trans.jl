#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, LinearAlgebra
using Plots, Test

N = 128
ν = 1e-2
p = nothing

""" space discr """
V  = FourierSpace(N)
Vh = transform(V)
discr  = Collocation()

(x,) = points(V)
(k,) = points(Vh)
iftr = transformOp(Vh)
ftr  = transformOp(V)

α = 5
u0 = @. sin(α*x)
û0 = ftr * u0

Â = -diffusionOp(ν, Vh, discr)
F̂ = SciMLOperators.NullOperator(Vh)

Â = cache_operator(Â, k)
F̂ = cache_operator(F̂, k)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = SplitODEProblem(Â, F̂, û0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-9, reltol=1e-9)

""" analysis """
pred = iftr.(sol.u, nothing, 0)
pred = hcat(pred...)

utrue(t) = @. u0 * (exp(-ν*α^2*t))
ut = utrue(sol.t[1])
for i=2:length(sol.t)
    utt = utrue(sol.t[i])
    global ut = hcat(ut, utt)
end

plt = plot()
for i=1:length(sol.u)
    plot!(plt, x, pred[:,i], legend=false)
end
display(plt)

err = norm(pred .- ut, Inf)
@test err < 1e-8
#

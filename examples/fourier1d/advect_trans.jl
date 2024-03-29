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

N = 128
p = nothing

""" space discr """
V = FourierSpace(N)
Vh = transform(V)
discr = Collocation()

(x,) = points(V)
(k,) = modes(V)
F  = transformOp(V)

""" operators """
v = 1.0;
vel = @. x*0 + v
vels = (F * vel,)

Ĉ = advectionOp(vels, Vh, discr)
F̂ = -Ĉ
F̂ = cache_operator(F̂, im*k)

""" IC """
function uIC(x)
    @. sin(1x)
end
u0 = uIC(x)
û0 = F * u0

""" time discr """
tsave = 0: pi/4: 4pi
tspan = (tsave[begin], tsave[end])
odealg = Tsit5()
prob = ODEProblem(F̂, û0, tspan, p)
@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-10, reltol=1e-10)

""" analysis """
pred = (F,) .\ sol.u
pred = hcat(pred...)

utrue(x,v,t) = uIC(@. x - v*t)
utr = utrue(x,v,sol.t[1])
for i=2:length(sol.u)
    ut = utrue(x,v,sol.t[i])
    global utr = hcat(utr, ut)
end

plt = plot()
for i=1:length(sol.u)
    plot!(plt, x, pred[:,i], legend=true)
end
display(plt)

err = norm(pred .- utr,Inf)
display(err)
@test err < 1e-8
#

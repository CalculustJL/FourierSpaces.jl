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
V = FourierSpace(N)
V = make_transform(V)
discr = Collocation()

(x,) = points(V)
(k,) = modes(V)
ftr  = transformOp(V)

α = 2
uic(x) = @. sin(α*x)
function utrue(t,x)
    cos(t) * uic(x)
end

A = -diffusionOp(ν, V, discr)

f = @. x*0 + .1

function forcing(f, u, p, t)
    ui = -sin(t)*uic(x)
    ud = -ν*α*α*uic(x)*cos(t)

    ui - ud
end

F = forcingOp(f, V, discr; f_update_func = forcing)
dudt = cache_operator(A + F, x)

""" time discr """
u0 = uic(x)
tspan = (0.0, 2π)
tsave = range(tspan...; length=10)
odealg = Tsit5()
#odealg = SSPRK43()

odefunc = ODEFunction{false}(dudt)
prob = ODEProblem(odefunc, u0, tspan, p)
@time sol = solve(prob, odealg, saveat=tsave, reltol=1e-10, abstol=1e-10)

""" analysis """
pred = Array(sol)

ut = utrue(sol.t[1],x)
for i=2:length(sol.t)
    utt = utrue(sol.t[i],x)
    global ut = hcat(ut, utt)
end

plt = plot()
for i=1:length(sol.u)
    plot!(plt, x, sol.u[i], legend=false)
end
display(plt)

err = norm(pred .- ut, Inf)
display(err)
@test err < 1e-7
#

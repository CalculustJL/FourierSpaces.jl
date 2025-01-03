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
using MLDataDevices
using Plots, Test

using CUDA
CUDA.allowscalar(false)

N = 128
ν = 1f-2
p = nothing

""" space discr """
V = FourierSpace(N)
V = gpu_device()(V)
discr = Collocation()

(x,) = points(V)
(k,) = modes(V)
ftr  = transformOp(V)

α = 5
u0 = @. sin(α*x)

A = -diffusionOp(ν, V, discr)
F = SciMLOperators.NullOperator(V)

A = cache_operator(A, x)
F = cache_operator(F, x)

""" time discr """
tspan = (0f0, 10f0) 
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = SplitODEProblem(A, F, gpu_device()(u0), tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, reltol=1f-6)

""" analysis """
pred = cpu_device()(Array(sol))

u0 = Array(u0)
utrue(t) = @. u0 * (exp(-ν*α^2*t))
ut = utrue(sol.t[1])
for i=2:length(sol.t)
    utt = utrue(sol.t[i])
    global ut = hcat(ut, utt)
end

plt = plot()
x = Array(x)
for i=1:length(sol.u)
    plot!(plt, x, pred[:,i], legend=false)
end
display(plt)

err = norm(pred .- ut, Inf)
display(err)
@test err < 1e-5
#

#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, Plots

nx = 32
ny = 32
ν = 1e-2
p = nothing

""" space discr """
V = FourierSpace(nx, ny)
discr = Collocation()

x, y = points(V)

uIC(x,y) = @. sin(1x) * sin(1y)
u0 = uIC(x,y)

A = -diffusionOp(ν, V, discr)

velx = @. x*0 + 1.0
vely = @. x*0 + 1.0
C = advectionOp((velx, vely), V, discr)
F = -C

A = cache_operator(A, x)
F = cache_operator(F, x)

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = SplitODEProblem(A, F, u0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave)
pred = Array(sol)

anim = animate(pred, V)
filename = joinpath(dirname(@__FILE__), "advection_diffusion" * ".gif")
gif(anim, filename, fps=5)
#

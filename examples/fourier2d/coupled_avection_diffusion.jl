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
using ComponentArrays

nx = 32
ny = 32
ν = 1e-2
p = nothing

""" spatial discr """
V = FourierSpace(nx, ny)
discr = Collocation()

x, y = points(V)

Ax = -diffusionOp(ν, V, discr)
Ay = -diffusionOp(ν, V, discr)

Cx = advectionOp((zero(x), zero(x)), V, discr;
                 vel_update_funcs! =(
                                   (v,u,p,t) -> fill!(v, true),
                                   (v,u,p,t) -> fill!(v, true),
                                  )
                )

Cy = advectionOp((zero(x), zero(x)), V, discr;
                 vel_update_funcs! =(
                                   (v,u,p,t) -> fill!(v, true),
                                   (v,u,p,t) -> fill!(v, true),
                                  )
                )

Fx = -Cx
Fy = -Cy

Dtx = cache_operator(Ax+Fx, x)
Dty = cache_operator(Ay+Fy, x)

""" IC """
u0 = begin
    vx0 = @. sin(2x) * sin(2y)
    vy0 = @. sin(3x) * sin(3y)

    ComponentArray(vx=vx0, vy=vy0)
end

function ddt(du, u, p, t)
    Dtx(du.vx, u.vx, p, t)
    Dty(du.vy, u.vy, p, t)

    du
end

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=10)
odealg = Tsit5()
prob = ODEProblem(ddt, u0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave)

pred = Array(sol)
vx = @views pred[:vx, :]
vy = @views pred[:vy, :]

anim = animate(vx, V)
filename = joinpath(dirname(@__FILE__), "burgers_x" * ".gif")
gif(anim, filename, fps=5)

anim = animate(vy, V)
filename = joinpath(dirname(@__FILE__), "burgers_y" * ".gif")
gif(anim, filename, fps=5)
#

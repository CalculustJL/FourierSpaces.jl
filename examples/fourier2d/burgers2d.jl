#
"""
Solve the 2D Burgers equation

∂t(vx) = -(vx*∂x(vx) + vy*∂y(vx)) + ν*Δvx
∂t(vy) = -(vx*∂x(vy) + vy*∂y(vy)) + ν*Δvy
"""

using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, Plots
using ComponentArrays, CUDA

T = Float64
nx = ny = 256
N = nx * ny
ν = 1e-3 |> T
p = nothing

abstol = reltol = 1e-4 |> T

odealg = Tsit5()
odealg = SSPRK43()

""" spatial discr """
interval = IntervalDomain(0.0, 1.0; periodic = true)
domain = interval × interval
V = FourierSpace(nx, ny; domain) |> T
discr = Collocation()
x, y = points(V)

callback = begin
    function affect!(int)
        println(
                "[$(int.iter)] \t Time $(round(int.t; digits=8))s"
               )
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

""" IC """

function uIC(x, y; μ = 0.9) # 0.9 - 1.1
    u = @. μ * sin(2π * x) * sin(2π * y)
    u[x .> 0.5] .= 0
    u[y .> 0.5] .= 0

    ComponentArray(;vx = u, vy = copy(u))
end

u0 = uIC(x,y) .|> T

ps = ComponentArray(vel=u0)
V = make_transform(V, u0.vx; p=ps)

# GPU
#CUDA.allowscalar(false)
#V = V |> gpu
#x, y = points(V)
#u0 = u0 |> gpu
#ps = ps |> gpu

""" spce ops """
Ax = -diffusionOp(ν, V, discr)
Ay = -diffusionOp(ν, V, discr)

Cx = advectionOp((zero(x), zero(x)), V, discr;
    vel_update_funcs = (
        (v,u,p,t) -> copy!(v, p.vel.vx),
        (v,u,p,t) -> copy!(v, p.vel.vy),

        # (v,u,p,t) -> fill!(v, true),
        # (v,u,p,t) -> fill!(v, true),
    ),
)

Cy = advectionOp((zero(x), zero(x)), V, discr;
    vel_update_funcs = (
        (v,u,p,t) -> copy!(v, p.vel.vx),
        (v,u,p,t) -> copy!(v, p.vel.vy),

        # (v,u,p,t) -> fill!(v, true),
        # (v,u,p,t) -> fill!(v, true),
    ),
)

Dtx = cache_operator(Ax - Cx, x)
Dty = cache_operator(Ay - Cy, x)

# function ddt(du, u, p, t)
#     ps = ComponentArray(vel=u)
#
#     Dtx(du.vx, u.vx, ps, t)
#     Dty(du.vy, u.vy, ps, t)
#
#     du
# end

function ddt(u, p, t)
    ps = ComponentArray(vel=u)

    dvx = Dtx(u.vx, ps, t)
    dvy = Dty(u.vy, ps, t)

    ComponentArray(; vx = dvx, vy = dvy)
end

""" time discr """
tspan = (0.0, 1.0) .|> T
tsave = range(tspan...; length=100) .|> T
prob  = ODEProblem(ddt, u0, tspan, p)

# plt = reshape(uIC(x, y).vx, nx, ny) |> heatmap
# display(plt)

@time sol = solve(prob, odealg; saveat = tsave, abstol, reltol, callback)

pred = Array(sol)
vx = @views pred[:vx, :]
vy = @views pred[:vy, :]

anim = animate(vx, V, sol.t)
filename = joinpath(dirname(@__FILE__), "burgers_x" * ".gif")
gif(anim, filename, fps=20)

anim = animate(vy, V, sol.t)
filename = joinpath(dirname(@__FILE__), "burgers_y" * ".gif")
gif(anim, filename, fps=20)
#

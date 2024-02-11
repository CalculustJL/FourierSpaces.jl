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
using Test, Plots

nx = 128
ny = 128
ν = 0e0
p = nothing

""" space discr """
L = 1.0
interval = IntervalDomain(-L, L; periodic = true)
domain = interval × interval
V = FourierSpace(nx, ny; domain)
discr = Collocation()

(x,y) = points(V)

""" operators """
A = -diffusionOp(ν, V, discr)

vx = 0.25; velx = @. x*0 + vx
vy = 0.25; vely = @. x*0 + vy
C = advectionOp((velx, vely), V, discr)

odeOp = cache_operator(A - C, x)

function meshplt(
    x::AbstractMatrix,
    y::AbstractMatrix,
    u::AbstractMatrix;
    a::Real = 45, b::Real = 30, c = :grays, legend = false,
    kwargs...)
    plt = plot(x, y, u; c, camera = (a,b), legend, kwargs...)
    plt = plot!(plt, x', y', u'; c, camera = (a,b), legend, kwargs...)
end

""" IC """
function uIC(x, y; σ = 0.1, μ = (-0.5, -0.5))
    r2 = @. (x - μ[1])^2 + (y - μ[2])^2
    @. exp(-1/2 * r2/(σ^2))
end
u0 = uIC(x,y)

u0_re = reshape(u0, nx, ny)
x_re = reshape(x, nx, ny)
y_re = reshape(y, nx, ny)
u0_re = reshape(u0, nx, ny)

# plt = heatmap(u0_re) |> display
plt = meshplt(x_re, y_re, u0_re)
display(plt)
# error("")

""" time discr """
tspan = (0.0, 4.0)
tsave = LinRange(tspan..., 11)
odealg = Tsit5()
prob = ODEProblem(odeOp, u0, tspan, p)
@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-8, reltol=1e-8)

""" analysis """

utrue(x, y, vx, vy, t) = uIC(x .- vx * t, y .- vy * t)
utr = utrue(x,y,vx,vy,sol.t[1])
for i=2:length(sol.u)
    ut = utrue(x, y, vx, vy, sol.t[i])
    global utr = hcat(utr, ut)
end

pred = Array(sol)
anim = animate(pred, V)
filename = joinpath(dirname(@__FILE__), "advect" * ".gif")
gif(anim, filename, fps=5)

err = norm(pred .- utr, Inf)
@test err < 1e-8
#

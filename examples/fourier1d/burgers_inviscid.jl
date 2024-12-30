#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using CUDA, OrdinaryDiffEq, LinearAlgebra, Random
using MLDataDevices
using Plots#, BSON

Random.seed!(0)
CUDA.allowscalar(false)

N = 1024
K = 1
p = nothing

_pi = Float32(pi)

function uIC(V::FourierSpace, mu = [0.9 0.11])
    x = points(V)[1]

    u = @. 1 + mu/2 * (sin(2_pi * x - _pi/2) + 1)

    u[x .> 1f0, :] .= 1
    u
end

odecb = begin
    function affect!(int)
        if int.iter % 1 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function burgers_inviscid(N, mu = LinRange(0.5, 1.4, 10)', p = nothing;
    tspan=(0.f0, 0.5f0),
    tsave=100,
    odealg=SSPRK43(),
	device = cpu_device(),
)

    """ space discr """
    domain = IntervalDomain(0, 2; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (k,) = modes(V)
    (x,) = points(V)

    """ IC """
    u0 = uIC(V, mu)
    V  = make_transform(V, u0; p=p)

    """ operators """
    function burgers!(v, u, p, t)
        copyto!(v, u)
    end

    function forcing!(f, u, p, t)
        # f .= (x .+ _pi) ./ 2_pi
        # lmul!(false, f)
    end

    # model setup
    C = advectionOp((zero(u0),), V, discr; vel_update_funcs! = (burgers!,))
    F = forcingOp(zero(u0), V, discr; f_update_func! = forcing!)

    odefunc = cache_operator(-C+F, u0)

    V  = V  |> device
    u0 = u0 |> device

    """ time discr """
    tsave = range(tspan...; length=tsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)

    """ solve """
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)
    @show sol.retcode

	x = x     |> cpu_device()
    u = sol   |> Array
	t = sol.t |> cpu_device()
	V = V     |> cpu_device()

    (sol, V), (x, u, t,)
end

(sol, V), (x, u, t) = burgers_inviscid(N; device = cpu_device())

plt = plot(x, @view u[:, 1, begin:49:end])
display(plt)

# dir = joinpath(@__DIR__, "invisc_burg_re01k")
# mkpath(dir)
#
# name = joinpath(dir, "data.bson")
# BSON.@save name x u
#
# for k in 1:10
#     uk = @view u[:, k, :]
#     anim = animate(uk, V, t, legend=false, linewidth=2, color=:black, xlabel="x", ylabel="u(x,t)")
#     gif(anim, joinpath(dir, "traj_$(k).gif"), fps=20)
# end

nothing
#

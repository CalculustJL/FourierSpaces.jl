#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, LinearAlgebra, Random
using Plots, CUDA

Random.seed!(0)
CUDA.allowscalar(false)

N = 1024
Nmodes = 10
ν = 1f-3
p = nothing

function uIC(V::FourierSpace)
    x = points(V)[1]
    X = truncationOp(V, (Nmodes/N,))

    u0 = if x isa CUDA.CuArray
        X * CUDA.rand(size(x)...)
    else
        X * rand(size(x)...)
    end

    u0
end

odecb = begin
    function affect!(int)
        println(
                "[$(int.iter)] \t Time $(round(int.t; digits=8))s"
               )
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function solve_burgers1D(N, ν, p;
                         uIC=uIC,
                         tspan=(0f0, 10f0),
                         nsims=10,
                         nsave=100,
                         odealg=SSPRK43(),
                        )

    """ space discr """
    V = FourierSpace(N) |> gpu
    discr = Collocation()

    (x,) = points(V)

    """ IC """
    u0 = [uIC(V) for i=1:nsims]
    u0 = hcat(u0...)
    V = make_transform(V, u0; p=p)

    """ operators """
    function burgers!(v, u, p, t)
        copyto!(v, u)
    end

    function forcing!(f, u, p, t)
        lmul!(false, f)
    end

    A = -diffusionOp(ν, V, discr)
    C = advectionOp((zero(u0),), V, discr; vel_update_funcs=(burgers!,))
    F = forcingOp(zero(u0), V, discr; f_update_func=forcing!)
    odefunc = cache_operator(A-C+F, u0) |> ODEFunction

    """ time discr """
    tsave = range(tspan...; length=nsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)

    sol, V
end

sol, V = solve_burgers1D(N, ν, p)
V = cpu(V)
pred = Array(sol)
nothing
#

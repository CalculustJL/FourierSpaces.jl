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
using Plots

N = 1024
ν = 1e-2
p = nothing

Random.seed!(0)
function uIC(V::FourierSpace)
    x = points(V)[1]
    X = truncationOp(V, (1/8,))

    u0 = X * rand(size(x)...)
    u0 = @. cos(x+pi/2)
end

odecb = begin
    function affect!(int)
        if int.iter % 100 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function solve_burgers1D(N, ν, p;
                         uIC=uIC,
                         tspan=(0.0, 5.0),
                         nsims=1,
                         nsave=100,
                         odealg=SSPRK43(),
                        )

    """ space discr """
    V = FourierSpace(N)
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
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1e-8, abstol=1e-8)
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)

    sol, V
end

sol, V = solve_burgers1D(N, ν, p)
V = cpu(V)
pred = Array(sol)

anim = animate(pred[:,1,:], V, sol.t,
               legend=false, linewidth=2, color=:black, xlabel="x", ylabel="u(x,t)")
gif(anim, joinpath(dirname(@__FILE__), "burgers.gif"), fps=20)
#

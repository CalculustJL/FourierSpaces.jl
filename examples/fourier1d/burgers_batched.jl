#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random, Sundials
using Plots

N = 1024
ν = 1f-3
p = nothing

Random.seed!(0)
function uIC(V::FourierSpace)
    x = points(V)[1]
    X = truncationOp(V,(1//20,))

    u0 = X * rand(size(x)...)

    u0
end

function solve_burgers(N, ν, p;
                       uIC=uIC,
                       tspan=(0.0, 10.0),
                       nsims=2,
                       nsave=100,
                      )

    """ space discr """
    V = FourierSpace(N)
    discr = Collocation()
    (x,)  = points(V)

    """ IC """
    u0 = uIC(V)
    u0 = u0 * ones(1,nsims)
    V = make_transform(V, u0)

    """ operators """
    A = -diffusionOp(ν, V, discr)

    function burgers!(v, u, p, t)
        copy!(v, u)
    end

    function forcing!(f, u, p, t)
        lmul!(false, f)
#       f .= 1e-2*rand(length(f))
    end

    C = advectionOp((zero(u0),), V, discr; vel_update_funcs=(burgers!,))
    F = -C + forcingOp(zero(u0), V, discr; f_update_func=forcing!)

    A = cache_operator(A, u0)
    F = cache_operator(F, u0)

    """ time discr """
    odealg = CVODE_BDF(method=:Functional)
    tsave = range(tspan...; length=nsave)
    prob = SplitODEProblem(A, F, u0, tspan, p)
    @time sol = solve(prob, odealg, saveat=tsave)

    sol, V
end

sol, V = solve_burgers(N, ν, p)
nothing
#

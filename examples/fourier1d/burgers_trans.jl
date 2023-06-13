#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, Test

N = 1024
ν = 1e-3
p = nothing

odecb = begin
    function affect!(int)
        if int.iter % 100 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

""" space discr """
V  = FourierSpace(N)
Vh = transform(V)
discr  = Collocation()

(x,) = points(V)
(k,) = points(Vh)
F    = transformOp(V)

""" initial condition """
function uIC(V)
    x = points(V)[1]
    X = truncationOp(V, (32/N,))

    u0 = X * rand(size(x)...)
end
u0 = uIC(V)
û0 = F * u0

function burgers!(v, u, p, t)
    copy!(v, u)
end

function forcing!(v, u, p, t)
    lmul!(false, v)
end

Â = -diffusionOp(ν, Vh, discr)
Ĉ = advectionOp((zero(û0),), Vh, discr; vel_update_funcs=(burgers!,))
F̂ = forcingOp(zero(û0), Vh, discr; f_update_func=forcing!)

Dt = cache_operator(Â-Ĉ+F̂, û0)

Dt_implicit = cache_operator(Â, û0)
Dt_explicit = cache_operator(-Ĉ+F̂, û0)

dudt_implicit = function(du, u, p, t)
    Dt_implicit(du, u, p, t)
end

dudt_explicit = function(du, u, p, t)
    Dt_explicit(du, u, p, t)
end

Dt = SplitFunction(dudt_implicit, dudt_explicit; jac_prototype=Diagonal(û0))

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=100)
odealg = Tsit5()
# odealg = TRBDF2(autodiff=false, linsolve=KrylovJL_GMRES())
prob = ODEProblem(Dt, û0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, callback=odecb)

""" analysis """
pred = [F,] .\ sol.u
pred = hcat(pred...)

anim = animate(pred, V, sol.t)
gif(anim, joinpath(dirname(@__FILE__), "burg_trans.gif"), fps=20)
#

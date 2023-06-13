#
using FourierSpaces
let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(FourierSpaces)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
    nothing
end

# TODO - error in mul!(_, Ĉ, û)

using OrdinaryDiffEq, LinearAlgebra
using Plots, Test

N = 1024
Nmodes = 10
ν = 1e-3
p = nothing

odecb = begin
    function affect!(int)
        println(
                "[$(int.iter)] \t Time $(round(int.t; digits=8))s"
               )
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

""" space discr """
V  = FourierSpace(N) |> gpu
Vh = transform(V)
discr  = Collocation()

(x,) = points(V)
(k,) = points(Vh)
F    = transformOp(V)

""" initial condition """
function uIC(V::FourierSpace)
    x = points(V)[1]
    X = truncationOp(V, (Nmodes/N,))

    u0 = X * CUDA.rand(size(x)...)
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

""" time discr """
tspan = (0.0, 10.0)
tsave = range(tspan...; length=100)
odealg = SSPRK43()
prob = ODEProblem(Dt, û0, tspan, p)

@time sol = solve(prob, odealg, saveat=tsave, callback=odecb, reltol=1f-6, abstol=1f-6)

""" analysis """
pred = [F,] .\ sol.u
pred = hcat(pred...)

anim = animate(pred, V, sol.t)
gif(anim, joinpath(dirname(@__FILE__), "burg_trans.gif"), fps=20)
#

#
###
# operators in phys space
###

Spaces.transformOp(V::FourierSpace) = V.ftransform

function Spaces.truncationOp(V::FourierSpace, fracs=nothing)
    X = truncationOp(transform(V), fracs)

    if X isa IdentityOperator
        return IdentityOperator(V)
    end

    F = transformOp(V)

    F \ X * F
end

function Spaces.massOp(V::FourierSpace, ::Galerkin)
    w = V.mass_mat
    DiagonalOperator(w)
end

function Spaces.gradientOp(V::FourierSpace{<:Any,D}) where{D}
    Vh = transform(V)
    DDh = gradientOp(Vh)   # ∇ in transformed space

    F  = transformOp(V)    # forward transform
    FF = [F for _ in 1:D]

    # https://github.com/vpuri3/PDEInterfaces.jl/issues/25
    FF .\ DDh .* FF # TODO - this is doing transform D times. should only be 1
end

function Spaces.hessianOp(V::FourierSpace{<:Any,D}) where{D}
    Vh  = transform(V)
    DD2h = hessianOp(Vh)

    F  = transformOp(V)
    FF = [F for i=1:D]

    FF .\ DD2h .* FF
end

function Spaces.laplaceOp(V::FourierSpace{<:Any,D}, discr::Collocation) where{D}
    Vh = transform(V)
    D2h = laplaceOp(Vh, discr)

    F = transformOp(V)

    F \ D2h * F
end

function Spaces.biharmonicOp(V::FourierSpace{<:Any,D}) where{D}
    Vh  = transform(V)
    DD4h = biharmonicOp(Vh)

    F  = transformOp(V)
    FF = [F for i=1:D]

    FF .\ DD4h .* FF
end

function _fusedGradientTruncationOp(V::FourierSpace{<:Any,D},
                                    truncation_fracs=nothing,
                                   ) where{D}
    Vh = transform(V)

    F   = transformOp(V)
    Xh  = truncationOp(Vh, truncation_fracs)
    DDh = gradientOp(Vh)

    FF  = [F  for i=1:D]
    XXh = [Xh for i=1:D]

    FF .\ XXh .* DDh .* FF
end

function Spaces.advectionOp(vels::NTuple{D},
    W::FourierSpace{<:Any,D},
    discr::AbstractDiscretization;
    vel_update_funcs = nothing,
    vel_update_funcs! = nothing,
    truncation_fracs = nothing,
) where{D}

    VV = Spaces._pair_update_funcs(vels, vel_update_funcs, vel_update_funcs!)
    M  = massOp(W, discr)

    # TODO - does truncation make sense for galerkin?

    C = if M isa IdentityOperator # Collocation
        Wh = transform(W)

        F   = transformOp(W)
        Xh  = truncationOp(Wh, truncation_fracs)
        DDh = gradientOp(Wh)

        FF  = [F  for _ in 1:D]
        XXh = [Xh for _ in 1:D]

        VV' * (Diagonal(FF) \ Diagonal(XXh) * Diagonal(DDh)) * FF
    else # Galerkin
        X   = truncationOp(W, truncation_fracs)
        XDD = _fusedGradientTruncationOp(W, truncation_fracs)

        XX = Diagonal([X for i=1:D])
        MM = Diagonal([M for i=1:D])
        (XX*VV)' * MM * XDD
    end

    C
end

function Spaces.interpOp(V1::FourierSpace{T1,1},
                         V2::FourierSpace{T2,1}) where{T1,T2}

    # TODO: interpOp - assert domains match
    # TODO: interpOp - if V1 == V2 return IdentityOperator(V1)

    F1   = transformOp(V1)
    F2   = transformOp(V2)
    sp1h = transform(V1)
    sp2h = transform(V2)

    J = interpOp(sp1h, sp2h)

    N1 = length(V1)
    N2 = length(V2)
    λ = T1(N2 / N1) # TODO - verify in AbstractFFTs documentation

    λ * F1 \ J * F2
end

#

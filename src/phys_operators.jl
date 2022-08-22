#
###
# operators in phys space
###

Spaces.transformOp(space::FourierSpace) = space.ftransform

function Spaces.truncationOp(space::FourierSpace, fracs=nothing)
    X = truncationOp(transform(space), fracs)

    if X isa IdentityOperator
        return IdentityOperator(space)
    end

    F = transformOp(space)

    F \ X * F
end

function Spaces.massOp(space::FourierSpace, ::Galerkin)
    w = mass_matrix(space)
    DiagonalOperator(w)
end

function Spaces.gradientOp(space::FourierSpace{<:Any,D}) where{D}
    sph = transform(space)  # transformed space
    DDh = gradientOp(sph)   # ∇ in transformed space

    F  = transformOp(space) # forward transform
    FF = [F for i=1:D]

    # https://github.com/vpuri3/PDEInterfaces.jl/issues/25
    FF .\ DDh .* FF # TODO - this is doing transform D times. should only be 1
end

function Spaces.hessianOp(space::FourierSpace{<:Any,D}) where{D}
    sph  = transform(space)
    DD2h = hessianOp(sph)

    F  = transformOp(space)
    FF = [F for i=1:D]

    FF .\ DD2h .* FF
end

function Spaces.laplaceOp(space::FourierSpace{<:Any,D}, discr::Collocation) where{D}
    sph = transform(space)
    D2h = laplaceOp(sph, discr)

    F = transformOp(space)

    F \ D2h * F
end

function Spaces.biharmonicOp(space::FourierSpace{<:Any,D}) where{D}
    sph  = transform(space)
    DD4h = biharmonicOp(sph)

    F  = transformOp(space)
    FF = [F for i=1:D]

    FF .\ DD4h .* FF
end

function _fusedGradientTruncationOp(space::FourierSpace{<:Any,D},
                                    truncation_fracs=nothing,
                                   ) where{D}
    tspace = transform(space)

    F   = transformOp(space)
    Xh  = truncationOp(tspace, truncation_fracs)
    DDh = gradientOp(tspace)

    FF  = [F  for i=1:D]
    XXh = [Xh for i=1:D]

    FF .\ XXh .* DDh .* FF
end

function Spaces.advectionOp(vels::NTuple{D},
                            space::FourierSpace{<:Any,D},
                            discr::Spaces.AbstractDiscretization;
                            vel_update_funcs=nothing,
                            truncation_fracs=nothing,
                           ) where{D}

    VV = _pair_update_funcs(vels, vel_update_funcs)
    M  = massOp(space, discr)

    C = if M isa IdentityOperator # Collocation
        tspace = transform(space)

        F   = transformOp(space)
        Xh  = truncationOp(tspace, truncation_fracs)
        DDh = gradientOp(tspace)

        FF  = [F  for i=1:D]
        XXh = [Xh for i=1:D]

        VV' * (Diagonal(FF) \ Diagonal(XXh) * Diagonal(DDh)) * FF
    else # Galerkin
        X   = truncationOp(space, truncation_fracs)
        XDD = _fusedGradientTruncationOp(space, truncation_fracs)

        XX = Diagonal([X for i=1:D])
        MM = Diagonal([M for i=1:D])
        (XX*VV)' * MM * XDD
    end

    C
end

function Spaces.interpOp(space1::FourierSpace{T1,1},
                         space2::FourierSpace{T2,1}) where{T1,T2}

    F1   = transformOp(space1)
    F2   = transformOp(space2)
    sp1h = transform(space1)
    sp2h = transform(space2)

    J = interpOp(sp1h, sp2h)

    N1 = length(space1)
    N2 = length(space2)
    λ = T1(N2 / N1) # TODO - verify in AbstractFFTs documentation

    λ * F1 \ J * F2
end

#

#
###
# operators in transformed space
###

function Spaces.truncationOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace},
                             fracs::Union{NTuple{D,Number},Nothing}=nothing) where{D}

    fracs = fracs isa Nothing ? ([2//3 for d=1:D]) : fracs

    if isone(prod(fracs))
        return IdentityOperator(space)
    end

    ns = size(space)

    a = ones(Bool, ns)
    for d=1:D
        n = ns[d]
        frac = fracs[d]

        idx = if d == 1
            cut = round(Int, n*frac, RoundUp)

            cut : n
        else
            cut = round(Int, n ÷ 2 * frac, RoundUp)

            cut : n-cut
        end

        a[(Colon() for i=1:d-1)..., idx, (Colon() for i=d+1:D)...] .= false
    end
    a = points(space)[1] isa GPUArraysCore.AbstractGPUArray ? gpu(a) : a

    DiagonalOperator(vec(a))
end

function Spaces.gradientOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}) where{D}
    ks = points(space)
    ns = size(transform(space))

    # https://math.mit.edu/~stevenj/fft-deriv.pdf
    gdiags = [@. im*ks[i] for i=1:D]
    for d = 1:D
        iseven(ns[d]) && GPUArraysCore.@allowscalar gdiags[d][end] = 0
    end

    DD = AbstractSciMLOperator[]
    push!(DD, DiagonalOperator.(gdiags)...)
end

function Spaces.hessianOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}) where{D}

    DD = gradientOp(space)
    DD_ = reshape(DD, (1, D))
    HH = DD * DD_

    # diagonals, ∂xx, require special treatment
    ks = points(space)
    hdiags = [@. -ks[i]^2 for i=1:D]
    Hdiags = DiagonalOperator.(hdiags)

    for d in 1:D
        HH[d, d] = Hdiags[d]
    end

    HH
end

function Spaces.laplaceOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}, ::Collocation) where{D}
    ks = points(space)
    ldiag = sum([@. ks[i]^2 for i=1:D])

    ldiag_ = copy(ldiag)
    GPUArraysCore.@allowscalar ldiag_[1] = Inf

    L = DiagonalOperator(ldiag)
    Li = DiagonalOperator(inv.(ldiag_)) |> InvertedOperator

    InvertibleOperator(L, Li)
end

function Spaces.biharmonicOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}, ::Collocation) where{D}
    ks = points(space)
    ldiag = sum([@. ks[i]^2 for i=1:D])
    bdiag = ldiag .^ 2

    bdiag_ = copy(bdiag)
    GPUArraysCore.@allowscalar bdiag_[1] = Inf

    L = DiagonalOperator(bdiag)
    Li = DiagonalOperator(inv.(bdiag_)) |> InvertedOperator

    InvertibleOperator(L, Li)
end

function Spaces.advectionOp(vels::NTuple{D},
                            tspace::Spaces.TransformedSpace{T,D,<:FourierSpace},
                            discr::Spaces.AbstractDiscretization;
                            vel_update_funcs=nothing,
                            truncation_fracs=nothing,
                           ) where{T,D}

    space = transform(tspace)

    F  = transformOp(space)
    Xh = truncationOp(tspace, truncation_fracs)
    M  = massOp(space, discr)

    VV = begin
        trunc = cache_operator(F \ Xh, vels[1])
        vel_funcs = Spaces.ComposedUpdateFunction.((trunc,), vel_update_funcs, deepcopy(vels))

        Spaces._pair_update_funcs((F,) .\ vels, vel_funcs)
    end

    MM  = Diagonal([M  for i=1:D])
    XXh = Diagonal([Xh for i=1:D])
    FFi = Diagonal([inv(F)  for i=1:D])
    DDh = gradientOp(tspace, discr)

    F * (VV' * MM * FFi * XXh * DDh)
end

# interpolation
function Spaces.interpOp(space1::Spaces.TransformedSpace{<:Any,D,<:FourierSpace},
                         space2::Spaces.TransformedSpace{<:Any,D,<:FourierSpace},
                        ) where{D}

    Ms = size(space1) # output
    Ns = size(space2) # input

    sz = Tuple(zip(Ms,Ns))
    Js = Tuple(sparse(I, sz[i]) for i=1:D)

    ⊗(Js...)
end
#

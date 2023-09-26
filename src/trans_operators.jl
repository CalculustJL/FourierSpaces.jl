#
###
# operators in transformed space
###

function Spaces.truncationOp(Vh::TransformedFourierSpace{<:Any,D},
                             fracs::Union{NTuple{D}, Nothing}=nothing) where{D}

    fracs = fracs isa Nothing ? ([2//3 for _ in 1:D]) : fracs

    if isone(prod(fracs))
        return IdentityOperator(Vh)
    end

    ns = size(Vh)
    diag = ones(Bool, ns)

    for d in 1:D
        n = ns[d]
        frac = fracs[d]

        idx = if d == 1
            cut = round(Int, n*frac, RoundUp)

            cut : n
        else
            cut = round(Int, n ÷ 2 * frac, RoundUp)

            cut : n-cut
        end

        diag[(Colon() for _ in 1:d-1)..., idx, (Colon() for i=d+1:D)...] .= false
    end
    diag = points(Vh)[1] isa AbstractGPUArray ? gpu(diag) : diag

    DiagonalOperator(vec(diag))
end

function Spaces.gradientOp(Vh::TransformedFourierSpace{<:Any,D}) where{D}
    ks = points(Vh)
    ns = size(transform(Vh))

    # https://math.mit.edu/~stevenj/fft-deriv.pdf
    gdiags = [@. im*ks[i] for i=1:D]
    for d = 1:D
        iseven(ns[d]) && @allowscalar gdiags[d][end] = 0
    end

    # inverse
    gdiags_ = deepcopy(gdiags)
    for d in 1:D
        diag = gdiags_[d]
        I = findall(iszero, diag)
        @allowscalar diag[I] .= Inf
    end
    gdiags_ = [inv.(diag) for diag in gdiags_]

    Ds  = DiagonalOperator.(gdiags)
    Ds_ = DiagonalOperator.(gdiags_) .|> InvertedOperator
    Ls  = InvertibleOperator.(Ds, Ds_)

    AbstractSciMLOperator[Ls...]
end

function Spaces.hessianOp(Vh::TransformedFourierSpace{<:Any,D}) where{D}

    DD = gradientOp(Vh)
    DD_ = reshape(DD, (1, D))
    HH = DD * DD_

    # diagonals, ∂xx, require special treatment
    ks = points(Vh)
    hdiags = [@. -ks[i]^2 for i=1:D]
    Hdiags = DiagonalOperator.(hdiags)

    for d in 1:D
        HH[d, d] = Hdiags[d]
    end

    HH
end

function Spaces.laplaceOp(Vh::TransformedFourierSpace{<:Any,D}, ::Collocation) where{D}
    ks = points(Vh)
    ldiag = sum([@. ks[i]^2 for i=1:D])

    ldiag_ = deepcopy(ldiag)
    @allowscalar ldiag_[1] = Inf

    L = DiagonalOperator(ldiag)
    Li = DiagonalOperator(inv.(ldiag_)) |> InvertedOperator

    InvertibleOperator(L, Li)
end

function Spaces.biharmonicOp(Vh::TransformedFourierSpace{<:Any,D}, ::Collocation) where{D}
    ks = points(Vh)
    ldiag = sum([@. ks[i]^2 for i=1:D])
    bdiag = ldiag .^ 2

    bdiag_ = deepcopy(bdiag)
    @allowscalar bdiag_[1] = Inf

    L = DiagonalOperator(bdiag)
    Li = DiagonalOperator(inv.(bdiag_)) |> InvertedOperator

    InvertibleOperator(L, Li)
end

function Spaces.advectionOp(vels::NTuple{D},
    Wh::TransformedFourierSpace{T,D},
    discr::AbstractDiscretization;
    vel_update_funcs = nothing,
    vel_update_funcs! = nothing,
    truncation_fracs= nothing,
) where{T,D}

    W = transform(Wh)

    # phys ops
    F = transformOp(W)
    M = massOp(W, discr)

    # trans ops
    Xh = truncationOp(Wh, truncation_fracs)

    VV = begin
        trunc = cache_operator(F \ Xh, vels[1])
        vel_funcs  = Spaces.ComposedUpdateFunction.((trunc,), vel_update_funcs , deepcopy(vels))
        vel_funcs! = Spaces.ComposedUpdateFunction.((trunc,), vel_update_funcs!, deepcopy(vels))

        Spaces._pair_update_funcs((F,) .\ vels, vel_funcs, vel_funcs!)
    end

    MM  = Diagonal([M  for i=1:D])
    XXh = Diagonal([Xh for i=1:D])
    FFi = Diagonal([inv(F)  for i=1:D])
    DDh = gradientOp(Wh, discr)

    F * (VV' * MM * FFi * XXh * DDh)
end

# interpolation
function Spaces.interpOp(Vh1::TransformedFourierSpace{<:Any,D},
                         Vh2::TransformedFourierSpace{<:Any,D},
                        ) where{D}

    Ms = size(Vh1) # output
    Ns = size(Vh2) # input

    sz = Tuple(zip(Ms,Ns))
    Js = Tuple(sparse(I, sz[i]) for i=1:D)

    ⊗(Js...)
end
#

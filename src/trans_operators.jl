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
    a = points(space)[1] isa CUDA.CuArray ? gpu(a) : a

    DiagonalOperator(vec(a))
end

function Spaces.gradientOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}) where{D}
    ks = points(space)
    ns = size(transform(space))

    # https://math.mit.edu/~stevenj/fft-deriv.pdf
    iks = [@. im*ks[i] for i=1:D]
    for i=1:D iseven(ns[i]) && CUDA.@allowscalar iks[i][end] = 0  end

    DiagonalOperator.(iks)
end

function Spaces.hessianOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}) where{D}
    ks = points(space)
    ik2s = [@. -ks[i]^2 for i=1:D]

    DiagonalOperator.(ik2s)
end

function Spaces.laplaceOp(space::Spaces.TransformedSpace{<:Any,D,FourierSpace}, ::Collocation) where{D}
    ks = points(space)
    ik2 = [@. -ks[i]^2 for i=1:D] |> sum

    DiagonalOperator.(ik2)
end

function Spaces.biharmonicOp(space::Spaces.TransformedSpace{<:Any,D,<:FourierSpace}) where{D}
    ks = points(space)
    ik4s = [@. ks[i]^4 for i=1:D]

    DiagonalOperator.(ik4s)
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
        vel_funcs = ComposedUpdateFunction.((trunc,), vel_update_funcs, deepcopy(vels))

        _pair_update_funcs((F,) .\ vels, vel_funcs)
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

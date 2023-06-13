#
function (::Type{T})(V::FourierSpace) where{T<:Number}
    npoints = size(V)
    nfreqs  = mode_size(V)
    dom     = T(domain(V))

    grid  = Tuple(T.(x) for x in points(V))
    freqs = Tuple(T.(k) for k in modes(V))

    mass_mat = T.(V.mass_mat)
    ftransform = transformOp(V)

    V = FourierSpace(
                     npoints, nfreqs, dom, grid, freqs,
                     mass_mat, nothing,
                    )

    p   = nothing # TODO
    u0  = T.(ftransform.cache[1])

    make_transform(V, u0; p=p)
end

function Adapt.adapt_structure(to, V::FourierSpace)
    grid  = adapt_structure(to, points(V))
    freqs = adapt_structure(to, modes(V))
    mass_mat = adapt_structure(to, V.mass_mat)

    x = first(grid)
    T = eltype(x)

    npoints = adapt_structure(to, size(V))
    nfreqs  = adapt_structure(to, mode_size(V))
    dom     = adapt_structure(to, T(domain(V)))
    ftransform = transformOp(V)

    V = FourierSpace(
                     npoints, nfreqs, dom, grid, freqs,
                     mass_mat, nothing,
                    )

    p   = adapt_structure(to, ftransform.p)
    u0  = adapt_structure(to, ftransform.cache[1])

    make_transform(V, u0; p=p)
end
#

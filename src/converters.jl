#
function (::Type{T})(space::FourierSpace) where{T<:Number}
    npoints = size(space)
    nfreqs  = mode_size(space)
    dom     = T(domain(space))

    grid  = Tuple(T.(x) for x in points(space))
    freqs = Tuple(T.(k) for k in modes(space))

    mass_mat = T.(mass_matrix(space))
    ftransform = transformOp(space)

    space = FourierSpace(
                         npoints, nfreqs, dom, grid, freqs,
                         mass_mat, nothing,
                        )

    p   = nothing # TODO
    u0  = T.(ftransform.cache[1])

    make_transform(space, u0; p=p)
end

function Adapt.adapt_structure(to, space::FourierSpace)
    grid  = adapt_structure(to, points(space))
    freqs = adapt_structure(to, modes(space))
    mass_mat = adapt_structure(to, mass_matrix(space))

    x = first(grid)
    T = eltype(x)

    npoints = adapt_structure(to, size(space))
    nfreqs  = adapt_structure(to, mode_size(space))
    dom     = adapt_structure(to, T(domain(space)))
    ftransform = transformOp(space)

    space = FourierSpace(
                         npoints, nfreqs, dom, grid, freqs,
                         mass_mat, nothing,
                        )

    p   = adapt_structure(to, ftransform.p)
    u0  = adapt_structure(to, ftransform.cache[1])

    make_transform(space, u0; p=p)
end
#

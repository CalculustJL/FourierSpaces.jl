#
"""
Fourier spectral space
"""
struct FourierSpace{
                    T,
                    D,
                    Tnpoints<:NTuple{D,Integer},
                    Tnfreqs<:NTuple{D,Integer},
                    Tdom<:Domains.AbstractDomain{T,D},
                    Tgrid<:NTuple{D,AbstractArray{T}},
                    Tfreqs<:NTuple{D,AbstractArray{T}},
                    Tmass_mat<:AbstractArray{T},
                    Tftransform,
                   } <: Spaces.AbstractSpace{T,D}
    """ # points """
    npoints::Tnpoints
    """ # freq size """
    nfreqs::Tnfreqs
    """ Domain """
    dom::Tdom
    """ grid points """
    grid::Tgrid
    """ frequencies """
    freqs::Tfreqs
    """ mass matrix """
    mass_mat::Tmass_mat
    """ forward transform `mul!(û, T , u)` """
    ftransform::Tftransform
end

###
# 1D
###

function FourierSpace(n::Integer;
                      domain::AbstractDomain{<:Any,1} = FourierDomain(1),
                      T::Type{<:Real} = Float64,
                     )

    if !isa(domain, Domains.LogicallyRectangularDomain)
        msg = """Trigonometric polynomials work with logically rectangular
            domains. `domain` must either be a `Domains.IntervalDomain`,
            or product of interval domains created with `LinearAlgebra.×`.
            Optionally `domain` may be a `Domains.MappedDomain` generated
            as `Domains.deform(dom, mapping)`.
            """
        throw(ArgumentError(msg))
    end

    # check for deformation
    dom, mapping = if domain isa Domains.MappedDomain
        domain.domain, domain.mapping
    else
        domain, nothing
    end

    # put domain in a box
    if dom isa IntervalDomain
        dom = ProductDomain(dom)
    end

    @assert dom isa Domains.BoxedDomain

    # get end points
    bd = boundaries(dom.domains[1])
    z0, z1 = convert.(Number, bd)

    # get points
    (L,) = expanse(dom)
    dz = L / n
    z = linspace(z0, z1 - dz, n, T)

    # establish frequencies
    k = AbstractFFTs.rfftfreq(n, 2π * n / L) |> Array

    npoints = (n,)
    nfreqs  = (length(k),)
    dom = T(dom)
    grid = (T.(z),)
    freqs = (T.(k),)
    mass_mat = ones(T, n) * T(2π / L)
    ftransform = nothing

    V = FourierSpace(
                     npoints, nfreqs, dom, grid, freqs,
                     mass_mat, ftransform,
                    )

    V = make_transform(V, z)

    isnothing(mapping) ? V : deform(V, mapping)
end

###
## 2D
###

function FourierSpace(nr::Integer, ns::Integer;
                      domain::Domains.AbstractDomain{<:Any,2} = FourierDomain(2),
                      T::Type{<:Number} = Float64,
                     )

    if !isa(domain, Domains.LogicallyRectangularDomain)
        msg = """Trigonometric polynomials work with logically rectangular
            domains. `domain` must be a product of `Domains.IntervalDomain`
            created with `LinearAlgebra.×`. Optionally `domain` may be a
            `Domains.MappedDomain` generated as `Domains.deform(dom, map)`.
            """
        throw(ArgumentError(msg))
    end

    # check for deformation
    dom, mapping = if domain isa Domains.MappedDomain
        domain.domain, domain.mapping
    else
        domain, nothing
    end

    # put domain in a box
    @assert isa(dom, Domains.BoxedDomain)

    # get end points
    bd_r = boundaries(dom.domains[1])
    bd_s = boundaries(dom.domains[2])

    r0, r1 = convert.(Number, bd_r)
    s0, s1 = convert.(Number, bd_s)

    # get points
    (Lr, Ls) = expanse(dom)

    dr = Lr / nr
    ds = Ls / ns

    zr = linspace(r0, r1 - dr, nr, T)
    zs = linspace(s0, s1 - ds, ns, T)

    # establish frequencies
    kr = AbstractFFTs.rfftfreq(nr, 2π * nr / Lr) |> Array
    ks = AbstractFFTs.fftfreq( ns, 2π * ns / Ls) |> Array

    nkr = length(kr)
    nks = length(ks)

    r, s   = vec.(ndgrid(zr, zs))
    kr, ks = vec.(ndgrid(kr, ks))

    npoints = (nr, ns)
    nfreqs  = (nkr, nks)
    dom  = T(dom)
    grid    = (T.(r), T.(s))
    freqs   = (T.(kr), T.(ks))
    mass_mat = ones(T, nr * ns) * T(2π / Lr) * T(2π / Ls)
    ftransform  = nothing

    V = FourierSpace(
                     npoints, nfreqs, dom, grid, freqs,
                     mass_mat, ftransform,
                    )

    V = make_transform(V, r)

    isnothing(mapping) ? V : deform(V, mapping)
end
#

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
                      domain::Domains.AbstractDomain{<:Any,1} = FourierDomain(1),
                      T::Type{<:Real} = Float64,
                     )

    dom = if domain isa IntervalDomain
        BoxDomain(domain)
    elseif domain isa BoxDomain
        domain
    else
        @error "Trigonometric polynomials work with logically rectangular domains"
    end

    (L,) = lengths(dom)

    # TOOD reset deformation to map from [-π,π]^D
    # ref_dom = reference_box(2)
    # dom = ref_dom # map_from_ref(dom, ref_dom) # TODO

    dz = L / n
    z = linspace(-L/2, L/2-dz, n, T)

    FFTLIB = _fft_lib(z)
    k = FFTLIB.rfftfreq(n, 2π * n / L) |> Array

    npoints = (n,)
    nfreqs  = (length(k),)
    dom = T(dom)
    grid = (z,)
    freqs = (k,)
    mass_mat = ones(T, n) * (2π / L)
    ftransform = nothing

    space = FourierSpace(
                         npoints, nfreqs, dom, grid, freqs,
                         mass_mat, ftransform,
                        )

    space = make_transform(space, z)

    dom isa Domains.DeformedDomain ? deform(space, mapping) : space
end

###
## 2D
###

function FourierSpace(nr::Integer, ns::Integer;
                      domain::Domains.AbstractDomain{<:Any,2}=FourierDomain(2),
                      T::Type{<:Number} = Float64,
                     )

    dom = if domain isa BoxDomain
        domain
    else
        @error "Trigonometric polynomials work with logically rectangular domains"
    end

    (Lr, Ls) = lengths(dom)

    # TODO - reset deformation to map from [-π,π]^D
    # ref_dom = reference_box(2)
    # dom = ref_dom # map_from_ref(dom, ref_dom) # TODO

    dr = Lr / nr
    ds = Ls / ns
    zr = linspace(-Lr/2, Lr/2-dr, nr, T)
    zs = linspace(-Ls/2, Ls/2-ds, ns, T)

    FFTLIB = _fft_lib(zr)
    kr = FFTLIB.rfftfreq(nr, 2π * nr / Lr) |> Array
    ks = FFTLIB.fftfreq( ns, 2π * ns / Ls) |> Array
    nkr = length(kr)
    nks = length(ks)

    r, s   = vec.(ndgrid(zr, zs))
    kr, ks = vec.(ndgrid(kr, ks))

    npoints = (nr, ns)
    nfreqs  = (nkr, nks)
    dom  = T(dom)
    grid    = (r, s)
    freqs   = (kr, ks)
    mass_mat = ones(T, nr * ns) * (2π / Lr) * (2π / Ls)
    ftransform  = nothing

    space = FourierSpace(
                         npoints, nfreqs, dom, grid, freqs,
                         mass_mat, ftransform,
                        )

    space = make_transform(space, r)

    dom isa Domains.DeformedDomain ? deform(space, mapping) : space
end
#

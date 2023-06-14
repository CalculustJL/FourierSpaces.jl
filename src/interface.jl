#
Spaces.size(V::FourierSpace) = V.npoints

Spaces.domain(V::FourierSpace) = V.dom

Spaces.points(V::FourierSpace) = V.grid

Spaces.global_numbering(V::FourierSpace) = 1:length(V)

Spaces.boundary_nodes(::FourierSpace) = ()

function Spaces.quadratures(V::FourierSpace{<:Any,1})
    x = points(V) |> first
    w = V.mass_mat

    ((x, w),)
end

# TODO: Spaces.quadratures for ND FourierSpaces.

Spaces.modes(V::FourierSpace) = V.freqs
Spaces.mode_size(V::FourierSpace) = V.nfreqs
#

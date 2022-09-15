#
Spaces.size(space::FourierSpace) = space.npoints

Spaces.mode_size(space::FourierSpace) = space.nfreqs

Spaces.domain(space::FourierSpace) = space.dom

Spaces.points(space::FourierSpace) = space.grid

Spaces.global_numbering(space::FourierSpace) = 1:length(space)

function Spaces.quadratures(space::FourierSpace{<:Any,1})
    x = points(space) |> first
    w = mass_matrix(space)

    ((x, w),)
end

Spaces.mass_matrix(space::FourierSpace) = space.mass_mat

Spaces.modes(space::FourierSpace) = space.freqs
#

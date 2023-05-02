#
_fft_lib(::AbstractArray) = FFTW

"""
    linspace(start, stop, len, [T])

Returns vector of equispaced range of `len` points from `start` through `stop`
of type T.
"""
function linspace(start::Number, stop::Number, len::Integer, T = nothing)

    T = isnothing(T) ? promote_type(eltype.(start, stop)...,) : T
    z = Base._linspace(T(start), T(stop), len)

    z |> Array
end

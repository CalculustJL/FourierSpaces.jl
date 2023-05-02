#
if isdefined(Base, :get_extension)
    using CUDA
    using FourierSpaces
else
    using ..CUDA
    using ..FourierSpaces
end

_fft_lib(::CUDA.CuArray) = CUDA.CUFFT
# overloads here

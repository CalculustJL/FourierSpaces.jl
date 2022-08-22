#
_fft_lib(u::AbstractArray) = FFTW
_fft_lib(u::CUDA.CuArray) = CUDA.CUFFT

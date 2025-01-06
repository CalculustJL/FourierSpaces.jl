module FourierSpaces

using Reexport

@reexport using CalculustCore
using CalculustCore.Spaces: AbstractSpace, AbstractDiscretization, TransformedSpace
using CalculustCore.Domains: AbstractDomain, ProductDomain

using SciMLOperators: AbstractSciMLOperator, InvertibleOperator, InvertedOperator

using AbstractFFTs
using FFTW # TODO - rm FFTW and have user manually load backend
using LinearAlgebra
using SparseArrays

using MLDataDevices
using Adapt: Adapt, adapt_structure
using GPUArraysCore: AbstractGPUArray, @allowscalar

include("utils.jl")
include("type.jl")
include("converters.jl")
include("interface.jl")
include("transform.jl")

const TransformedFourierSpace{T, D} = TransformedSpace{T, D,<:FourierSpace} where{T, D}

include("trans_operators.jl")
include("phys_operators.jl")

export FourierSpace

function __init__()
    # https://github.com/JuliaMath/AbstractFFTs.jl/issues/111
    # if FFT_BACKEND_NOT_LOADED
    #     @debug """Please load an FFT backend such as FFTW.jl, or CUDA """
    # end
end

end # module

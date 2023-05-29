module FourierSpaces

using Reexport
@reexport using CalculustCore
using SciMLOperators: AbstractSciMLOperator, InvertibleOperator, InvertedOperator

using FFTW
using LinearAlgebra
using SparseArrays

using Adapt: Adapt, adapt_structure
using GPUArraysCore

include("utils.jl")
include("type.jl")
include("converters.jl")
include("interface.jl")
include("transform.jl")
include("trans_operators.jl")
include("phys_operators.jl")

export FourierSpace

@static if !isdefined(Base, :get_extension)
    import Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        Requires.@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
            include("../ext/FourierSpacesCUDAExt.jl")
        end
    end
end

end # module

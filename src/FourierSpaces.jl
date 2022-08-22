module FourierSpaces

using Reexport
@reexport using AbstractPDEInterfaces

using LinearAlgebra

using FFTW
using CUDA
using Adapt

include("utils.jl")
include("type.jl")
include("converters.jl")
include("interface.jl")
include("transform.jl")
include("trans_operators.jl")
include("phys_operators.jl")

export FourierSpace

end # module

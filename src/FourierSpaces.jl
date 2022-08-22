module FourierSpaces

using FFTW
using CUDA
import Adapt

using Reexport
@reexport using AbstractPDEInterfaces

include("utils.jl")
include("type.jl")
include("converters.jl")
include("interface.jl")
include("transform.jl")
include("trans_operators.jl")
include("phys_operators.jl")

export FourierSpace

end # module

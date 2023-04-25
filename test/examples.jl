#
@testset "1D tests" begin

    @time @safetestset "Advection" begin include("../examples/fourier1d/advect.jl") end
    @time @safetestset "Advection trans" begin include("../examples/fourier1d/advect_trans.jl") end
    @time @safetestset "Heat" begin include("../examples/fourier1d/heat.jl") end
    @time @safetestset "Heat forcing" begin include("../examples/fourier1d/heat_forcing.jl") end
    @time @safetestset "Heat trans" begin include("../examples/fourier1d/heat_trans.jl") end
end

@testset "2D tests" begin
    @time @safetestset "Advection" begin include("../examples/fourier2d/advect.jl") end
    @time @safetestset "Heat" begin include("../examples/fourier2d/heat.jl") end
    @time @safetestset "Heat forcing" begin include("../examples/fourier2d/heat_forcing.jl") end
    @time @safetestset "Heat trans" begin include("../examples/fourier2d/heat_trans.jl")
    end
end
#

#
@testset "1D tests" begin

    @time @safetestset "Advection" begin include("../examples/fourier1d/advect.jl") end
    @time @safetestset "Advection trans" begin include("../examples/fourier1d/advect_trans.jl") end
    @time @safetestset "Heat" begin include("../examples/fourier1d/heat.jl") end
    @time @safetestset "Heat forcing" begin include("../examples/fourier1d/heat_forcing.jl") end
    @time @safetestset "Heat trans" begin include("../examples/fourier1d/heat_trans.jl") end

    ## https://github.com/YingboMa/SafeTestsets.jl/issues/4
    #dir = "../examples/fourier1d"
    #files = (
    #         "advect.jl",
    #         "heat.jl",
    #         "heat_forcing.jl",
    #        )
    #for file in files
    #    path = joinpath(dir, file) |> Symbol
    #    @eval begin
    #        @time @safetestset "$path" begin include($path) end
    #    end
    #end
end

@testset "2D tests" begin
    @time @safetestset "Advection" begin include("../examples/fourier2d/advect.jl") end
    @time @safetestset "Heat" begin include("../examples/fourier2d/heat.jl") end
    @time @safetestset "Heat forcing" begin include("../examples/fourier2d/heat_forcing.jl") end
    @time @safetestset "Heat trans" begin include("../examples/fourier2d/heat_trans.jl")
    end

end
#

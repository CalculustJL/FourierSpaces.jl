#
@testset "1D tests" begin

    # Advection equation
    @time @safetestset "Advection" begin include("../examples/fourier1d/advect.jl") end
    @time @safetestset "Advection trans" begin include("../examples/fourier1d/advect_trans.jl") end
    
    # # Diffusion (heat) equation
    @time @safetestset "Diffusion" begin include("../examples/fourier1d/heat.jl") end
    @time @safetestset "Diffusion forcing" begin include("../examples/fourier1d/heat_forcing.jl") end
    @time @safetestset "Diffusion trans" begin include("../examples/fourier1d/heat_trans.jl") end
    @time @safetestset "Diffusion forcing OOP" begin include("../examples/fourier1d/heat_forcing_oop.jl") end
    
    # Advection Diffusion equation
    @time @safetestset "Advection Diffusion" begin include("../examples/fourier1d/advection_diffusion.jl") end
    @time @safetestset "Advection Diffusion trans" begin include("../examples/fourier1d/advection_diffusion_trans.jl") end
    
    # Burgers equation
    @time @safetestset "Burgers" begin include("../examples/fourier1d/burgers.jl") end
    @time @safetestset "Burgers Trans" begin include("../examples/fourier1d/burgers_trans.jl") end
    @time @safetestset "Burgers Batched" begin include("../examples/fourier1d/burgers_batched.jl") end
    
    # Kuramoto-Sivashinsky equation
    @time @safetestset "Kuramoto-Sivashinsky" begin include("../examples/fourier1d/ks.jl") end

    # TODO - steady state problem

    # TODO - Cahn-Hilliard equation

    # TODO - Allen Cahn equation
end

@testset "2D tests" begin

    # Advection equation
    @time @safetestset "Advection" begin include("../examples/fourier2d/advect.jl") end
    
    # Diffusion (heat) equation
    @time @safetestset "Heat" begin include("../examples/fourier2d/heat.jl") end
    @time @safetestset "Heat forcing" begin include("../examples/fourier2d/heat_forcing.jl") end
    @time @safetestset "Heat trans" begin include("../examples/fourier2d/heat_trans.jl") end
end
#

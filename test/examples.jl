#
dir = "../examples"
files = (
         "fourier1d/advect.jl",
         "fourier1d/heat.jl",
         "fourier1d/heat_forcing.jl",
        )
for file in files
    @time @safetestset "$file" begin include("$dir/$file") end
end
#

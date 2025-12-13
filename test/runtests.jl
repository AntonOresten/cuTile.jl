using Test
import cuTile as ct

@testset "cuTile" verbose=true begin

include("types.jl")
include("codegen.jl")
include("execution.jl")
include("examples.jl")

end

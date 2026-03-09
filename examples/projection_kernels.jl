using CUDA

include("projection_kernels_impl.jl")

const pk = ProjectionKernelsExample

prepare(args...; kwargs...) = pk.prepare(args...; kwargs...)
run(args...; kwargs...) = pk.run(args...; kwargs...)
run_others(args...; kwargs...) = pk.run_others(args...; kwargs...)
verify(args...; kwargs...) = pk.verify(args...; kwargs...)

pk.main()

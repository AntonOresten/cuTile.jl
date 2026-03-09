using CUDA

include("sparse_flashmhffn_impl.jl")

const sf = SparseFlashMHFExample

prepare(args...; kwargs...) = sf.prepare(args...; kwargs...)
run(args...; kwargs...) = sf.run(args...; kwargs...)
verify(args...; kwargs...) = sf.verify(args...; kwargs...)

sf.main()

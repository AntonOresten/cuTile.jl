public Tiled

"""
    Tiled(x)

Wrapper for CUDA arrays to allow dispatching to cuTile kernels.
"""
struct Tiled{A <: AbstractArray}
    parent::A
end
Tiled(x) = x  # passthrough for non-arrays (Numbers, etc.)
Base.parent(t::Tiled) = t.parent
Base.axes(t::Tiled) = axes(parent(t))
Base.size(t::Tiled) = size(parent(t))
Base.ndims(::Tiled{A}) where A = ndims(A)
Base.eltype(::Tiled{A}) where A = eltype(A)
Base.Broadcast.broadcastable(t::Tiled) = t

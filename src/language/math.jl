# Math operations

public rsqrt, exp2, truediv


## scalar math

# unary
rsqrt(x::T) where {T <: AbstractFloat} = Intrinsics.rsqrt(x)
for fn in (:ceil, :floor, :exp, :exp2, :log, :log2, :sqrt,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @overlay Base.$fn(x::T) where {T <: ScalarFloat} = Intrinsics.$fn(x)
end

@overlay Base.abs(x::T) where {T <: ScalarFloat} = Intrinsics.absf(x)
@overlay Base.abs(x::T) where {T <: Signed} = Intrinsics.absi(x)

@overlay Base.rem(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.remf(x, y)

@overlay Base.fma(x::T, y::T, z::T) where {T <: ScalarFloat} = Intrinsics.fma(x, y, z)

# max/min
@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, Signedness.Signed)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, Signedness.Unsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, Signedness.Signed)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, Signedness.Unsigned)
@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y, false)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y, false)


## tile math

# All tile broadcasting (unary, binary, ternary) is handled by the generic
# Broadcast.copy → map path: scalar @overlay methods provide the element-wise
# implementation, and map handles broadcasting + to_scalar/from_scalar.
# No explicit broadcasted methods needed here.

# exp2 with flush_to_zero — the scalar overlay doesn't support ftz, so provide
# a direct tile-level method that calls the intrinsic.
"""
    exp2(tile::Tile; flush_to_zero=false) -> Tile

Compute 2^x element-wise with optional flush-to-zero semantics.
"""
@inline exp2(tile::Tile{T}; flush_to_zero::Bool=false) where {T <: AbstractFloat} =
    Intrinsics.exp2(tile, flush_to_zero)

"""
    truediv(a, b; flush_to_zero=false, rounding_mode=nothing) -> Tile

Element-wise floating-point division with optional flush-to-zero and rounding
mode control. Supports broadcasting between tiles and scalars.

Equivalent to Python cuTile's `ct.truediv(a, b, ...)`.

# Example
```julia
result = ct.truediv(acc, normalizer; flush_to_zero=true, rounding_mode=ct.Rounding.Approx)
```
"""
@inline function truediv(a::Tile{T,S}, b::Tile{T,S};
                         flush_to_zero::Bool=false, rounding_mode=nothing) where {T <: AbstractFloat, S}
    Intrinsics.divf(a, b, rounding_mode, flush_to_zero)
end
# Broadcasting variants: auto-broadcast mismatched shapes before calling divf
@inline function truediv(a::Tile{T}, b::Tile{T};
                         flush_to_zero::Bool=false, rounding_mode=nothing) where {T <: AbstractFloat}
    S = Broadcast.broadcast_shape(size(a), size(b))
    a_bc = broadcast_to(a, S)
    b_bc = broadcast_to(b, S)
    Intrinsics.divf(a_bc, b_bc, rounding_mode, flush_to_zero)
end

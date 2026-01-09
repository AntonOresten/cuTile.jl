# Math operations

public rsqrt


## scalar math

# unary
rsqrt(x::T) where {T <: AbstractFloat} = Intrinsics.rsqrt(x)
for fn in (:abs, :ceil, :floor, :exp, :exp2, :log, :log2, :sqrt,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @overlay Base.$fn(x::T) where {T <: ScalarFloat} = Intrinsics.$fn(x)
end

@overlay Base.fma(x::T, y::T, z::T) where {T <: ScalarFloat} = Intrinsics.fma(x, y, z)

# max/min
@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, SignednessSigned)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, SignednessUnsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, SignednessSigned)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, SignednessUnsigned)
@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y)


## tile math

# element-wise unary
for fn in (:exp, :exp2, :log, :log2, :sqrt, :rsqrt, :ceil, :floor,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($fn), a::Tile{T}) where {T<:AbstractFloat} =
        Intrinsics.$fn(a)
end
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(abs), a::Tile{T}) where {T<:AbstractFloat} =
    Intrinsics.absf(a)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(abs), a::Tile{T}) where {T<:Integer} =
    Intrinsics.absi(a)

# element-wise binary
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.remf(broadcast_to(a, S), broadcast_to(b, S))
end

# element-wise ternary
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(fma), a::Tile{T,S}, b::Tile{T,S}, c::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.fma(a, b, c)


## mixed math

# Float remainder (rem.(tile, scalar), rem.(scalar, tile))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
    Intrinsics.remf(a, broadcast_to(Tile(T(b)), S))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.remf(broadcast_to(Tile(T(a)), S), b)

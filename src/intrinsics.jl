#=============================================================================
 Tile Arithmetic
=============================================================================#

# These are stub implementations that the compiler intercepts.
# They return a new Tile with the same shape, enabling proper type inference.

@noinline function tile_add(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_sub(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

@noinline function tile_mul(a::Tile{T, S}, b::Tile{T, S})::Tile{T, S} where {T, S}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

# Operator overloads dispatch to the intrinsic functions
Base.:(+)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_add(a, b)
Base.:(-)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_sub(a, b)
Base.:(*)(a::Tile{T, S}, b::Tile{T, S}) where {T, S} = tile_mul(a, b)

#=============================================================================
 Tile Shape Operations
=============================================================================#

public transpose

"""
    transpose(tile::Tile{T, (M, N)}) -> Tile{T, (N, M)}

Transpose a 2D tile, swapping its dimensions.
"""
@noinline function transpose(tile::Tile{T, S})::Tile{T, reverse(S)} where {T, S}
    Tile{T, reverse(S)}()
end

#=============================================================================
 GPU Intrinsics (stub implementations for host-side)
=============================================================================#

public bid, num_blocks, load, store

# Note: These stubs are markers for the compiler to recognize.
# The actual implementations are handled by the compiler which emits Tile IR ops.
#
# We use Base.compilerbarrier(:const, ...) to prevent constant folding while
# allowing other optimizations. This lets us use optimized IR.
# If these reach runtime (not compiled), they error with a clear message.

"""
    bid(axis) -> Int32

Get the block ID along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetTileBlockIdOp.
"""
@noinline bid(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    num_blocks(axis) -> Int32

Get the grid size along the given axis (0=x, 1=y, 2=z).
In kernel code, this is compiled to GetNumTileBlocksOp.
"""
@noinline num_blocks(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    load(ptr, index, shape::NTuple{N, Int}) -> Tile{T, shape}

Load a tile from a pointer at the given index with the specified shape.
In kernel code, this is compiled to LoadViewTkoOp.

Returns a `Tile{T, Shape}` where T is the pointer element type and Shape
is the compile-time constant shape tuple.
"""
# Internal function with shape as type parameter for proper type inference
@noinline function _load(ptr::Ptr{T}, index, ::Val{shape}) where {T, shape}
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape as type parameter
@inline load(ptr::Ptr{T}, index, shape::NTuple{N, Int}) where {T, N} = _load(ptr, index, Val(shape))

"""
    store(ptr, index, tile::Tile) -> Nothing

Store a tile to a pointer at the given index.
In kernel code, this is compiled to StoreViewTkoOp.
"""
@noinline function store(ptr::Ptr{T}, index, tile::Tile{T})::Nothing where T
    # donotdelete prevents the optimizer from eliminating this call
    # even though it has no observable effects in Julia
    Base.donotdelete(ptr, index, tile)
    nothing
end

# TileArray overloads - these are intercepted by the compiler
# The compiler extracts ptr/sizes/strides from the destructured TileArray

"""
    load(arr::TileArray, index, shape) -> Tile

Load a tile from a TileArray at the given index with the specified shape.
The TileArray's sizes and strides are used to construct the TensorView.
"""
# Internal function with shape as type parameter for proper type inference
@noinline function _load(arr::TileArray{T, N}, index, ::Val{shape}) where {T, N, shape}
    Base.donotdelete(arr, index)
    Tile{T, shape}()
end
# Public API - inline wrapper that captures shape as type parameter
@inline load(arr::TileArray{T, N}, index, shape::NTuple{M, Int}) where {T, N, M} = _load(arr, index, Val(shape))

# Load with Constant shape tuple (1D) - extracts value from Constant type parameter
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V}}) where {T, N, V}
    _load(arr, index, Val((V,)))
end

# Load with Constant shape tuple (2D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}}) where {T, N, V1, V2}
    _load(arr, index, Val((V1, V2)))
end

# Load with Constant shape tuple (3D)
@inline function load(arr::TileArray{T, N}, index, shape::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}}) where {T, N, V1, V2, V3}
    _load(arr, index, Val((V1, V2, V3)))
end

# Keyword argument version for ct.load(arr; index=..., shape=...)
@inline function load(arr::TileArray{T, N}; index, shape) where {T, N}
    shape_val = _extract_shape(shape)
    _load(arr, index, Val(shape_val))
end

# Helper to extract compile-time shape from various tuple types
@inline _extract_shape(s::NTuple{N, Int}) where N = s
@inline _extract_shape(s::Tuple{Constant{Int, V}}) where V = (V,)
@inline _extract_shape(s::Tuple{Constant{Int, V1}, Constant{Int, V2}}) where {V1, V2} = (V1, V2)
@inline _extract_shape(s::Tuple{Constant{Int, V1}, Constant{Int, V2}, Constant{Int, V3}}) where {V1, V2, V3} = (V1, V2, V3)

"""
    store(arr::TileArray, index, tile::Tile) -> Nothing

Store a tile to a TileArray at the given index.
"""
@noinline function store(arr::TileArray{T, N}, index, tile::Tile{T})::Nothing where {T, N}
    Base.donotdelete(arr, index, tile)
    nothing
end

# Keyword argument version for ct.store(arr; index=..., tile=...)
@noinline function store(arr::TileArray{T, N}; index, tile::Tile{T})::Nothing where {T, N}
    Base.donotdelete(arr, index, tile)
    nothing
end

#=============================================================================
 Matrix Multiply-Accumulate
=============================================================================#

public mma

"""
    mma(a::Tile{T1, (M, K)}, b::Tile{T2, (K, N)}, acc::Tile{T3, (M, N)}) -> Tile{T3, (M, N)}

Perform matrix-multiply-accumulate: result = a @ b + acc.
Uses tensor cores when available.

The input tiles must have compatible shapes:
- a: (M, K)
- b: (K, N)
- acc: (M, N)
- result: (M, N)
"""
@noinline function mma(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC}
    Base.donotdelete(a, b, acc)
    Tile{T3, SC}()
end

#=============================================================================
 Tile Construction
=============================================================================#

public full, astype

"""
    full(shape::NTuple{N, Int}, value, dtype::Type{T}) -> Tile{T, shape}

Create a tile filled with a constant value.

# Example
```julia
zeros_tile = ct.full((32, 32), 0, Float32)  # 32x32 tile of zeros
```
"""
@noinline function full(shape::NTuple{N, Int}, value, ::Type{T})::Tile{T, shape} where {N, T}
    Base.donotdelete(value)  # shape and T are type parameters, can't be deleted
    Tile{T, shape}()
end

"""
    convert(Tile{T2}, tile::Tile{T1, Shape}) -> Tile{T2, Shape}
    astype(tile::Tile{T1, Shape}, ::Type{T2}) -> Tile{T2, Shape}

Convert a tile's element type from T1 to T2.

# Example
```julia
acc = ct.full((64, 64), 0.0f0, Float32)
result = convert(ct.Tile{ct.TFloat32}, acc)  # Convert to TF32 for tensor cores
result = convert(ct.Tile{Float16}, acc)      # Convert to Float16
```
"""
@noinline function astype(tile::Tile{T1, Shape}, ::Type{T2})::Tile{T2, Shape} where {T1, Shape, T2}
    Base.donotdelete(tile)
    Tile{T2, Shape}()
end

# Julia-style convert syntax builds on astype
Base.convert(::Type{Tile{T2}}, tile::Tile{T1, Shape}) where {T1, T2, Shape} = astype(tile, T2)

#=============================================================================
 Array Dimension Operations
=============================================================================#

public num_tiles

"""
    num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int}) -> Int32

Get the number of tiles along a specific axis of an array, given the tile shape.
This is equivalent to cdiv(arr.sizes[axis+1], shape[axis+1]).

# Arguments
- `arr`: The array to query
- `axis`: The axis (0-indexed) to count tiles along
- `shape`: The tile shape used for partitioning

# Example
```julia
# For a 1024x768 matrix with 32x32 tiles:
# num_tiles(arr, 0, (32, 32)) returns cdiv(1024, 32) = 32
# num_tiles(arr, 1, (32, 32)) returns cdiv(768, 32) = 24
```
"""
@noinline function num_tiles(arr::TileArray{T, N}, axis::Integer, shape::NTuple{M, Int})::Int32 where {T, N, M}
    Base.inferencebarrier(zero(Int32))
end

#=============================================================================
 Integer Arithmetic Operations
=============================================================================#

public cdiv, floordiv

"""
    cdiv(a::Integer, b::Integer) -> Int32

Ceiling division: ⌈a/b⌉ = (a + b - 1) ÷ b

This is useful for computing grid dimensions from array sizes and tile sizes.
"""
@noinline cdiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    floordiv(a::Integer, b::Integer) -> Int32

Floor division: ⌊a/b⌋

This is equivalent to `a ÷ b` but provided for consistency with the cuTile API.
"""
@noinline floordiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    rem(a::Integer, b::Integer) -> Int32

Remainder operation: a % b (C-style, result has same sign as dividend)
"""
@noinline Base.rem(a::Int32, b::Int32)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    min(a::Integer, b::Integer) -> Int32

Minimum of two integers.
"""
@noinline Base.min(a::Int32, b::Int32)::Int32 = Base.inferencebarrier(zero(Int32))

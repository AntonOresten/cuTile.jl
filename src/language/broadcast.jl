# Broadcasting Infrastructure for Tiles
#
# Defines the broadcast style and shape computation for Tile types.
# Actual broadcasted operations are defined in arithmetic.jl and math.jl.

import Base.Broadcast: BroadcastStyle, Broadcasted, broadcastable


#=============================================================================
 Broadcast Shape Computation
=============================================================================#

"""
    broadcast_shape(s1::Tuple, s2::Tuple) -> Tuple

Compute the broadcast shape from two tile shapes using NumPy-style broadcasting rules.
- Shapes are compared from right to left (trailing dimensions)
- Dimensions are compatible if they're equal or one of them is 1
- Missing dimensions are treated as 1

This is a pure function that Julia's const-prop can evaluate at compile time.

# Examples
```julia
broadcast_shape((128,), (1, 128))   # => (1, 128)
broadcast_shape((1,), (128,))       # => (128,)
broadcast_shape((4, 1), (1, 8))     # => (4, 8)
broadcast_shape((16, 32), (16, 32)) # => (16, 32)
```
"""
@inline function broadcast_shape(s1::Tuple, s2::Tuple)
    max_ndim = max(length(s1), length(s2))
    ntuple(max_ndim) do i
        # Index from the right (trailing dimensions)
        idx1 = length(s1) - max_ndim + i
        idx2 = length(s2) - max_ndim + i
        d1 = idx1 > 0 ? s1[idx1] : 1
        d2 = idx2 > 0 ? s2[idx2] : 1
        # Check compatibility
        (d1 == d2 || d1 == 1 || d2 == 1) || error("Shapes $s1 and $s2 are not broadcastable")
        max(d1, d2)
    end
end


#=============================================================================
 Custom BroadcastStyle for Tiles
=============================================================================#

struct TileStyle <: BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:Tile}) = TileStyle()

# When combining TileStyle with itself, return TileStyle
Base.Broadcast.BroadcastStyle(::TileStyle, ::TileStyle) = TileStyle()

# When combining TileStyle with scalars, TileStyle wins
Base.Broadcast.BroadcastStyle(::TileStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = TileStyle()

# Tiles are already broadcastable - return as-is
Base.Broadcast.broadcastable(t::Tile) = t

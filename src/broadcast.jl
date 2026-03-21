import Base.Broadcast: BroadcastStyle, Broadcasted

#=============================================================================
 Tiled wrapper — routes broadcast expressions through cuTile kernels
=============================================================================#

"""
    Tiled(x)

Wrapper that routes broadcast expressions through cuTile kernels.

    Tiled(B) .= A .+ A

Uses Julia's `Base.Broadcast` fusion machinery to build a `Broadcasted` tree,
then dispatches to a generic cuTile kernel that evaluates the tree on tiles.
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

# Walk dotted AST, wrap value-position leaves in Tiled()
_wrap_tiled(x) = x  # literals pass through
_wrap_tiled(s::Symbol) = :($Tiled($s))
function _wrap_tiled(ex::Expr)
    if ex.head === :.=
        Expr(:.=, _wrap_tiled(ex.args[1]), _wrap_tiled(ex.args[2]))
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        # f.(args...) — wrap args, NOT function position
        new_args = map(_wrap_tiled, ex.args[2].args)
        Expr(:., ex.args[1], Expr(:tuple, new_args...))
    else
        Expr(ex.head, map(_wrap_tiled, ex.args)...)
    end
end

"""
    @. expr

Like `Base.@.` but wraps every value-position leaf in `Tiled()`, routing
the broadcast through cuTile kernels.

    using cuTile; const ct = cuTile
    ct.@. C = A + sin(B)
    # equivalent to: Tiled(C) .= Tiled(A) .+ sin.(Tiled(B))
"""
macro __dot__(ex)
    esc(_wrap_tiled(Base.Broadcast.__dot__(ex)))
end

#=============================================================================
 TiledStyle — routes broadcast through cuTile kernels
=============================================================================#

struct TiledStyle{N} <: BroadcastStyle end
TiledStyle{M}(::Val{N}) where {N,M} = TiledStyle{N}()

BroadcastStyle(::Type{<:Tiled{A}}) where A = TiledStyle{ndims(A)}()

# TiledStyle wins over DefaultArrayStyle
BroadcastStyle(::TiledStyle{N}, ::Base.Broadcast.DefaultArrayStyle{M}) where {N,M} = TiledStyle{max(N,M)}()
BroadcastStyle(::TiledStyle{N}, ::TiledStyle{M}) where {N,M} = TiledStyle{max(N,M)}()

#=============================================================================
 materialize! and copy — dispatch to _tiled_broadcast!
=============================================================================#

function Base.Broadcast.materialize!(dest::Tiled, bc::Broadcasted)
    _tiled_broadcast!(parent(dest), bc)
    return dest
end

function Base.copy(bc::Broadcasted{TiledStyle{N}}) where N
    arr = @something _find_tiled_array(bc) error("tiled broadcast requires at least one Tiled() argument")
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    dest = similar(arr, ElType, axes(bc))
    _tiled_broadcast!(dest, bc)
    return dest
end

"""Find the first underlying array from a Tiled leaf in a Broadcasted tree."""
_find_tiled_array(t::Tiled) = parent(t)
_find_tiled_array(x) = nothing
function _find_tiled_array(bc::Broadcasted)
    for arg in bc.args
        arr = _find_tiled_array(arg)
        arr !== nothing && return arr
    end
    return nothing
end

#=============================================================================
 _tiled_broadcast! — generic AbstractArray implementation
=============================================================================#

function _tiled_broadcast!(dest::AbstractArray{T,N}, bc::Broadcasted) where {T, N}
    dest_ta = TileArray(dest)
    tiled_bc = _to_tiled_bc(bc)

    ts = _compute_tile_sizes(size(dest))
    grid = ntuple(i -> cld(size(dest, i), ts[i]), N)

    launch_grid = N <= 3 ? grid : (grid[1], grid[2], prod(grid[i] for i in 3:N))
    overflow = N > 3 ? grid[3:end] : ()

    launch(_tiled_bc_kernel, launch_grid, dest_ta, tiled_bc,
           Constant(ts), Constant(overflow))
end

#=============================================================================
 Generic tree walk — convert leaves to TileArrays
=============================================================================#

_to_tiled_bc(t::Tiled) = TileArray(parent(t))
_to_tiled_bc(arr::AbstractArray) = TileArray(arr)
_to_tiled_bc(x::Number) = x
_to_tiled_bc(x) = x  # fallback for other types
function _to_tiled_bc(bc::Broadcasted)
    new_args = map(_to_tiled_bc, bc.args)
    Broadcasted{Nothing}(bc.f, new_args, nothing)
end

#=============================================================================
 Broadcast kernel — evaluates Broadcasted tree on tiles
=============================================================================#

@generated function _tiled_bc_kernel(dest::TileArray{T, N}, bc, tile_size, overflow_grids) where {T, N}
    body = Expr[]
    bid_vars = [Symbol("bid_$d") for d in 1:N]

    if N <= 3
        for d in 1:N
            push!(body, :($(bid_vars[d]) = cuTile.bid($d)))
        end
    else
        push!(body, :($(bid_vars[1]) = cuTile.bid(1)))
        push!(body, :($(bid_vars[2]) = cuTile.bid(2)))
        push!(body, :(_rem = cuTile.bid(3) - Int32(1)))
        for d in 3:N
            if d < N
                push!(body, :($(bid_vars[d]) = rem(_rem, Int32(overflow_grids[$(d-2)])) + Int32(1)))
                push!(body, :(_rem = fld(_rem, Int32(overflow_grids[$(d-2)]))))
            else
                push!(body, :($(bid_vars[d]) = _rem + Int32(1)))
            end
        end
    end

    idx = N == 1 ? bid_vars[1] : Expr(:tuple, bid_vars...)
    push!(body, :(result = _eval_bc(bc, $idx, tile_size)))
    push!(body, :(result_converted = convert(cuTile.Tile{$T}, result)))
    push!(body, :(cuTile.store(dest, $idx, result_converted)))
    push!(body, :(return))
    Expr(:block, body...)
end

#=============================================================================
 Recursive tree evaluation inside kernel
=============================================================================#

@inline _eval_bc(arr::TileArray, bid, tile_size) = cuTile.load(arr, bid, tile_size)
@inline _eval_bc(x::Number, bid, tile_size) = x

@inline function _eval_bc(bc::Broadcasted, bid, tile_size)
    args = _eval_bc_args(bc.args, bid, tile_size)
    # Use broadcast to get element-wise semantics (not direct call, which
    # would dispatch to e.g. matmul for * on tiles)
    broadcast(bc.f, args...)
end

@inline _eval_bc_args(::Tuple{}, bid, tile_size) = ()
@inline _eval_bc_args(args::Tuple, bid, tile_size) =
    (_eval_bc(args[1], bid, tile_size), _eval_bc_args(Base.tail(args), bid, tile_size)...)

#=============================================================================
 Tile sizing
=============================================================================#

"""
    _compute_tile_sizes(dest_size; budget=4096)

Distribute a total element budget greedily across dimensions, skipping singletons.
Each tile dimension is a power of 2, capped by the array size in that dimension.
"""
function _compute_tile_sizes(dest_size::NTuple{N,Int}; budget::Int=4096) where N
    ts = ones(Int, N)
    remaining = budget
    for i in 1:N
        s = dest_size[i]
        s == 1 && continue
        t = prevpow(2, min(remaining, s))
        ts[i] = t
        remaining = remaining ÷ t
        remaining < 2 && break
    end
    return NTuple{N,Int}(ts)
end

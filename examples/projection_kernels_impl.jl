# EXCLUDE FROM TESTING

module ProjectionKernelsExample

using CUDA
using LinearAlgebra
using Random
import cuTile as ct

export ProjectionDispatch,
       SlotProjectionDispatch,
       DenseFixedHeadBaseline,
       PackedFixedHeadBaseline,
       build_projection_dispatch,
       make_dense_fixed_head_baseline,
       make_packed_fixed_head_baseline,
       dense_fixed_heads_in_proj!,
       dense_fixed_heads_out_proj!,
       dense_fixed_heads_fwd!,
       launch_in_proj_fwd!,
       launch_in_proj_fwd_contiguous!,
       launch_out_proj_fwd!,
       launch_in_proj_bwd_dx!,
       launch_out_proj_bwd_dy!,
       launch_in_proj_bwd_dw!,
       launch_out_proj_bwd_dw!,
       packed_in_proj_gemm!,
       packed_in_proj_unpack!,
       packed_in_proj_practical!,
       packed_out_proj_gemm!,
       packed_out_proj_from_sparse!,
       packed_total_gemm!,
       packed_total_practical!,
       reference_in_proj_fwd,
       reference_out_proj_fwd,
       reference_in_proj_bwd_dx,
       reference_out_proj_bwd_dy,
       reference_in_proj_bwd_dw,
       reference_out_proj_bwd_dw,
       make_example_problem,
       prepare,
       run,
       run_others,
       verify,
       run_example,
       main

const GROUP_SIZE_M = 8

Base.@kwdef struct ProjectionDispatch
    sorted_ids::CuArray{Int32, 1}
    tok_cols::CuArray{Int32, 1}
    head_block_starts::CuArray{Int32, 1}
    block_heads::CuArray{Int32, 1}
    token_heads::CuArray{Int32, 2}
    slot_dispatches::Vector{Any}
    padded_M::Int32
    tile_m::Int32
    k::Int32
    H::Int32
end

Base.@kwdef struct SlotProjectionDispatch
    sorted_tok_ids::CuArray{Int32, 1}
    sorted_flat_ids::CuArray{Int32, 1}
    head_block_starts::CuArray{Int32, 1}
    block_heads::CuArray{Int32, 1}
    padded_L::Int32
    tile_l::Int32
    slot::Int32
end

Base.@kwdef struct DenseFixedHeadBaseline{T}
    X::CuArray{T, 2}
    W_in_slots::Vector{CuArray{T, 2, CUDA.DeviceMemory}}
    W_out_slots::Vector{CuArray{T, 2, CUDA.DeviceMemory}}
    Y_slots::Vector{CuArray{T, 2, CUDA.DeviceMemory}}
    Z::CuArray{T, 2}
end

Base.@kwdef struct PackedFixedHeadBaseline{T}
    X::CuArray{T, 2}
    W_in_cat::CuArray{T, 2}
    W_out_cat::CuArray{T, 2}
    Y_cat::CuArray{T, 2}
    Y_sparse::CuArray{T, 2}
    Z::CuArray{T, 2}
end

@inline function swizzle_2d(M, N, tm, tn, group_size_m, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(group_size_m) * num_bid_n
    group_id = fld(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(group_size_m)
    group_span_m = min(num_bid_m - first_bid_m, Int32(group_size_m))
    bid_m = first_bid_m + rem(bid, group_span_m)
    bid_n = fld(rem(bid, num_bid_in_group), group_span_m)
    return bid_m, bid_n
end

@inline block_count(dispatch::ProjectionDispatch) = dispatch.padded_M ÷ dispatch.tile_m
@inline block_count(dispatch::SlotProjectionDispatch) = dispatch.padded_L ÷ dispatch.tile_l

@inline function tile_range(tile_idx::Int32, tile_size::Int)
    base = (tile_idx - Int32(1)) * Int32(tile_size) + Int32(1)
    base .+ ct.arange((tile_size,), Int32) .- Int32(1)
end

@inline reshape_col(idx::ct.Tile) = reshape(idx, (1, size(idx, 1)))
@inline reshape_row(idx::ct.Tile) = reshape(idx, (size(idx, 1), 1))

function _routing_matrix(token_heads::AbstractMatrix{<:Integer}, L::Integer)
    @assert size(token_heads, 2) == L
    return Int32.(token_heads)
end

function _routing_matrix(k_to_head::AbstractVector{<:Integer}, L::Integer)
    return repeat(reshape(Int32.(k_to_head), :, 1), 1, L)
end

function _global_slot_heads(token_heads::AbstractMatrix{<:Integer})
    slot_heads = Int32.(token_heads[:, 1])
    @assert all(token_heads[:, tok] == slot_heads for tok in axes(token_heads, 2)) "packed baseline requires fixed routing"
    return slot_heads
end

@inline function _is_fixed_routing(token_heads::AbstractMatrix{<:Integer})
    slot_heads = token_heads[:, 1]
    return all(token_heads[:, tok] == slot_heads for tok in axes(token_heads, 2))
end

function build_slot_dispatches(token_heads::AbstractMatrix{<:Integer}, tile_l::Integer)
    token_heads_i32 = Int32.(token_heads)
    k, L = size(token_heads_i32)
    H = maximum(token_heads_i32)
    slot_dispatches = Vector{SlotProjectionDispatch}(undef, k)

    for slot in 1:k
        per_head = [Int32[] for _ in 1:H]
        for tok in 1:L
            head = Int(token_heads_i32[slot, tok])
            push!(per_head[head], Int32(tok))
        end

        sorted_tok_ids = Int32[]
        sorted_flat_ids = Int32[]
        block_heads = Int32[]
        head_block_starts = Vector{Int32}(undef, H + 1)
        next_block = Int32(1)

        for head in 1:H
            head_block_starts[head] = next_block
            tok_ids = per_head[head]
            append!(sorted_tok_ids, tok_ids)
            append!(sorted_flat_ids, Int32.((tok_ids .- Int32(1)) .* Int32(k) .+ Int32(slot)))

            padded = cld(length(tok_ids), tile_l) * tile_l
            pad = padded - length(tok_ids)
            append!(sorted_tok_ids, zeros(Int32, pad))
            append!(sorted_flat_ids, zeros(Int32, pad))
            nblocks = Int32(cld(length(tok_ids), tile_l))
            append!(block_heads, fill(Int32(head), nblocks))
            next_block += nblocks
        end

        head_block_starts[end] = next_block
        slot_dispatches[slot] = SlotProjectionDispatch(
            sorted_tok_ids=CuArray(sorted_tok_ids),
            sorted_flat_ids=CuArray(sorted_flat_ids),
            head_block_starts=CuArray(head_block_starts),
            block_heads=CuArray(block_heads),
            padded_L=Int32(length(sorted_tok_ids)),
            tile_l=Int32(tile_l),
            slot=Int32(slot),
        )
    end

    return slot_dispatches
end

function build_projection_dispatch(token_heads::AbstractMatrix{<:Integer}, tile_m::Integer)
    k, L = size(token_heads)
    token_heads_i32 = Int32.(token_heads)
    H = maximum(token_heads_i32)
    per_head = [Int32[] for _ in 1:H]

    for tok in 1:L
        for slot in 1:k
            head = Int(token_heads_i32[slot, tok])
            flat_id = Int32((tok - 1) * k + slot)
            push!(per_head[head], flat_id)
        end
    end

    sorted_ids = Int32[]
    tok_cols = Int32[]
    block_heads = Int32[]
    head_block_starts = Vector{Int32}(undef, H + 1)
    next_block = Int32(1)

    for head in 1:H
        head_block_starts[head] = next_block
        ids = per_head[head]
        append!(sorted_ids, ids)
        append!(tok_cols, Int32.(fld.(ids .- Int32(1), Int32(k)) .+ Int32(1)))

        padded = cld(length(ids), tile_m) * tile_m
        pad = padded - length(ids)
        append!(sorted_ids, zeros(Int32, pad))
        append!(tok_cols, zeros(Int32, pad))
        nblocks = Int32(cld(length(ids), tile_m))
        append!(block_heads, fill(Int32(head), nblocks))
        next_block += nblocks
    end

    head_block_starts[end] = next_block

    ProjectionDispatch(
        sorted_ids=CuArray(sorted_ids),
        tok_cols=CuArray(tok_cols),
        head_block_starts=CuArray(head_block_starts),
        block_heads=CuArray(block_heads),
        token_heads=CuArray(token_heads_i32),
        slot_dispatches=build_slot_dispatches(token_heads_i32, tile_m),
        padded_M=Int32(length(sorted_ids)),
        tile_m=Int32(tile_m),
        k=Int32(k),
        H=Int32(H),
    )
end

function build_projection_dispatch(k_to_head::AbstractVector{<:Integer}, L::Integer, tile_m::Integer)
    return build_projection_dispatch(_routing_matrix(k_to_head, L), tile_m)
end

@inline function split_block_bounds(head_block_starts::ct.TileArray{Int32, 1},
                                    head::Int32,
                                    split::Int32,
                                    num_splits::Int32)
    start_block = head_block_starts[head]
    stop_block = head_block_starts[head + Int32(1)]
    total_blocks = stop_block - start_block
    rel_start = fld((split - Int32(1)) * total_blocks, num_splits)
    rel_stop = fld(split * total_blocks, num_splits)
    return start_block + rel_start, start_block + rel_stop
end

function in_proj_fwd_kernel(
    W_in::ct.TileArray{T, 3},
    X::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    tok_cols::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    Y::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_E::Int,
    TILE_D::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_e_0 = swizzle_2d(padded_M, size(Y, 1), TILE_M, TILE_E, GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_e = bid_e_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    tok_tile = ct.load(tok_cols, bid_m, (TILE_M,))
    head = block_heads[bid_m]

    acc = ct.zeros((TILE_E, TILE_M), Float32)
    num_d_tiles = cld(size(X, 1), TILE_D)
    d_tile = Int32(1)
    while d_tile <= num_d_tiles
        d_idx = reshape_row(tile_range(d_tile, TILE_D))
        x_tile = ct.gather(X, (d_idx, reshape_col(tok_tile)))
        w_tile = reshape(
            ct.load(W_in, (bid_e, d_tile, head), (TILE_E, TILE_D, 1);
                    padding_mode=ct.PaddingMode.Zero),
            (TILE_E, TILE_D),
        )
        acc = muladd(w_tile, x_tile, acc)
        d_tile += Int32(1)
    end

    e_idx = reshape_row(tile_range(bid_e, TILE_E))
    ct.scatter(Y, (e_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, acc))
    return nothing
end

function in_proj_fwd_contiguous_kernel(
    W_in::ct.TileArray{T, 3},
    sorted_ids::ct.TileArray{Int32, 1},
    head_block_starts::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    X::ct.TileArray{T, 2},
    Y::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_E::Int,
    TILE_D::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_e_0 = swizzle_2d(padded_M, size(Y, 1), TILE_M, TILE_E, GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_e = bid_e_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    head = block_heads[bid_m]
    tok_block = bid_m - head_block_starts[head] + Int32(1)

    acc = ct.zeros((TILE_E, TILE_M), Float32)
    num_d_tiles = cld(size(X, 1), TILE_D)
    d_tile = Int32(1)
    while d_tile <= num_d_tiles
        x_tile = ct.load(X, (d_tile, tok_block), (TILE_D, TILE_M); padding_mode=ct.PaddingMode.Zero)
        w_tile = reshape(
            ct.load(W_in, (bid_e, d_tile, head), (TILE_E, TILE_D, 1);
                    padding_mode=ct.PaddingMode.Zero),
            (TILE_E, TILE_D),
        )
        if T === Float32
            acc = muladd(convert(ct.Tile{ct.TFloat32}, w_tile),
                         convert(ct.Tile{ct.TFloat32}, x_tile), acc)
        else
            acc = muladd(w_tile, x_tile, acc)
        end
        d_tile += Int32(1)
    end

    e_idx = reshape_row(tile_range(bid_e, TILE_E))
    ct.scatter(Y, (e_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, acc))
    return nothing
end

function out_proj_fwd_slot_kernel(
    W_out::ct.TileArray{T, 3},
    Y::ct.TileArray{T, 2},
    sorted_tok_ids::ct.TileArray{Int32, 1},
    sorted_flat_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    Z::ct.TileArray{T, 2},
    padded_L::Int32,
    TILE_L::Int,
    TILE_D::Int,
    TILE_E::Int,
) where {T}
    bid = ct.bid(1)
    bid_l_0, bid_d_0 = swizzle_2d(padded_L, size(Z, 1), TILE_L, TILE_D, GROUP_SIZE_M, bid - Int32(1))
    bid_l = bid_l_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    tok_ids = ct.load(sorted_tok_ids, bid_l, (TILE_L,))
    flat_ids = ct.load(sorted_flat_ids, bid_l, (TILE_L,))
    head = block_heads[bid_l]

    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    acc = ct.zeros((TILE_D, TILE_L), Float32)
    num_e_tiles = cld(size(Y, 1), TILE_E)
    e_tile = Int32(1)
    while e_tile <= num_e_tiles
        e_idx = reshape_row(tile_range(e_tile, TILE_E))
        y_tile = ct.gather(Y, (e_idx, reshape_col(flat_ids)))
        w_tile = reshape(
            ct.load(W_out, (bid_d, e_tile, head), (TILE_D, TILE_E, 1);
                    padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_E),
        )
        if T === Float32
            acc = muladd(convert(ct.Tile{ct.TFloat32}, w_tile),
                         convert(ct.Tile{ct.TFloat32}, y_tile), acc)
        else
            acc = muladd(w_tile, y_tile, acc)
        end
        e_tile += Int32(1)
    end

    z_prev = ct.gather(Z, (d_idx, reshape_col(tok_ids)))
    ct.scatter(Z, (d_idx, reshape_col(tok_ids)), z_prev + convert(ct.Tile{T}, acc))
    return nothing
end

function in_proj_bwd_dx_slot_kernel(
    W_in::ct.TileArray{T, 3},
    dY::ct.TileArray{T, 2},
    sorted_tok_ids::ct.TileArray{Int32, 1},
    sorted_flat_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    dX::ct.TileArray{T, 2},
    padded_L::Int32,
    TILE_L::Int,
    TILE_D::Int,
    TILE_E::Int,
) where {T}
    bid = ct.bid(1)
    bid_l_0, bid_d_0 = swizzle_2d(padded_L, size(dX, 1), TILE_L, TILE_D, GROUP_SIZE_M, bid - Int32(1))
    bid_l = bid_l_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    tok_ids = ct.load(sorted_tok_ids, bid_l, (TILE_L,))
    flat_ids = ct.load(sorted_flat_ids, bid_l, (TILE_L,))
    head = block_heads[bid_l]

    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    acc = ct.zeros((TILE_D, TILE_L), Float32)
    num_e_tiles = cld(size(dY, 1), TILE_E)
    e_tile = Int32(1)
    while e_tile <= num_e_tiles
        e_idx = reshape_row(tile_range(e_tile, TILE_E))
        dy_tile = ct.gather(dY, (e_idx, reshape_col(flat_ids)))
        w_tile = reshape(
            ct.load(W_in, (e_tile, bid_d, head), (TILE_E, TILE_D, 1);
                    padding_mode=ct.PaddingMode.Zero),
            (TILE_E, TILE_D),
        )
        wt = transpose(w_tile)
        if T === Float32
            acc = muladd(convert(ct.Tile{ct.TFloat32}, wt),
                         convert(ct.Tile{ct.TFloat32}, dy_tile), acc)
        else
            acc = muladd(wt, dy_tile, acc)
        end
        e_tile += Int32(1)
    end

    dx_prev = ct.gather(dX, (d_idx, reshape_col(tok_ids)))
    ct.scatter(dX, (d_idx, reshape_col(tok_ids)), dx_prev + convert(ct.Tile{T}, acc))
    return nothing
end

function out_proj_bwd_dy_kernel(
    W_out::ct.TileArray{T, 3},
    dZ::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    tok_cols::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    dY::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_E::Int,
    TILE_D::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_e_0 = swizzle_2d(padded_M, size(dY, 1), TILE_M, TILE_E, GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_e = bid_e_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    tok_tile = ct.load(tok_cols, bid_m, (TILE_M,))
    head = block_heads[bid_m]

    acc = ct.zeros((TILE_E, TILE_M), Float32)
    num_d_tiles = cld(size(dZ, 1), TILE_D)
    d_tile = Int32(1)
    while d_tile <= num_d_tiles
        d_idx = reshape_row(tile_range(d_tile, TILE_D))
        dz_tile = ct.gather(dZ, (d_idx, reshape_col(tok_tile)))
        w_tile = reshape(
            ct.load(W_out, (d_tile, bid_e, head), (TILE_D, TILE_E, 1);
                    padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_E),
        )
        wt = transpose(w_tile)
        if T === Float32
            acc = muladd(convert(ct.Tile{ct.TFloat32}, wt),
                         convert(ct.Tile{ct.TFloat32}, dz_tile), acc)
        else
            acc = muladd(wt, dz_tile, acc)
        end
        d_tile += Int32(1)
    end

    e_idx = reshape_row(tile_range(bid_e, TILE_E))
    ct.scatter(dY, (e_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, acc))
    return nothing
end

function in_proj_bwd_dw_kernel(
    dY::ct.TileArray{T, 2},
    X::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    tok_cols::ct.TileArray{Int32, 1},
    head_block_starts::ct.TileArray{Int32, 1},
    dW_in::ct.TileArray{T, 3},
    num_m_splits::Int32,
    TILE_M::Int,
    TILE_E::Int,
    TILE_D::Int,
) where {T}
    bid_e = ct.bid(1)
    bid_d = ct.bid(2)
    bid_hs = ct.bid(3)
    head = fld(bid_hs - Int32(1), num_m_splits) + Int32(1)
    split = rem(bid_hs - Int32(1), num_m_splits) + Int32(1)

    block_start, block_stop = split_block_bounds(head_block_starts, head, split, num_m_splits)
    acc = ct.zeros((TILE_E, TILE_D), Float32)

    block_m = block_start
    while block_m < block_stop
        flat_ids = ct.load(sorted_ids, block_m, (TILE_M,))
        tok_tile = ct.load(tok_cols, block_m, (TILE_M,))

        e_idx = reshape_row(tile_range(bid_e, TILE_E))
        d_idx = reshape_row(tile_range(bid_d, TILE_D))
        dy_tile = ct.gather(dY, (e_idx, reshape_col(flat_ids)))
        x_tile = ct.gather(X, (d_idx, reshape_col(tok_tile)))
        acc = muladd(dy_tile, transpose(x_tile), acc)
        block_m += Int32(1)
    end

    ct.atomic_add(dW_in, (bid_e, bid_d, head), convert(ct.Tile{T}, acc))
    return nothing
end

function out_proj_bwd_dw_kernel(
    dZ::ct.TileArray{T, 2},
    Y::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    tok_cols::ct.TileArray{Int32, 1},
    head_block_starts::ct.TileArray{Int32, 1},
    dW_out::ct.TileArray{T, 3},
    num_m_splits::Int32,
    TILE_M::Int,
    TILE_D::Int,
    TILE_E::Int,
) where {T}
    bid_d = ct.bid(1)
    bid_e = ct.bid(2)
    bid_hs = ct.bid(3)
    head = fld(bid_hs - Int32(1), num_m_splits) + Int32(1)
    split = rem(bid_hs - Int32(1), num_m_splits) + Int32(1)

    block_start, block_stop = split_block_bounds(head_block_starts, head, split, num_m_splits)
    acc = ct.zeros((TILE_D, TILE_E), Float32)

    block_m = block_start
    while block_m < block_stop
        flat_ids = ct.load(sorted_ids, block_m, (TILE_M,))
        tok_tile = ct.load(tok_cols, block_m, (TILE_M,))

        d_idx = reshape_row(tile_range(bid_d, TILE_D))
        e_idx = reshape_row(tile_range(bid_e, TILE_E))
        dz_tile = ct.gather(dZ, (d_idx, reshape_col(tok_tile)))
        y_tile = ct.gather(Y, (e_idx, reshape_col(flat_ids)))
        acc = muladd(dz_tile, transpose(y_tile), acc)
        block_m += Int32(1)
    end

    ct.atomic_add(dW_out, (bid_d, bid_e, head), convert(ct.Tile{T}, acc))
    return nothing
end

function unpack_ycat_to_sparse_kernel(
    Y_cat::ct.TileArray{T, 2},
    Y_sparse::ct.TileArray{T, 2},
    K::Int32,
    TILE_E::Int,
    TILE_L::Int,
) where {T}
    slot = ct.bid(1)
    bid_l = ct.bid(2)

    e_idx = reshape_row(tile_range(Int32(1), TILE_E))
    l_idx = tile_range(bid_l, TILE_L)
    packed_rows = e_idx .+ Int32((slot - Int32(1)) * TILE_E)
    flat_ids = (l_idx .- Int32(1)) .* K .+ slot

    y_tile = ct.gather(Y_cat, (packed_rows, reshape_col(l_idx)))
    ct.scatter(Y_sparse, (e_idx, reshape_col(flat_ids)), y_tile)
    return nothing
end

function pack_sparse_to_ycat_kernel(
    Y_sparse::ct.TileArray{T, 2},
    Y_cat::ct.TileArray{T, 2},
    K::Int32,
    TILE_E::Int,
    TILE_L::Int,
) where {T}
    slot = ct.bid(1)
    bid_l = ct.bid(2)

    e_idx = reshape_row(tile_range(Int32(1), TILE_E))
    l_idx = tile_range(bid_l, TILE_L)
    packed_rows = e_idx .+ Int32((slot - Int32(1)) * TILE_E)
    flat_ids = (l_idx .- Int32(1)) .* K .+ slot

    y_tile = ct.gather(Y_sparse, (e_idx, reshape_col(flat_ids)))
    ct.scatter(Y_cat, (packed_rows, reshape_col(l_idx)), y_tile)
    return nothing
end

function launch_in_proj_fwd!(Y, W_in, X, dispatch::ProjectionDispatch; tile_e::Int=8, tile_d::Int=8)
    grid = Int(block_count(dispatch) * cld(size(Y, 1), tile_e))
    ct.launch(in_proj_fwd_kernel, grid,
        W_in, X, dispatch.sorted_ids, dispatch.tok_cols, dispatch.block_heads, Y,
        dispatch.padded_M,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_e), ct.Constant(tile_d))
    return Y
end

function launch_in_proj_fwd_contiguous!(Y, W_in, X, dispatch::ProjectionDispatch; tile_e::Int=8, tile_d::Int=8)
    grid = Int(block_count(dispatch) * cld(size(Y, 1), tile_e))
    ct.launch(in_proj_fwd_contiguous_kernel, grid,
        W_in, dispatch.sorted_ids, dispatch.head_block_starts, dispatch.block_heads, X, Y,
        dispatch.padded_M,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_e), ct.Constant(tile_d))
    return Y
end

function launch_out_proj_fwd!(Z, W_out, Y, dispatch::ProjectionDispatch;
                              tile_d::Int=64, tile_l::Int=64, tile_e::Int=32)
    fill!(Z, zero(eltype(Z)))
    for slot_dispatch in dispatch.slot_dispatches
        @assert Int(slot_dispatch.tile_l) == tile_l "slot dispatch was built for tile_l=$(Int(slot_dispatch.tile_l)), got $tile_l"
        grid = Int(block_count(slot_dispatch) * cld(size(Z, 1), tile_d))
        ct.launch(out_proj_fwd_slot_kernel, grid,
            W_out, Y, slot_dispatch.sorted_tok_ids, slot_dispatch.sorted_flat_ids, slot_dispatch.block_heads, Z,
            slot_dispatch.padded_L,
            ct.Constant(tile_l), ct.Constant(tile_d), ct.Constant(tile_e))
    end
    return Z
end

function launch_in_proj_bwd_dx!(dX, W_in, dY, dispatch::ProjectionDispatch;
                                tile_d::Int=64, tile_l::Int=64, tile_e::Int=32)
    fill!(dX, zero(eltype(dX)))
    for slot_dispatch in dispatch.slot_dispatches
        @assert Int(slot_dispatch.tile_l) == tile_l "slot dispatch was built for tile_l=$(Int(slot_dispatch.tile_l)), got $tile_l"
        grid = Int(block_count(slot_dispatch) * cld(size(dX, 1), tile_d))
        ct.launch(in_proj_bwd_dx_slot_kernel, grid,
            W_in, dY, slot_dispatch.sorted_tok_ids, slot_dispatch.sorted_flat_ids, slot_dispatch.block_heads, dX,
            slot_dispatch.padded_L,
            ct.Constant(tile_l), ct.Constant(tile_d), ct.Constant(tile_e))
    end
    return dX
end

function launch_out_proj_bwd_dy!(dY, W_out, dZ, dispatch::ProjectionDispatch; tile_e::Int=8, tile_d::Int=8)
    grid = Int(block_count(dispatch) * cld(size(dY, 1), tile_e))
    ct.launch(out_proj_bwd_dy_kernel, grid,
        W_out, dZ, dispatch.sorted_ids, dispatch.tok_cols, dispatch.block_heads, dY,
        dispatch.padded_M,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_e), ct.Constant(tile_d))
    return dY
end

function launch_in_proj_bwd_dw!(dW_in, dY, X, dispatch::ProjectionDispatch;
                                tile_e::Int=8, tile_d::Int=8, num_m_splits::Int=2)
    grid = (
        cld(size(dW_in, 1), tile_e),
        cld(size(dW_in, 2), tile_d),
        Int(dispatch.H) * num_m_splits,
    )
    ct.launch(in_proj_bwd_dw_kernel, grid,
        dY, X, dispatch.sorted_ids, dispatch.tok_cols, dispatch.head_block_starts, dW_in,
        Int32(num_m_splits),
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_e), ct.Constant(tile_d))
    return dW_in
end

function launch_out_proj_bwd_dw!(dW_out, dZ, Y, dispatch::ProjectionDispatch;
                                 tile_d::Int=8, tile_e::Int=4, num_m_splits::Int=2)
    grid = (
        cld(size(dW_out, 1), tile_d),
        cld(size(dW_out, 2), tile_e),
        Int(dispatch.H) * num_m_splits,
    )
    ct.launch(out_proj_bwd_dw_kernel, grid,
        dZ, Y, dispatch.sorted_ids, dispatch.tok_cols, dispatch.head_block_starts, dW_out,
        Int32(num_m_splits),
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_d), ct.Constant(tile_e))
    return dW_out
end

function make_dense_fixed_head_baseline(W_in::CuArray{T, 3},
                                        W_out::CuArray{T, 3},
                                        X::CuArray{T, 2},
                                        token_heads::AbstractArray{<:Integer}) where {T}
    d_head, d_model, _ = size(W_in)
    _, L = size(X)
    slot_heads = _global_slot_heads(_routing_matrix(token_heads, L))
    k = length(slot_heads)
    W_in_slots = CuArray{T, 2, CUDA.DeviceMemory}[]
    W_out_slots = CuArray{T, 2, CUDA.DeviceMemory}[]
    Y_slots = CuArray{T, 2, CUDA.DeviceMemory}[]
    for slot in 1:k
        push!(W_in_slots, copy(selectdim(W_in, 3, Int(slot_heads[slot]))))
        push!(W_out_slots, copy(selectdim(W_out, 3, Int(slot_heads[slot]))))
        push!(Y_slots, CUDA.zeros(T, d_head, L))
    end

    Z = CUDA.zeros(T, size(W_out, 1), L)
    return DenseFixedHeadBaseline(; X, W_in_slots, W_out_slots, Y_slots, Z)
end

function make_packed_fixed_head_baseline(W_in::CuArray{T, 3},
                                         W_out::CuArray{T, 3},
                                         X::CuArray{T, 2},
                                         token_heads::AbstractArray{<:Integer}) where {T}
    d_head, d_model, _ = size(W_in)
    _, L = size(X)
    slot_heads = _global_slot_heads(_routing_matrix(token_heads, L))
    k = length(slot_heads)
    W_in_cat = CUDA.zeros(T, k * d_head, d_model)
    W_out_cat = CUDA.zeros(T, size(W_out, 1), k * d_head)
    for slot in 1:k
        row_range = ((slot - 1) * d_head + 1):(slot * d_head)
        col_range = ((slot - 1) * d_head + 1):(slot * d_head)
        copyto!(view(W_in_cat, row_range, :), selectdim(W_in, 3, Int(slot_heads[slot])))
        copyto!(view(W_out_cat, :, col_range), selectdim(W_out, 3, Int(slot_heads[slot])))
    end
    Y_cat = CUDA.zeros(T, k * d_head, L)
    Y_sparse = CUDA.zeros(T, d_head, k * L)
    Z = CUDA.zeros(T, size(W_out, 1), L)
    return PackedFixedHeadBaseline(; X, W_in_cat, W_out_cat, Y_cat, Y_sparse, Z)
end

function dense_fixed_heads_in_proj!(baseline::DenseFixedHeadBaseline{T}) where {T}
    for slot in eachindex(baseline.Y_slots)
        mul!(baseline.Y_slots[slot], baseline.W_in_slots[slot], baseline.X)
    end
    return baseline
end

function dense_fixed_heads_out_proj!(baseline::DenseFixedHeadBaseline{T};
                                     reduce_tile_d::Int=64, reduce_tile_l::Int=64) where {T}
    fill!(baseline.Z, zero(T))
    for slot in eachindex(baseline.Y_slots)
        mul!(baseline.Z, baseline.W_out_slots[slot], baseline.Y_slots[slot], one(T), one(T))
    end
    return baseline.Z
end

function dense_fixed_heads_fwd!(baseline::DenseFixedHeadBaseline{T};
                                reduce_tile_d::Int=64, reduce_tile_l::Int=64) where {T}
    dense_fixed_heads_in_proj!(baseline)
    dense_fixed_heads_out_proj!(baseline; reduce_tile_d, reduce_tile_l)
    return baseline.Z
end

function packed_in_proj_gemm!(baseline::PackedFixedHeadBaseline{T}) where {T}
    mul!(baseline.Y_cat, baseline.W_in_cat, baseline.X)
    return baseline.Y_cat
end

function packed_in_proj_unpack!(baseline::PackedFixedHeadBaseline{T};
                                tile_e::Int, tile_l::Int, K::Int32) where {T}
    grid = (Int(K), cld(size(baseline.Y_cat, 2), tile_l))
    ct.launch(unpack_ycat_to_sparse_kernel, grid,
        baseline.Y_cat, baseline.Y_sparse, K, ct.Constant(tile_e), ct.Constant(tile_l))
    return baseline.Y_sparse
end

function packed_in_proj_practical!(baseline::PackedFixedHeadBaseline{T};
                                   tile_e::Int, tile_l::Int, K::Int32) where {T}
    packed_in_proj_gemm!(baseline)
    packed_in_proj_unpack!(baseline; tile_e, tile_l, K)
    return baseline.Y_sparse
end

function packed_out_proj_gemm!(baseline::PackedFixedHeadBaseline{T}) where {T}
    mul!(baseline.Z, baseline.W_out_cat, baseline.Y_cat)
    return baseline.Z
end

function packed_out_proj_from_sparse!(baseline::PackedFixedHeadBaseline{T}, Y_sparse;
                                      tile_e::Int, tile_l::Int, K::Int32) where {T}
    grid = (Int(K), cld(size(Y_sparse, 2) ÷ Int(K), tile_l))
    ct.launch(pack_sparse_to_ycat_kernel, grid,
        Y_sparse, baseline.Y_cat, K, ct.Constant(tile_e), ct.Constant(tile_l))
    packed_out_proj_gemm!(baseline)
    return baseline.Z
end

function packed_total_gemm!(baseline::PackedFixedHeadBaseline{T}) where {T}
    packed_in_proj_gemm!(baseline)
    packed_out_proj_gemm!(baseline)
    return baseline.Z
end

function packed_total_practical!(baseline::PackedFixedHeadBaseline{T};
                                 tile_e::Int, tile_l::Int, K::Int32) where {T}
    packed_in_proj_practical!(baseline; tile_e, tile_l, K)
    packed_out_proj_from_sparse!(baseline, baseline.Y_sparse; tile_e, tile_l, K)
    return baseline.Z
end

function reference_in_proj_fwd(W_in, X, token_heads)
    heads = _routing_matrix(token_heads, size(X, 2))
    d_head, d_model, H = size(W_in)
    @assert H >= maximum(heads)
    k, L = size(heads)
    L = size(X, 2)
    Y = zeros(eltype(W_in), d_head, k * L)
    for tok in 1:L
        x = view(X, :, tok)
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            Y[:, flat_id] .= view(W_in, :, :, head) * x
        end
    end
    return Y
end

function reference_out_proj_fwd(W_out, Y, token_heads)
    d_model, d_head, H = size(W_out)
    if token_heads isa AbstractVector
        k = length(token_heads)
        L = div(size(Y, 2), k)
    else
        k = size(token_heads, 1)
        L = size(token_heads, 2)
    end
    heads = _routing_matrix(token_heads, L)
    @assert H >= maximum(heads)
    @assert size(Y, 2) == k * L
    Z = zeros(eltype(W_out), d_model, L)
    for tok in 1:L
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            Z[:, tok] .+= view(W_out, :, :, head) * view(Y, :, flat_id)
        end
    end
    return Z
end

function reference_in_proj_bwd_dx(W_in, dY, token_heads)
    d_head, d_model, H = size(W_in)
    if token_heads isa AbstractVector
        k = length(token_heads)
        L = div(size(dY, 2), k)
    else
        k = size(token_heads, 1)
        L = size(token_heads, 2)
    end
    heads = _routing_matrix(token_heads, L)
    @assert H >= maximum(heads)
    @assert size(dY, 2) == k * L
    dX = zeros(eltype(W_in), d_model, L)
    for tok in 1:L
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            dX[:, tok] .+= transpose(view(W_in, :, :, head)) * view(dY, :, flat_id)
        end
    end
    return dX
end

function reference_out_proj_bwd_dy(W_out, dZ, token_heads)
    d_model, d_head, H = size(W_out)
    L = size(dZ, 2)
    heads = _routing_matrix(token_heads, L)
    @assert H >= maximum(heads)
    k = size(heads, 1)
    dY = zeros(eltype(W_out), d_head, k * L)
    for tok in 1:L
        dz = view(dZ, :, tok)
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            dY[:, flat_id] .= transpose(view(W_out, :, :, head)) * dz
        end
    end
    return dY
end

function reference_in_proj_bwd_dw(dY, X, token_heads, H)
    heads = _routing_matrix(token_heads, size(X, 2))
    d_head = size(dY, 1)
    d_model = size(X, 1)
    k, L = size(heads)
    L = size(X, 2)
    dW = zeros(eltype(X), d_head, d_model, H)
    for tok in 1:L
        x = view(X, :, tok)
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            dW[:, :, head] .+= view(dY, :, flat_id) * transpose(x)
        end
    end
    return dW
end

function reference_out_proj_bwd_dw(dZ, Y, token_heads, H)
    d_model = size(dZ, 1)
    d_head = size(Y, 1)
    L = size(dZ, 2)
    heads = _routing_matrix(token_heads, L)
    k = size(heads, 1)
    dW = zeros(eltype(dZ), d_model, d_head, H)
    for tok in 1:L
        dz = view(dZ, :, tok)
        for slot in 1:k
            head = heads[slot, tok]
            flat_id = (tok - 1) * k + slot
            dW[:, :, head] .+= dz * transpose(view(Y, :, flat_id))
        end
    end
    return dW
end

function make_example_problem(; T::Type{<:AbstractFloat}=Float32,
                              d_model::Int=8, d_head::Int=8, L::Int=5,
                              token_heads::Union{Nothing, AbstractMatrix{<:Integer}}=nothing,
                              k_to_head::AbstractVector{<:Integer}=[1, 2, 1],
                              tile_m::Int=4)
    token_heads_i32 = token_heads === nothing ? _routing_matrix(k_to_head, L) : _routing_matrix(token_heads, L)
    H = maximum(token_heads_i32)
    W_in = CUDA.rand(T, d_head, d_model, H)
    W_out = CUDA.rand(T, d_model, d_head, H)
    X = CUDA.rand(T, d_model, L)
    dispatch = build_projection_dispatch(token_heads_i32, tile_m)
    return (; W_in, W_out, X, dispatch, token_heads=token_heads_i32)
end

function prepare(; benchmark::Bool=false,
                  T::Type{<:AbstractFloat}=benchmark ? Float16 : Float32,
                  d_model::Int=benchmark ? 3072 : 8,
                  d_head::Int=benchmark ? 128 : 8,
                  L::Int=benchmark ? 8192 : 5,
                  H::Int=benchmark ? 96 : 3,
                  k::Int=benchmark ? 24 : 3,
                  routing::Symbol=:fixed,
                  token_heads::Union{Nothing, AbstractMatrix{<:Integer}}=nothing,
                  tile_m::Int=benchmark ? 64 : 4)
    routing_matrix = if token_heads !== nothing
        _routing_matrix(token_heads, L)
    elseif routing === :fixed
        k_to_head = H >= k ? Int32.(collect(1:k)) : Int32[mod1(slot, H) for slot in 1:k]
        _routing_matrix(k_to_head, L)
    elseif routing === :random
        Int32.(rand(1:H, k, L))
    else
        error("unsupported routing=$(routing); expected :fixed or :random")
    end

    problem = make_example_problem(; T, d_model, d_head, L, token_heads=routing_matrix, tile_m)
    fixed_routing = _is_fixed_routing(problem.token_heads)
    dense = fixed_routing ? make_dense_fixed_head_baseline(problem.W_in, problem.W_out, problem.X, problem.token_heads) : nothing
    packed = fixed_routing ? make_packed_fixed_head_baseline(problem.W_in, problem.W_out, problem.X, problem.token_heads) : nothing
    Y = CUDA.zeros(T, size(problem.W_in, 1), size(problem.token_heads, 1) * size(problem.X, 2))
    Z = CUDA.zeros(T, size(problem.W_out, 1), size(problem.X, 2))
    return merge(problem, (; dense, packed, Y, Z, fixed_routing))
end

function verify(data, result)
    sparse_Z = Array(result.Z)
    ref_Z = reference_out_proj_fwd(
        Array(data.W_out),
        reference_in_proj_fwd(Array(data.W_in), Array(data.X), Array(data.token_heads)),
        Array(data.token_heads),
    )
    tol = eltype(data.X) === Float16 ? (rtol=5e-2, atol=5e-2) : (rtol=1e-4, atol=1e-4)
    @assert isapprox(sparse_Z, ref_Z; tol...)
    if data.dense !== nothing
        @assert isapprox(Array(data.dense.Z), ref_Z; tol...)
    end
    if data.packed !== nothing
        @assert isapprox(Array(data.packed.Z), ref_Z; tol...)
    end
end

function _time_cuda_ms(run_once::Function; warmup::Int, nruns::Int)
    CUDA.@sync for _ in 1:warmup
        run_once()
    end
    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed run_once()
        push!(times, t * 1000)
    end
    return times
end

function _time_cpu_ms(run_once::Function; warmup::Int, nruns::Int)
    for _ in 1:warmup
        run_once()
    end
    times = Float64[]
    for _ in 1:nruns
        t = @elapsed run_once()
        push!(times, t * 1000)
    end
    return times
end

function run(data; tile_e::Int=64, tile_d::Int=64, tile_e_small::Int=128,
             tile_l::Int=64,
             reduce_tile_d::Int=64, reduce_tile_l::Int=64,
             nruns::Int=1, warmup::Int=0)
    use_contiguous = data.fixed_routing
    sparse_in_gather = _time_cuda_ms(;
        warmup, nruns
    ) do
        launch_in_proj_fwd!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
    end
    sparse_in_contiguous = use_contiguous ? _time_cuda_ms(;
        warmup, nruns
    ) do
        launch_in_proj_fwd_contiguous!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
    end : Float64[]

    if use_contiguous
        launch_in_proj_fwd_contiguous!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
    else
        launch_in_proj_fwd!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
    end
    sparse_out = _time_cuda_ms(;
        warmup, nruns
    ) do
        launch_out_proj_fwd!(data.Z, data.W_out, data.Y, data.dispatch;
                             tile_d, tile_l, tile_e=tile_e_small)
    end
    total = _time_cuda_ms(;
        warmup, nruns
    ) do
        if use_contiguous
            launch_in_proj_fwd_contiguous!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
        else
            launch_in_proj_fwd!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
        end
        launch_out_proj_fwd!(data.Z, data.W_out, data.Y, data.dispatch;
                             tile_d, tile_l, tile_e=tile_e_small)
    end

    if data.dense !== nothing
        dense_fixed_heads_fwd!(data.dense; reduce_tile_d, reduce_tile_l)
    end
    if data.packed !== nothing
        packed_total_practical!(data.packed; tile_e=tile_e_small, tile_l, K=data.dispatch.k)
    end

    return (;
        Z=data.Z,
        times=total,
        times_in=use_contiguous ? sparse_in_contiguous : sparse_in_gather,
        times_in_gather=sparse_in_gather,
        times_in_contiguous=sparse_in_contiguous,
        times_out=sparse_out,
    )
end

function run_others(data; reduce_tile_d::Int=64, reduce_tile_l::Int=64,
                    tile_d::Int=64, tile_e::Int=64, tile_e_small::Int=128, tile_l::Int=64,
                    nruns::Int=1, warmup::Int=0)
    results = Dict{String, Vector{Float64}}()
    dispatch_ms = _time_cpu_ms(; warmup, nruns) do
        build_projection_dispatch(Array(data.token_heads), Int(data.dispatch.tile_m))
    end
    sparse_in_gather = _time_cuda_ms(; warmup, nruns) do
        launch_in_proj_fwd!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
    end
    results["Dispatch build CPU"] = dispatch_ms
    results["Sparse in-proj gather"] = sparse_in_gather
    if data.fixed_routing
        sparse_in_contiguous = _time_cuda_ms(; warmup, nruns) do
            launch_in_proj_fwd_contiguous!(data.Y, data.W_in, data.X, data.dispatch; tile_e, tile_d)
        end
        results["Sparse in-proj contiguous"] = sparse_in_contiguous
    end
    if data.dense !== nothing
        dense_in = _time_cuda_ms(; warmup, nruns) do
            dense_fixed_heads_in_proj!(data.dense)
        end
        dense_fixed_heads_in_proj!(data.dense)
        dense_out = _time_cuda_ms(; warmup, nruns) do
            dense_fixed_heads_out_proj!(data.dense; reduce_tile_d, reduce_tile_l)
        end
        dense_total = _time_cuda_ms(; warmup, nruns) do
            dense_fixed_heads_fwd!(data.dense; reduce_tile_d, reduce_tile_l)
        end
        results["Dense fixed-head total"] = dense_total
        results["Dense fixed-head in"] = dense_in
        results["Dense fixed-head out"] = dense_out
    end
    if data.packed !== nothing
        packed_in_gemm = _time_cuda_ms(; warmup, nruns) do
            packed_in_proj_gemm!(data.packed)
        end
        packed_in_layout = _time_cuda_ms(; warmup, nruns) do
            packed_in_proj_practical!(data.packed; tile_e=tile_e_small, tile_l, K=data.dispatch.k)
        end
        packed_in_proj_gemm!(data.packed)
        packed_out_gemm = _time_cuda_ms(; warmup, nruns) do
            packed_out_proj_gemm!(data.packed)
        end
        packed_in_proj_practical!(data.packed; tile_e=tile_e_small, tile_l, K=data.dispatch.k)
        packed_out_layout = _time_cuda_ms(; warmup, nruns) do
            packed_out_proj_from_sparse!(data.packed, data.packed.Y_sparse; tile_e=tile_e_small, tile_l, K=data.dispatch.k)
        end
        packed_total_gemm = _time_cuda_ms(; warmup, nruns) do
            packed_total_gemm!(data.packed)
        end
        packed_total_layout = _time_cuda_ms(; warmup, nruns) do
            packed_total_practical!(data.packed; tile_e=tile_e_small, tile_l, K=data.dispatch.k)
        end
        results["Packed in-proj GEMM only"] = packed_in_gemm
        results["Packed in-proj + layout"] = packed_in_layout
        results["Packed out-proj GEMM only"] = packed_out_gemm
        results["Packed out-proj + layout"] = packed_out_layout
        results["Packed total GEMM only"] = packed_total_gemm
        results["Packed total + layout"] = packed_total_layout
    end
    return results
end

function run_example(; tile_e::Int=8, tile_d::Int=8, tile_e_small::Int=8)
    problem = make_example_problem()
    Y = CUDA.zeros(Float32, size(problem.W_in, 1), Int(problem.dispatch.k) * size(problem.X, 2))
    Z = CUDA.zeros(Float32, size(problem.W_out, 1), size(problem.X, 2))
    dY = CUDA.rand(Float32, size(Y)...)
    dZ = CUDA.rand(Float32, size(Z)...)
    dX = CUDA.zeros(Float32, size(problem.X)...)
    back_dY = CUDA.zeros(Float32, size(Y)...)
    dW_in = CUDA.zeros(Float32, size(problem.W_in)...)
    dW_out = CUDA.zeros(Float32, size(problem.W_out)...)

    launch_in_proj_fwd!(Y, problem.W_in, problem.X, problem.dispatch; tile_e, tile_d)
    launch_out_proj_fwd!(Z, problem.W_out, Y, problem.dispatch; tile_d, tile_l=4, tile_e=tile_e_small)
    launch_in_proj_bwd_dx!(dX, problem.W_in, dY, problem.dispatch; tile_d, tile_l=4, tile_e=tile_e_small)
    launch_out_proj_bwd_dy!(back_dY, problem.W_out, dZ, problem.dispatch; tile_e, tile_d)
    launch_in_proj_bwd_dw!(dW_in, dY, problem.X, problem.dispatch; tile_e, tile_d, num_m_splits=2)
    launch_out_proj_bwd_dw!(dW_out, dZ, Y, problem.dispatch; tile_d, tile_e=tile_e_small, num_m_splits=2)
    CUDA.synchronize()

    refs = (
        Y=reference_in_proj_fwd(Array(problem.W_in), Array(problem.X), Array(problem.token_heads)),
        Z=reference_out_proj_fwd(Array(problem.W_out), Array(Y), Array(problem.token_heads)),
        dX=reference_in_proj_bwd_dx(Array(problem.W_in), Array(dY), Array(problem.token_heads)),
        dY=reference_out_proj_bwd_dy(Array(problem.W_out), Array(dZ), Array(problem.token_heads)),
        dW_in=reference_in_proj_bwd_dw(Array(dY), Array(problem.X), Array(problem.token_heads), Int(problem.dispatch.H)),
        dW_out=reference_out_proj_bwd_dw(Array(dZ), Array(Y), Array(problem.token_heads), Int(problem.dispatch.H)),
    )

    @assert Array(Y) ≈ refs.Y rtol=1e-4 atol=1e-4
    @assert Array(Z) ≈ refs.Z rtol=1e-4 atol=1e-4
    @assert Array(dX) ≈ refs.dX rtol=1e-4 atol=1e-4
    @assert Array(back_dY) ≈ refs.dY rtol=1e-4 atol=1e-4
    @assert Array(dW_in) ≈ refs.dW_in rtol=1e-4 atol=1e-4
    @assert Array(dW_out) ≈ refs.dW_out rtol=1e-4 atol=1e-4

    return nothing
end

function main()
    run_example()
    return nothing
end

end

# EXCLUDE FROM TESTING

module SparseFlashMHFExample

using CUDA
using LinearAlgebra
using Random
import cuTile as ct

include(joinpath(@__DIR__, "projection_kernels_impl.jl"))
const pk = ProjectionKernelsExample

export tensorcore_type,
       accumulate_type,
       sparse_mhffn!,
       ∇sparse_mhffn!,
       reference_sparse_mhffn,
       reference_sparse_mhffn_dq,
       reference_sparse_mhffn_dkuv,
       make_example_problem,
       prepare,
       run,
       verify,
       run_example,
       main

tensorcore_type(::Type{Float32}) = ct.TFloat32
tensorcore_type(::Type{Float16}) = Float16
tensorcore_type(::Type{T}) where {T} = T

accumulate_type(::Type{Float16}) = Float32
accumulate_type(::Type{ct.TFloat32}) = Float32
accumulate_type(::Type{Float32}) = Float32
accumulate_type(::Type{T}) where {T} = T

@inline function reshape_scalar_row(value::Int32)
    reshape(ct.broadcast_to(ct.Tile(value), (1,)), (1, 1))
end

@inline function tile_range(tile_idx::Int32, tile_size::Int)
    base = (tile_idx - Int32(1)) * Int32(tile_size) + Int32(1)
    base .+ ct.arange((tile_size,), Int32) .- Int32(1)
end

@inline reshape_col(idx::ct.Tile) = reshape(idx, (1, size(idx, 1)))
@inline reshape_row(idx::ct.Tile) = reshape(idx, (size(idx, 1), 1))
@inline valid_flat_ids(flat_ids::ct.Tile) = flat_ids .>= Int32(1)
@inline safe_flat_ids(flat_ids::ct.Tile) = max.(flat_ids, Int32(1))

function sparse_mhffn_fwd_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    sorted_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    O::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_d_0 = pk.swizzle_2d(padded_M, size(Q, 1), TILE_M, TILE_D, pk.GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    valid = valid_flat_ids(flat_ids)
    safe_ids = safe_flat_ids(flat_ids)
    head = block_heads[bid_m]
    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
    q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
    acc = ct.zeros((TILE_D, TILE_M), Float32)

    num_i = cld(size(K, 1), TILE_I)
    i = Int32(1)
    while i <= num_i
        k = reshape(
            ct.load(K, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        u = reshape(
            ct.load(U, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        v = reshape(
            ct.load(V, (bid_d, i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_I),
        )

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))

        a = m ./ (1 .+ exp.(0 .- m)) .* n

        acc = muladd(v, a, acc)
        i += Int32(1)
    end

    acc = ifelse.(reshape(valid, (1, TILE_M)), acc, 0f0)
    ct.scatter(O, (d_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, acc))
    return nothing
end

function sparse_mhffn_fwd_gated_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    R::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    O::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
    D_E::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_d_0 = pk.swizzle_2d(padded_M, size(Q, 1), TILE_M, TILE_D, pk.GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    valid = valid_flat_ids(flat_ids)
    safe_ids = safe_flat_ids(flat_ids)
    head = block_heads[bid_m]
    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
    q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
    acc = ct.zeros((TILE_D, TILE_M), Float32)
    tiles_per_expert = Int32(D_E ÷ TILE_I)

    num_i = cld(size(K, 1), TILE_I)
    i = Int32(1)
    while i <= num_i
        k = reshape(
            ct.load(K, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        u = reshape(
            ct.load(U, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        v = reshape(
            ct.load(V, (bid_d, i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_I),
        )

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))

        a = m ./ (1 .+ exp.(0 .- m)) .* n
        e = fld(i - Int32(1), tiles_per_expert) + Int32(1)
        r = ct.gather(R, (reshape_scalar_row(e), reshape_col(safe_ids)))
        r = ifelse.(reshape(valid, (1, TILE_M)), r, T(0))
        a = a .* r

        acc = muladd(v, a, acc)
        i += Int32(1)
    end

    acc = ifelse.(reshape(valid, (1, TILE_M)), acc, 0f0)
    ct.scatter(O, (d_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, acc))
    return nothing
end

function sparse_mhffn_bwd_dq_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    dO::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    dQ::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_d_0 = pk.swizzle_2d(padded_M, size(Q, 1), TILE_M, TILE_D, pk.GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    valid = valid_flat_ids(flat_ids)
    safe_ids = safe_flat_ids(flat_ids)
    head = block_heads[bid_m]
    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
    q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
    ō = ct.gather(dO, (d_idx, reshape_col(safe_ids)))
    ō = ifelse.(reshape(valid, (1, TILE_M)), ō, T(0))
    q̄_acc = ct.zeros((TILE_D, TILE_M), Float32)

    num_i = cld(size(K, 1), TILE_I)
    i = Int32(1)
    while i <= num_i
        k = reshape(
            ct.load(K, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        u = reshape(
            ct.load(U, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        v = reshape(
            ct.load(V, (bid_d, i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_I),
        )

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))
        ā = muladd(transpose(v), ō, ct.zeros((TILE_I, TILE_M), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm
        n̄ = ā .* silu_m

        q̄_acc = muladd(transpose(k), m̄, q̄_acc)
        q̄_acc = muladd(transpose(u), n̄, q̄_acc)
        i += Int32(1)
    end

    q̄_acc = ifelse.(reshape(valid, (1, TILE_M)), q̄_acc, 0f0)
    ct.scatter(dQ, (d_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, q̄_acc))
    return nothing
end

function sparse_mhffn_bwd_dq_gated_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    dO::ct.TileArray{T, 2},
    R::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    block_heads::ct.TileArray{Int32, 1},
    dQ::ct.TileArray{T, 2},
    dR::ct.TileArray{T, 2},
    padded_M::Int32,
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
    D_E::Int,
) where {T}
    bid = ct.bid(1)
    bid_m_0, bid_d_0 = pk.swizzle_2d(padded_M, size(Q, 1), TILE_M, TILE_D, pk.GROUP_SIZE_M, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_d = bid_d_0 + Int32(1)

    flat_ids = ct.load(sorted_ids, bid_m, (TILE_M,))
    valid = valid_flat_ids(flat_ids)
    safe_ids = safe_flat_ids(flat_ids)
    head = block_heads[bid_m]
    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
    q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
    ō = ct.gather(dO, (d_idx, reshape_col(safe_ids)))
    ō = ifelse.(reshape(valid, (1, TILE_M)), ō, T(0))
    q̄_acc = ct.zeros((TILE_D, TILE_M), Float32)
    dr_acc = ct.zeros((1, TILE_M), Float32)
    tiles_per_expert = Int32(D_E ÷ TILE_I)

    num_i = cld(size(K, 1), TILE_I)
    i = Int32(1)
    while i <= num_i
        k = reshape(
            ct.load(K, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        u = reshape(
            ct.load(U, (i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_I, TILE_D),
        )
        v = reshape(
            ct.load(V, (bid_d, i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
            (TILE_D, TILE_I),
        )

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))
        ā = muladd(transpose(v), ō, ct.zeros((TILE_I, TILE_M), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
        e = fld(i - Int32(1), tiles_per_expert) + Int32(1)
        r = ct.gather(R, (reshape_scalar_row(e), reshape_col(safe_ids)))
        r = ifelse.(reshape(valid, (1, TILE_M)), r, T(0))

        dr_acc = dr_acc .+ sum(ā .* silu_m .* n, dims=1)
        if iszero(mod(i, tiles_per_expert))
            dr_acc = ifelse.(reshape(valid, (1, TILE_M)), dr_acc, 0f0)
            ct.scatter(dR, (reshape_scalar_row(e), reshape_col(flat_ids)), convert(ct.Tile{T}, dr_acc))
            dr_acc = ct.zeros((1, TILE_M), Float32)
        end

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm .* r
        n̄ = ā .* silu_m .* r

        q̄_acc = muladd(transpose(k), m̄, q̄_acc)
        q̄_acc = muladd(transpose(u), n̄, q̄_acc)
        i += Int32(1)
    end

    q̄_acc = ifelse.(reshape(valid, (1, TILE_M)), q̄_acc, 0f0)
    ct.scatter(dQ, (d_idx, reshape_col(flat_ids)), convert(ct.Tile{T}, q̄_acc))
    return nothing
end

function sparse_mhffn_bwd_dkuv_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    dO::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    head_block_starts::ct.TileArray{Int32, 1},
    dK::ct.TileArray{T, 3},
    dU::ct.TileArray{T, 3},
    dV::ct.TileArray{T, 3},
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
) where {T}
    bid_i = ct.bid(1)
    bid_d = ct.bid(2)
    head = ct.bid(3)

    k = reshape(
        ct.load(K, (bid_i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_I, TILE_D),
    )
    u = reshape(
        ct.load(U, (bid_i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_I, TILE_D),
    )
    v = reshape(
        ct.load(V, (bid_d, bid_i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_D, TILE_I),
    )

    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    k̄_acc = ct.zeros((TILE_I, TILE_D), Float32)
    ū_acc = ct.zeros((TILE_I, TILE_D), Float32)
    v̄_acc = ct.zeros((TILE_D, TILE_I), Float32)

    block_m = head_block_starts[head]
    block_stop = head_block_starts[head + Int32(1)]
    while block_m < block_stop
        flat_ids = ct.load(sorted_ids, block_m, (TILE_M,))
        valid = valid_flat_ids(flat_ids)
        safe_ids = safe_flat_ids(flat_ids)
        q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
        q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
        ō = ct.gather(dO, (d_idx, reshape_col(safe_ids)))
        ō = ifelse.(reshape(valid, (1, TILE_M)), ō, T(0))

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))
        ā = muladd(transpose(v), ō, ct.zeros((TILE_I, TILE_M), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
        a = silu_m .* n
        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm
        n̄ = ā .* silu_m

        k̄_acc = muladd(m̄, transpose(q), k̄_acc)
        ū_acc = muladd(n̄, transpose(q), ū_acc)
        v̄_acc = muladd(ō, transpose(a), v̄_acc)
        block_m += Int32(1)
    end

    ct.store(dK, (bid_i, bid_d, head), reshape(convert(ct.Tile{T}, k̄_acc), (TILE_I, TILE_D, 1)))
    ct.store(dU, (bid_i, bid_d, head), reshape(convert(ct.Tile{T}, ū_acc), (TILE_I, TILE_D, 1)))
    ct.store(dV, (bid_d, bid_i, head), reshape(convert(ct.Tile{T}, v̄_acc), (TILE_D, TILE_I, 1)))
    return nothing
end

function sparse_mhffn_bwd_dkuv_gated_kernel(
    Q::ct.TileArray{T, 2},
    K::ct.TileArray{T, 3},
    U::ct.TileArray{T, 3},
    V::ct.TileArray{T, 3},
    dO::ct.TileArray{T, 2},
    R::ct.TileArray{T, 2},
    sorted_ids::ct.TileArray{Int32, 1},
    head_block_starts::ct.TileArray{Int32, 1},
    dK::ct.TileArray{T, 3},
    dU::ct.TileArray{T, 3},
    dV::ct.TileArray{T, 3},
    TILE_M::Int,
    TILE_D::Int,
    TILE_I::Int,
    D_E::Int,
) where {T}
    bid_i = ct.bid(1)
    bid_d = ct.bid(2)
    head = ct.bid(3)

    k = reshape(
        ct.load(K, (bid_i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_I, TILE_D),
    )
    u = reshape(
        ct.load(U, (bid_i, bid_d, head), (TILE_I, TILE_D, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_I, TILE_D),
    )
    v = reshape(
        ct.load(V, (bid_d, bid_i, head), (TILE_D, TILE_I, 1); padding_mode=ct.PaddingMode.Zero),
        (TILE_D, TILE_I),
    )

    tiles_per_expert = Int32(D_E ÷ TILE_I)
    e = fld(bid_i - Int32(1), tiles_per_expert) + Int32(1)
    d_idx = reshape_row(tile_range(bid_d, TILE_D))
    k̄_acc = ct.zeros((TILE_I, TILE_D), Float32)
    ū_acc = ct.zeros((TILE_I, TILE_D), Float32)
    v̄_acc = ct.zeros((TILE_D, TILE_I), Float32)

    block_m = head_block_starts[head]
    block_stop = head_block_starts[head + Int32(1)]
    while block_m < block_stop
        flat_ids = ct.load(sorted_ids, block_m, (TILE_M,))
        valid = valid_flat_ids(flat_ids)
        safe_ids = safe_flat_ids(flat_ids)
        q = ct.gather(Q, (d_idx, reshape_col(safe_ids)))
        q = ifelse.(reshape(valid, (1, TILE_M)), q, T(0))
        ō = ct.gather(dO, (d_idx, reshape_col(safe_ids)))
        ō = ifelse.(reshape(valid, (1, TILE_M)), ō, T(0))
        r = ct.gather(R, (reshape_scalar_row(e), reshape_col(safe_ids)))
        r = ifelse.(reshape(valid, (1, TILE_M)), r, T(0))

        m = muladd(k, q, ct.zeros((TILE_I, TILE_M), Float32))
        n = muladd(u, q, ct.zeros((TILE_I, TILE_M), Float32))
        ā = muladd(transpose(v), ō, ct.zeros((TILE_I, TILE_M), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
        a = silu_m .* n .* r
        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm .* r
        n̄ = ā .* silu_m .* r

        k̄_acc = muladd(m̄, transpose(q), k̄_acc)
        ū_acc = muladd(n̄, transpose(q), ū_acc)
        v̄_acc = muladd(ō, transpose(a), v̄_acc)
        block_m += Int32(1)
    end

    ct.store(dK, (bid_i, bid_d, head), reshape(convert(ct.Tile{T}, k̄_acc), (TILE_I, TILE_D, 1)))
    ct.store(dU, (bid_i, bid_d, head), reshape(convert(ct.Tile{T}, ū_acc), (TILE_I, TILE_D, 1)))
    ct.store(dV, (bid_d, bid_i, head), reshape(convert(ct.Tile{T}, v̄_acc), (TILE_D, TILE_I, 1)))
    return nothing
end

function sparse_mhffn!(O, Q, K, U, V, dispatch::pk.ProjectionDispatch;
                       R=nothing, D_E::Union{Nothing, Int}=nothing,
                       tile_d::Int=size(Q, 1), tile_i::Int=64)
    grid = Int(pk.block_count(dispatch) * cld(size(Q, 1), tile_d))
    @assert !isnothing(R) "Sparse FlashMHF v1 requires explicit expert weights R"
    @assert !isnothing(D_E) "Sparse FlashMHF v1 requires D_E when R is provided"
    @assert D_E % tile_i == 0 "D_E=$D_E must be divisible by tile_i=$tile_i"
    ct.launch(sparse_mhffn_fwd_gated_kernel, grid,
        Q, K, U, V, R,
        dispatch.sorted_ids, dispatch.block_heads, O,
        dispatch.padded_M,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_d), ct.Constant(tile_i), D_E)
    return O
end

function ∇sparse_mhffn!(dQ, dK, dU, dV, dO, Q, K, U, V, dispatch::pk.ProjectionDispatch;
                        R=nothing, dR=nothing, D_E::Union{Nothing, Int}=nothing,
                        tile_d::Int=size(Q, 1), tile_i::Int=64)
    grid_q = Int(pk.block_count(dispatch) * cld(size(Q, 1), tile_d))
    @assert !isnothing(R) "Sparse FlashMHF v1 requires explicit expert weights R"
    @assert !isnothing(dR) "Sparse FlashMHF v1 requires dR when R is provided"
    @assert !isnothing(D_E) "Sparse FlashMHF v1 requires D_E when R is provided"
    @assert D_E % tile_i == 0 "D_E=$D_E must be divisible by tile_i=$tile_i"
    ct.launch(sparse_mhffn_bwd_dq_gated_kernel, grid_q,
        Q, K, U, V, dO, R,
        dispatch.sorted_ids, dispatch.block_heads, dQ, dR,
        dispatch.padded_M,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_d), ct.Constant(tile_i), D_E)

    grid_w = (cld(size(K, 1), tile_i), cld(size(Q, 1), tile_d), Int(dispatch.H))
    ct.launch(sparse_mhffn_bwd_dkuv_gated_kernel, grid_w,
        Q, K, U, V, dO, R,
        dispatch.sorted_ids, dispatch.head_block_starts,
        dK, dU, dV,
        ct.Constant(Int(dispatch.tile_m)), ct.Constant(tile_d), ct.Constant(tile_i), D_E)
    return nothing
end

function _head_for_flat(token_heads, flat_id)
    k = size(token_heads, 1)
    return token_heads[mod1(flat_id, k), fld1(flat_id, k)]
end

function reference_sparse_mhffn(Q, K, U, V, token_heads; R=nothing, D_E::Union{Nothing, Int}=nothing)
    D = size(Q, 1)
    k, L = size(token_heads)
    I = size(K, 1)
    O = zeros(eltype(Q), D, k * L)
    for flat_id in 1:(k * L)
        h = _head_for_flat(token_heads, flat_id)
        q = view(Q, :, flat_id)
        m = view(K, :, :, h) * q
        n = view(U, :, :, h) * q
        a = (m ./ (1 .+ exp.(-m))) .* n
        if !isnothing(R)
            @assert !isnothing(D_E)
            for e in 1:size(R, 1)
                rng = ((e - 1) * D_E + 1):(e * D_E)
                a[rng] .*= R[e, flat_id]
            end
        end
        O[:, flat_id] .= view(V, :, :, h) * a
    end
    return O
end

function reference_sparse_mhffn_dq(Q, K, U, V, dO, token_heads; R=nothing, D_E::Union{Nothing, Int}=nothing)
    D = size(Q, 1)
    k, L = size(token_heads)
    dQ = zeros(eltype(Q), D, k * L)
    dR = isnothing(R) ? nothing : zeros(eltype(Q), size(R))
    for flat_id in 1:(k * L)
        h = _head_for_flat(token_heads, flat_id)
        q = view(Q, :, flat_id)
        ō = view(dO, :, flat_id)

        m = view(K, :, :, h) * q
        n = view(U, :, :, h) * q
        sig = 1 ./ (1 .+ exp.(-m))
        silu_m = m .* sig
        ā = transpose(view(V, :, :, h)) * ō
        if !isnothing(R)
            @assert !isnothing(D_E)
            for e in 1:size(R, 1)
                rng = ((e - 1) * D_E + 1):(e * D_E)
                dR[e, flat_id] = sum(ā[rng] .* silu_m[rng] .* n[rng])
                ā[rng] .*= R[e, flat_id]
            end
        end
        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm
        n̄ = ā .* silu_m
        dQ[:, flat_id] .= transpose(view(K, :, :, h)) * m̄ + transpose(view(U, :, :, h)) * n̄
    end
    return isnothing(R) ? (dQ, nothing) : (dQ, dR)
end

function reference_sparse_mhffn_dkuv(Q, K, U, V, dO, token_heads; R=nothing, D_E::Union{Nothing, Int}=nothing)
    dK = zeros(eltype(K), size(K))
    dU = zeros(eltype(U), size(U))
    dV = zeros(eltype(V), size(V))
    k, L = size(token_heads)
    for flat_id in 1:(k * L)
        h = _head_for_flat(token_heads, flat_id)
        q = view(Q, :, flat_id)
        ō = view(dO, :, flat_id)

        m = view(K, :, :, h) * q
        n = view(U, :, :, h) * q
        sig = 1 ./ (1 .+ exp.(-m))
        silu_m = m .* sig
        a = silu_m .* n
        ā = transpose(view(V, :, :, h)) * ō
        if !isnothing(R)
            @assert !isnothing(D_E)
            for e in 1:size(R, 1)
                rng = ((e - 1) * D_E + 1):(e * D_E)
                a[rng] .*= R[e, flat_id]
                ā[rng] .*= R[e, flat_id]
            end
        end
        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))
        m̄ = ā .* n .* dsilu_dm
        n̄ = ā .* silu_m
        dK[:, :, h] .+= m̄ * transpose(q)
        dU[:, :, h] .+= n̄ * transpose(q)
        dV[:, :, h] .+= ō * transpose(a)
    end
    return dK, dU, dV
end

function make_example_problem(; T::Type{<:AbstractFloat}=Float32,
                              d_head::Int=8, H::Int=4, L::Int=5, k::Int=2,
                              d_inter::Int=16, tile_m::Int=4,
                              routing::Symbol=:random, with_router::Bool=true,
                              seed::Int=0)
    Random.seed!(seed)
    token_heads = routing === :fixed ?
        repeat(reshape(Int32.(collect(1:k)), :, 1), 1, L) :
        Int32.(rand(1:H, k, L))
    dispatch = pk.build_projection_dispatch(token_heads, tile_m)
    Q = CUDA.cu(rand(T, d_head, k * L))
    K = CUDA.cu(rand(T, d_inter, d_head, H))
    U = CUDA.cu(rand(T, d_inter, d_head, H))
    V = CUDA.cu(rand(T, d_head, d_inter, H))
    if with_router
        E = 4
        @assert d_inter % E == 0
        D_E = d_inter ÷ E
        R = CUDA.cu(rand(T, E, k * L))
    else
        E = nothing
        D_E = nothing
        R = nothing
    end
    return (; Q, K, U, V, R, dispatch, token_heads, D_E, E)
end

function prepare(; benchmark::Bool=false, kwargs...)
    return make_example_problem(; d_head=benchmark ? 128 : 8,
        H=benchmark ? 96 : 4,
        L=benchmark ? 8192 : 5,
        k=benchmark ? 24 : 2,
        d_inter=benchmark ? 3072 : 16,
        tile_m=benchmark ? 64 : 4,
        T=Float32,
        kwargs...)
end

function run(data; tile_d::Int=size(data.Q, 1), tile_i::Int=64, nruns::Int=1, warmup::Int=0)
    O = similar(data.Q)
    CUDA.@sync for _ in 1:warmup
        sparse_mhffn!(O, data.Q, data.K, data.U, data.V, data.dispatch; R=data.R, D_E=data.D_E, tile_d, tile_i)
    end
    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed sparse_mhffn!(O, data.Q, data.K, data.U, data.V, data.dispatch; R=data.R, D_E=data.D_E, tile_d, tile_i)
        push!(times, t * 1000)
    end
    return (; O, times)
end

function verify(data, result)
    ref = reference_sparse_mhffn(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(data.token_heads); R=isnothing(data.R) ? nothing : Array(data.R), D_E=data.D_E)
    tol = eltype(data.Q) === Float16 ? (rtol=5e-2, atol=5e-2) : (rtol=1e-3, atol=1e-3)
    @assert isapprox(Array(result.O), ref; tol...)
end

function run_example()
    data = prepare()
    tile_i = isnothing(data.D_E) ? min(8, size(data.K, 1)) : min(8, data.D_E)
    result = run(data; tile_d=size(data.Q, 1), tile_i)
    verify(data, result)

    dO = CUDA.rand(eltype(data.Q), size(data.Q)...)
    dQ = CUDA.zeros(eltype(data.Q), size(data.Q)...)
    dK = CUDA.zeros(eltype(data.K), size(data.K)...)
    dU = CUDA.zeros(eltype(data.U), size(data.U)...)
    dV = CUDA.zeros(eltype(data.V), size(data.V)...)
    dR = isnothing(data.R) ? nothing : CUDA.zeros(eltype(data.R), size(data.R)...)
    ∇sparse_mhffn!(dQ, dK, dU, dV, dO, data.Q, data.K, data.U, data.V, data.dispatch;
        R=data.R, dR, D_E=data.D_E, tile_d=size(data.Q, 1), tile_i)
    CUDA.synchronize()

    ref_dQ, ref_dR = reference_sparse_mhffn_dq(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=isnothing(data.R) ? nothing : Array(data.R), D_E=data.D_E)
    ref_dK, ref_dU, ref_dV = reference_sparse_mhffn_dkuv(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=isnothing(data.R) ? nothing : Array(data.R), D_E=data.D_E)
    @assert Array(dQ) ≈ ref_dQ rtol=1e-3 atol=1e-3
    @assert Array(dK) ≈ ref_dK rtol=1e-3 atol=1e-3
    @assert Array(dU) ≈ ref_dU rtol=1e-3 atol=1e-3
    @assert Array(dV) ≈ ref_dV rtol=1e-3 atol=1e-3
    if !isnothing(dR)
        @assert Array(dR) ≈ ref_dR rtol=1e-3 atol=1e-3
    end
    return nothing
end

function main()
    run_example()
    return nothing
end

end

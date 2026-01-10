# Mixture of Experts example - Julia port of cuTile Python's MoE.py example
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

const ConstInt = ct.Constant{Int}
const ConstBool = ct.Constant{Bool}

function fused_moe_kernel(
    A, # input tokens: K, batch
    B, # expert weights: K, N, num_experts
    C, # output array: N, topk * num_tokens
    topk_weights, # router weights: (topk * num_tokens) 
    sorted_token_ids, # token indices sorted by expert assignment, replicated topk times, padded to align with TILE_M
    sorted_expert_ids; # expert index for each TILE_M, sorted
    num_token_replicas::Integer, # replication factor applied to each token row in A (topk or 1)
    mul_routed_weight::ConstBool, # whether to multiply output by router weights
    TILE_M::ConstInt,
    TILE_N::ConstInt,
    TILE_K::ConstInt,
)
    M = size(sorted_token_ids, 1)
    N = size(B, 2)
    K = size(B, 1)

    GROUP_SIZE_M = 8
    bid_m, bid_n = swizzle_2d(M, N, TILE_M[], TILE_N[], GROUP_SIZE_M)

    zero_pad = ct.PaddingMode.Zero

    token_id_indices = bid_m * TILE_M[] .+ ct.arange(TILE_M[], Int32)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)

    a_row_indices = cld.(token_ids, num_token_replicas) # TILE_M

    expert_id = ct.load(sorted_expert_ids, (bid_m,), ())

    accumulator = ct.zeros((TILE_N[], TILE_M[]), Float32)
    k = Int32(1)
    while k <= cld(K, TILE_K[])
        a_col_indices = k * TILE_K[] .+ ct.arange(TILE_K[], Int32) # TILE_K
        a = ct.gather(A, (ct.reshape(a_col_indices, (:, TILE_K[])), ct.reshape(a_row_indices, (1, TILE_M[]))))
        
        b = ct.load(B, (bid_n, k, expert_id), (TILE_N[], TILE_K[], 1), padding_mode=zero_pad)
        b = ct.transpose(ct.reshape(b, (TILE_N[], TILE_K[])))

        accumulator = ct.muladd(a, b, accumulator)
        k += Int32(1)
    end

    if mul_routed_weight[]
        moe_weight = ct.gather(topk_weights, token_ids)
        accumulator = accumulator .* ct.reshape(moe_weight, (1, TILE_M[]))
    end

    c_col_indices = bid_n * TILE_N[] .+ ct.arange(TILE_N[], Int32)
    accumulator = ct.astype(accumulator, eltype(C))
    ct.scatter(C, (ct.reshape(c_col_indices, (TILE[N], 1)), ct.reshape(token_ids, (1, TILE_M[]))), accumulator)
    return
end

function silu_and_mul_kernel(A, B, C, TILE_N::ConstInt)
    bid_m = ct.bid(1)
    ta = ct.astype(ct.load(A, (1, bid_m), (TILE_N[], 1)), Float32)
    tb = ct.astype(ct.load(B, (1, bid_m), (TILE_N[], 1)), Float32)

    denom = 1f0 .+ ct.exp(-ta) # flush to zero?
    sigmoid_ta = 1f0 ./ denom # flush to zero? rounding mode?

    silu_ta = ct.mul(ta, sigmoid_ta) # flush to zero?
    tc = silu_ta .* tb # flush to zero?

    ct.store(C, (1, bid_m), ct.astype(tc, eltype(C)))
    return
end

function moe_align_tile_size(   
    topk_ids, tile_m::Int, num_experts::Int
)
    topk, num_tokens = size(topk_ids)
    total_tokens = num_tokens * topk

    # flatten expert ids and sort by experts
    flat_expert_ids = reshape(topk_ids, :)
    sorted_token_indices = sortperm(flat_expert_ids)

    # may need to move to cpu
    # FIXME: torch.bincount equivalent?
    expert_token_counts = bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = cld.(expert_token_counts - 1 + tile_m, tile_m) # double check for julia
    total_blocks = sum(expert_block_counts)

    sorted_token_ids = similar(topk_ids, Int32, total_blocks * tile_m) .= total_tokens
    sorted_expert_ids = similar(topk_ids, Int32, total_blocks) .= 0

    # FIXME: rest of function. might need to be a cuda kernel
end

@inline function swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    group_id = fld(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid, group_size_m)
    bid_n = fld(rem(bid, num_bid_in_group), group_size_m)
    return bid_m, bid_n
end

function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)
end

zeros_like(args...) = fill!(similar(args...), false)

function cutile_moe(
    hidden_states::AbstractArray{T,2}, # hidden_size, num_tokens
    w1::AbstractArray{T,3}, # hidden_size, intermediate_size * 2, num_experts
    w2::AbstractArray{T,3}, # intermediate_size, hidden_size, num_experts
    topk_weights::AbstractArray{T,2}, # topk, num_tokens
    topk_ids::AbstractArray{T,2}, # topk, num_tokens
    tile_m::Int,
    tile_n::Int,
    tile_k::Int,
) where T
    out_T = eltype(hidden_states)
    hidden_size, num_tokens = size(hidden_states)
    intermediate_size, _, num_experts = size(w2)
    topk, _ = size(topk_ids)

    if size(w1, 2) != intermediate_size * 2
        throw(ArgumentError("w1 must have 2 * intermediate_size rows (gate + up projection)"))
    end
    
    intermediate_cache1 = zeros_like(w1, out_T, intermediate_size * 2, topk, num_tokens)
    intermediate_cache2 = zeros_like(w1, out_T, intermediate_size, topk * num_tokens)
    intermediate_cache3 = zeros_like(w1, out_T, hidden_size, topk, num_tokens)

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size(topk_ids, tile_m, num_experts)

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids;
        mul_routed_weight=false,
        num_token_replicas=topk,
        tile_m, tile_n, tile_k
    )

    invoke_silu_and_mul_kernel(
        reshape(intermediate_cache1, size(intermediate_cache1, 1), :),
        intermediate_cache2
    )

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids;
        mul_routed_weight=true,
        num_token_replicas=1,
        tile_m, tile_n, tile_k
    )

    return sum(intermediate_cache3, dims=2)
end


function julia_moe(
    hidden_states::AbstractArray{T,2}, # hidden_size, num_tokens
    w1::AbstractArray{T,3}, # hidden_size, intermediate_size * 2, num_experts
    w2::AbstractArray{T,3}, # intermediate_size, hidden_size, num_experts
    topk_weights::AbstractArray{T,2}, # topk, num_tokens
    topk_ids::AbstractArray{T,2}, # topk, num_tokens
) where T
    intermediate_size, _, num_experts = size(w2)
    gate_proj, up_proj = view(w1, :, 1:intermediate_size, :), view(w1, :, intermediate_size+1:2*intermediate_size, :)
    down_proj = w2

    # FIXME: finish implementation
end


# --- Helper utilities ---
function invoke_fused_moe_kernel(
    A::AbstractArray{T,2},
    B::AbstractArray{T,3},
    C::AbstractArray{T,3},
    topk_weights::AbstractArray{T,2},
    sorted_token_ids::AbstractArray{T,2},
    sorted_expert_ids::AbstractArray{T,2};
    mul_routed_weight::Bool,
    num_token_replicas::Int,
    tile_m::Int,
    tile_n::Int,
    tile_k::Int
) where T
    m = size(sorted_token_ids)[end] # is this 1D?
    n = size(B, 1)

    grid = (cld(m, tile_m) * cld(n, tile_n),)
    topk_weights = reshape(topk_weights, :)
    C = reshape(C, size(C, 1), :)

    ct.launch(fused_moe_kernel, grid,
        A, B, C,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        num_token_replicas,
        ct.Constant(mul_routed_weight),
        ct.Constant(tile_m),
        ct.Constant(tile_n),
        ct.Constant(tile_k)
    )
end

function invoke_silu_and_mul_kernel(
    AB::AbstractArray{T,2},
    C::AbstractArray{T,2};
)
    intermediate_size = size(AB, 1) ÷ 2

    grid = (size(AB, 2),)
    A, B = view(AB, 1:intermediate_size, :), view(AB, intermediate_size+1:2*intermediate_size, :)

    ct.launch(silu_and_mul_kernel, grid,
        A, B, C,
        next_power_of_2(intermediate_size)
    )
end

function bincount(x::CuArray{<:Integer}; minlength::Int=0)
    n = length(x)
    bins = max(maximum(a -> a + 1, x; init=0), minlength)
    out = CUDA.zeros(Int32, bins)

    function bincount_kernel!()
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        for idx in i:stride:n
            CUDA.@atomic out[x[idx] + 1] += 1 
        end
        return
    end

    threads = 256
    blocks = min(cld(n, threads), 65_535)
    @cuda threads=threads blocks=blocks bincount_kernel!()

    return out
end

function next_power_of_2(n::Int)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
end

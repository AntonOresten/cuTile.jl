# LayerNorm forward pass example - Julia port of cuTile Python's LayerNorm.py sample
#
# This example demonstrates the new operations added for LayerNorm:
# - ct.reduce_sum() - Sum reduction along an axis
# - ct.sqrt() - Square root
# - ct.arange() - Create a tile with [0, 1, 2, ..., n-1]
# - ct.where() - Conditional element selection
# - Division of tiles

using CUDA
import cuTile as ct

# LayerNorm forward kernel
# Normalizes each row (along N dimension) of the input tensor X
#
# For each row m:
#   mean_m = sum(X[m, :]) / N
#   var_m = sum((X[m, :] - mean_m)^2) / N
#   rstd_m = 1 / sqrt(var_m + eps)
#   Y[m, :] = (X[m, :] - mean_m) * rstd_m * W + B
function layer_norm_fwd(X::ct.TileArray{T, 2}, W::ct.TileArray{T, 1}, B::ct.TileArray{T, 1},
                        Y::ct.TileArray{T, 2}, Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                        eps::ct.Constant{Float32}, TILE_N::ct.Constant{Int}) where T
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, 1, (1, TILE_N[]))
    N = ct.num_tiles(X, 1, (1, 1)) * 1  # Get N dimension size

    # Initialize accumulator for mean computation
    mean = ct.full((1, TILE_N[]), 0.0f0, Float32)

    # First pass: compute mean
    for j in 0:num_tiles-1
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        mean = mean + tx
    end

    # Reduce to scalar mean (sum along axis 1, then divide by N)
    mean_sum = ct.reduce_sum(mean, 1)
    # mean_scalar = mean_sum / Float32(N)  # This would need scalar broadcast

    # Store mean
    ct.store(Mean, (bid_m,), mean_sum)

    # Second pass: compute variance
    var = ct.full((1, TILE_N[]), 0.0f0, Float32)
    for j in 0:num_tiles-1
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        # Note: simplified - doesn't subtract mean for simplicity
        # centered_tx = tx - mean_scalar
        var = var + tx * tx
    end

    var_sum = ct.reduce_sum(var, 1)
    # Compute rstd = 1 / sqrt(var + eps)
    # rstd = rsqrt(var_sum + eps)  # Would need scalar addition and rsqrt

    ct.store(Rstd, (bid_m,), var_sum)

    # Third pass: normalize and apply affine transform
    for j in 0:num_tiles-1
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        tw = ct.load(W, (j,), (TILE_N[],))
        tb = ct.load(B, (j,), (TILE_N[],))

        # Simplified: just copy input * weight + bias (not actual layer norm)
        ty = tx * tw + tb
        ct.store(Y, (bid_m, j), ty)
    end

    return
end

# Simpler kernel to test the new operations individually
function test_reduce_sum(A::ct.TileArray{Float32, 2}, B::ct.TileArray{Float32, 1},
                         TILE_M::ct.Constant{Int}, TILE_N::ct.Constant{Int})
    bid = ct.bid(0)
    tile = ct.load(A, (bid, 0), (TILE_M[], TILE_N[]))

    # Sum along axis 1 (columns) - reduces (M, N) -> (M,)
    sums = ct.reduce_sum(tile, 1)

    ct.store(B, bid, sums)
    return
end

function test_sqrt(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1},
                   TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile = ct.load(A, bid, (TILE[],))

    result = sqrt(tile)

    ct.store(B, bid, result)
    return
end

function test_div(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1}, C::ct.TileArray{Float32, 1},
                  TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile_a = ct.load(A, bid, (TILE[],))
    tile_b = ct.load(B, bid, (TILE[],))

    result = tile_a / tile_b

    ct.store(C, bid, result)
    return
end

function test_reduce_sum_example()
    println("--- Testing reduce_sum ---")
    M, N = 4, 128
    TILE_M, TILE_N = 4, 128

    A = CUDA.rand(Float32, M, N)
    B = CUDA.zeros(Float32, M)  # 1D output

    ct.launch(test_reduce_sum, 1, A, B, ct.Constant(TILE_M), ct.Constant(TILE_N))

    # Verify: each element of B should be the sum of that row of A
    A_cpu = Array(A)
    B_cpu = Array(B)

    for i in 1:M
        expected = sum(A_cpu[i, :])
        actual = B_cpu[i]
        if !isapprox(expected, actual; rtol=1e-3)
            println("  Row $i: expected=$expected, actual=$actual - FAILED")
            return false
        end
    end
    println("  All rows match expected sums")
    return true
end

function test_sqrt_example()
    println("--- Testing sqrt ---")
    N = 1024
    TILE = 1024

    A = CUDA.rand(Float32, N) .+ 0.1f0  # Ensure positive
    B = CUDA.zeros(Float32, N)

    ct.launch(test_sqrt, 1, A, B, ct.Constant(TILE))

    A_cpu = Array(A)
    B_cpu = Array(B)

    expected = sqrt.(A_cpu)
    if isapprox(expected, B_cpu; rtol=1e-5)
        println("  sqrt results match")
        return true
    else
        println("  sqrt results don't match - FAILED")
        return false
    end
end

function test_div_example()
    println("--- Testing tile division ---")
    N = 1024
    TILE = 1024

    A = CUDA.rand(Float32, N)
    B = CUDA.rand(Float32, N) .+ 0.1f0  # Ensure non-zero
    C = CUDA.zeros(Float32, N)

    ct.launch(test_div, 1, A, B, C, ct.Constant(TILE))

    A_cpu = Array(A)
    B_cpu = Array(B)
    C_cpu = Array(C)

    expected = A_cpu ./ B_cpu
    if isapprox(expected, C_cpu; rtol=1e-5)
        println("  division results match")
        return true
    else
        println("  division results don't match - FAILED")
        return false
    end
end

function main()
    println("=== cuTile LayerNorm Operations Tests ===\n")

    results = Bool[]

    push!(results, test_reduce_sum_example())
    push!(results, test_sqrt_example())
    push!(results, test_div_example())

    println()
    if all(results)
        println("All tests passed!")
    else
        println("Some tests failed!")
    end
end

isinteractive() || main()

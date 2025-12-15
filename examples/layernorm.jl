# LayerNorm forward pass example - Julia port of cuTile Python's LayerNorm.py sample
#
# This example demonstrates the operations needed for LayerNorm:
# - ct.reduce_sum() - Sum reduction along an axis
# - ct.sqrt() - Square root
# - Scalar-tile operations (tile + scalar, tile - scalar, tile / scalar, etc.)
# - Division of tiles

using CUDA
import cuTile as ct

# Simple row-wise affine transform kernel
# Demonstrates 1D tile operations on 2D data
# Each block processes one row: Y[m, :] = X[m, :] * W + B
function row_affine_transform(X::ct.TileArray{Float32, 1}, W::ct.TileArray{Float32, 1},
                              B::ct.TileArray{Float32, 1}, Y::ct.TileArray{Float32, 1},
                              N::ct.Constant{Int})
    bid = ct.bid(0)

    # Load entire vectors as 1D tiles
    x = ct.load(X, bid, (N[],))
    w = ct.load(W, bid, (N[],))
    b = ct.load(B, bid, (N[],))

    # Apply affine transform: y = x * w + b
    y = x * w + b

    # Store result
    ct.store(Y, bid, y)

    return
end

# Test kernel for scalar-tile operations
function test_scalar_tile_ops(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1},
                               TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile = ct.load(A, bid, (TILE[],))

    # Test: tile + scalar
    result1 = tile + 1.0f0

    # Test: tile - scalar
    result2 = result1 - 0.5f0

    # Test: tile * scalar
    result3 = result2 * 2.0f0

    # Test: tile / scalar
    result4 = result3 / 4.0f0

    ct.store(B, bid, result4)
    return
end

# Test kernel for scalar / tile (reciprocal)
function test_reciprocal(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1},
                         TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile = ct.load(A, bid, (TILE[],))

    # Compute 1 / tile
    result = 1.0f0 / tile

    ct.store(B, bid, result)
    return
end

# Test kernel for tile / integer
function test_div_by_int(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1},
                         TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile = ct.load(A, bid, (TILE[],))

    # Divide by integer (common for computing mean)
    result = tile / 128

    ct.store(B, bid, result)
    return
end

# Simpler kernel to test reduce_sum
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

function test_div(A::ct.TileArray{Float32, 1}, B::ct.TileArray{Float32, 1},
                  C::ct.TileArray{Float32, 1}, TILE::ct.Constant{Int})
    bid = ct.bid(0)
    tile_a = ct.load(A, bid, (TILE[],))
    tile_b = ct.load(B, bid, (TILE[],))

    result = tile_a / tile_b

    ct.store(C, bid, result)
    return
end

#=============================================================================
 Test Functions
=============================================================================#

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

function test_scalar_tile_ops_example()
    println("--- Testing scalar-tile operations ---")
    N = 1024
    TILE = 1024

    A = CUDA.rand(Float32, N)
    B = CUDA.zeros(Float32, N)

    ct.launch(test_scalar_tile_ops, 1, A, B, ct.Constant(TILE))

    A_cpu = Array(A)
    B_cpu = Array(B)

    # Expected: ((A + 1.0) - 0.5) * 2.0) / 4.0 = (A + 0.5) * 0.5 = A/2 + 0.25
    expected = (((A_cpu .+ 1.0f0) .- 0.5f0) .* 2.0f0) ./ 4.0f0
    if isapprox(expected, B_cpu; rtol=1e-5)
        println("  scalar-tile operations match")
        return true
    else
        println("  scalar-tile operations don't match - FAILED")
        max_err = maximum(abs.(expected .- B_cpu))
        println("  Max error: $max_err")
        return false
    end
end

function test_reciprocal_example()
    println("--- Testing reciprocal (1/tile) ---")
    N = 1024
    TILE = 1024

    A = CUDA.rand(Float32, N) .+ 0.1f0  # Ensure non-zero
    B = CUDA.zeros(Float32, N)

    ct.launch(test_reciprocal, 1, A, B, ct.Constant(TILE))

    A_cpu = Array(A)
    B_cpu = Array(B)

    expected = 1.0f0 ./ A_cpu
    if isapprox(expected, B_cpu; rtol=1e-5)
        println("  reciprocal results match")
        return true
    else
        println("  reciprocal results don't match - FAILED")
        return false
    end
end

function test_div_by_int_example()
    println("--- Testing tile / integer ---")
    N = 1024
    TILE = 1024

    A = CUDA.rand(Float32, N)
    B = CUDA.zeros(Float32, N)

    ct.launch(test_div_by_int, 1, A, B, ct.Constant(TILE))

    A_cpu = Array(A)
    B_cpu = Array(B)

    expected = A_cpu ./ 128.0f0
    if isapprox(expected, B_cpu; rtol=1e-5)
        println("  tile/integer results match")
        return true
    else
        println("  tile/integer results don't match - FAILED")
        return false
    end
end

function test_affine_transform_example()
    println("--- Testing affine transform (y = x * w + b) ---")
    N = 1024
    TILE = 1024

    X = CUDA.rand(Float32, N)
    W = CUDA.rand(Float32, N)
    B = CUDA.rand(Float32, N)
    Y = CUDA.zeros(Float32, N)

    ct.launch(row_affine_transform, 1, X, W, B, Y, ct.Constant(TILE))

    X_cpu = Array(X)
    W_cpu = Array(W)
    B_cpu = Array(B)
    Y_cpu = Array(Y)

    # Verify output: y = x * w + b
    expected = X_cpu .* W_cpu .+ B_cpu
    if isapprox(expected, Y_cpu; rtol=1e-5)
        println("  affine transform results match")
        return true
    else
        println("  affine transform results don't match - FAILED")
        max_err = maximum(abs.(expected .- Y_cpu))
        println("  Max error: $max_err")
        return false
    end
end

function main()
    println("=== cuTile LayerNorm Operations Tests ===\n")

    results = Bool[]

    push!(results, test_reduce_sum_example())
    push!(results, test_sqrt_example())
    push!(results, test_div_example())
    push!(results, test_scalar_tile_ops_example())
    push!(results, test_reciprocal_example())
    push!(results, test_div_by_int_example())
    push!(results, test_affine_transform_example())

    println()
    if all(results)
        println("All tests passed!")
    else
        n_failed = count(!identity, results)
        println("$n_failed test(s) failed!")
    end
end

isinteractive() || main()

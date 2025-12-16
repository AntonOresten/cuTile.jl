# LayerNorm example - Julia port of cuTile Python's LayerNorm.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

const ConstInt = ct.Constant{Int}

#=============================================================================
 LayerNorm Forward Kernel

 Forward pass: computes mean/var, normalizes input, and applies affine transform.

 Args:
     X: Input tensor (M, N).
     W: Weight tensor (N,).
     B: Bias tensor (N,).
     Y: Output tensor (M, N).
     Mean: Output mean tensor (M,).
     Rstd: Output reciprocal standard deviation tensor (M,).
     eps: Epsilon for numerical stability.
     TILE_N: Tile size along N dimension.
=============================================================================#
function layer_norm_fwd(X::ct.TileArray{Float32, 2}, W::ct.TileArray{Float32, 1},
                        B::ct.TileArray{Float32, 1}, Y::ct.TileArray{Float32, 2},
                        Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                        eps::ct.Constant{Float32}, TILE_N::ConstInt)
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, 1, (1, TILE_N[]))
    N = X.sizes[2]

    # Compute mean
    mean = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(0)
    while j < num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        mean = mean .+ tx
        j += Int32(1)
    end
    mean = ct.reduce_sum(mean, 1) / N
    ct.store(Mean, bid_m, mean)

    # Compute variance
    var = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(0)
    while j < num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        mask = ct.broadcast_to((j * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)) .< N, (1, TILE_N[]))
        centered_tx = ct.where(mask, tx .- mean, ct.full((1, TILE_N[]), 0.0f0, Float32))
        var = var .+ (centered_tx .^ 2.0f0)
        j += Int32(1)
    end
    var = ct.reduce_sum(var, 1) / N
    rstd = 1.0f0 / sqrt(var .+ eps[])
    ct.store(Rstd, bid_m, rstd)

    # Normalize and apply affine transformation
    j = Int32(0)
    while j < num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]))
        tw = ct.load(W, j, (TILE_N[],))
        tb = ct.load(B, j, (TILE_N[],))
        ty = (tx .- mean) .* rstd
        ty = ty .* tw .+ tb
        ct.store(Y, (bid_m, j), ty)
        j += Int32(1)
    end

    return
end

#=============================================================================
 Test / Validation
=============================================================================#

function main()
    println("--- Running cuTile LayerNorm Forward Sample ---")

    M, N = 1024, 2048
    TILE_N = 1024
    eps = 1f-5

    println("Input shape: ($M, $N), dtype: Float32, eps: $eps")

    # Input data
    X = -2.3f0 .+ 0.5f0 .* CUDA.rand(Float32, M, N)
    W = CUDA.randn(Float32, N)
    B = CUDA.randn(Float32, N)

    # Output buffers
    Y = CUDA.zeros(Float32, M, N)
    Mean = CUDA.zeros(Float32, M)
    Rstd = CUDA.zeros(Float32, M)

    println("\n--- Executing cuTile LayerNorm Forward ---")
    ct.launch(layer_norm_fwd, M, X, W, B, Y, Mean, Rstd,
              ct.Constant(eps), ct.Constant(TILE_N))

    println("\n--- Running correctness check against reference ---")

    # Compute expected values on CPU
    X_cpu = Array(X)
    W_cpu = Array(W)
    B_cpu = Array(B)

    expected_mean = vec(sum(X_cpu, dims=2) ./ N)
    expected_var = vec(sum((X_cpu .- expected_mean) .^ 2, dims=2) ./ N)
    expected_rstd = 1.0f0 ./ sqrt.(expected_var .+ eps)
    normalized = (X_cpu .- expected_mean) .* expected_rstd
    expected_Y = normalized .* W_cpu' .+ B_cpu'

    # Verify results
    Mean_cpu = Array(Mean)
    Rstd_cpu = Array(Rstd)
    Y_cpu = Array(Y)

    atol, rtol = 1f-2, 1f-2
    mean_ok = isapprox(expected_mean, Mean_cpu; rtol, atol)
    rstd_ok = isapprox(expected_rstd, Rstd_cpu; rtol, atol)
    y_ok = isapprox(expected_Y, Y_cpu; rtol, atol)

    if mean_ok && rstd_ok && y_ok
        println("Correctness check passed")
    else
        println("Correctness check FAILED:")
        mean_ok || println("  Mean max error: $(maximum(abs.(expected_mean .- Mean_cpu)))")
        rstd_ok || println("  Rstd max error: $(maximum(abs.(expected_rstd .- Rstd_cpu)))")
        y_ok || println("  Y max error: $(maximum(abs.(expected_Y .- Y_cpu)))")
    end

    println("\n--- cuTile LayerNorm Sample complete ---")
end

isinteractive() || main()

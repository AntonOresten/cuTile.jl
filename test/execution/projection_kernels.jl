using CUDA

include(joinpath(@__DIR__, "..", "..", "examples", "projection_kernels_impl.jl"))

const pk = ProjectionKernelsExample

@testset "projection dispatch" begin
    dispatch = pk.build_projection_dispatch([1, 3, 1, 2], 3, 4)
    @test Array(dispatch.sorted_ids) == Int32[
        1, 3, 5, 7,
        9, 11, 0, 0,
        4, 8, 12, 0,
        2, 6, 10, 0,
    ]
    @test Array(dispatch.tok_cols) == Int32[
        1, 1, 2, 2,
        3, 3, 0, 0,
        1, 2, 3, 0,
        1, 2, 3, 0,
    ]
    @test Array(dispatch.head_block_starts) == Int32[1, 3, 4, 5]
    @test Array(dispatch.block_heads) == Int32[1, 1, 2, 3]

    token_heads = Int32[
        1 2 1;
        3 1 2;
        1 3 3;
        2 2 1;
    ]
    routed = pk.build_projection_dispatch(token_heads, 4)
    routed_ids = Array(routed.sorted_ids)
    routed_heads = Int32[
        flat_id == 0 ? 0 : token_heads[mod1(flat_id, size(token_heads, 1)), fld1(flat_id, size(token_heads, 1))]
        for flat_id in routed_ids
    ]
    nonzero_heads = filter(!=(Int32(0)), routed_heads)
    @test nonzero_heads == sort(nonzero_heads)
    @test Array(routed.token_heads) == token_heads
end

@testset "projection kernels" begin
    kernel_tol = (rtol=1e-3, atol=1e-3)
    token_heads = Int32[
        1 2 1 3 2;
        3 1 2 1 3;
        1 3 3 2 1;
        2 2 1 3 2;
    ]
    problem = pk.make_example_problem(; d_model=8, d_head=8, L=5, token_heads, tile_m=4)
    W_in = problem.W_in
    W_out = problem.W_out
    X = problem.X
    dispatch = problem.dispatch
    token_heads_cpu = Array(problem.token_heads)
    kL = size(token_heads_cpu, 1) * size(X, 2)

    @testset "in proj fwd / out proj bwd dy" begin
        Y = CUDA.zeros(Float32, size(W_in, 1), kL)
        pk.launch_in_proj_fwd!(Y, W_in, X, dispatch; tile_e=8, tile_d=8)
        ref_Y = pk.reference_in_proj_fwd(Array(W_in), Array(X), token_heads_cpu)
        @test Array(Y) ≈ ref_Y rtol=kernel_tol.rtol atol=kernel_tol.atol

        dZ = CUDA.rand(Float32, size(W_out, 1), size(X, 2))
        dY = CUDA.zeros(Float32, size(W_in, 1), kL)
        pk.launch_out_proj_bwd_dy!(dY, W_out, dZ, dispatch; tile_e=8, tile_d=8)
        ref_dY = pk.reference_out_proj_bwd_dy(Array(W_out), Array(dZ), token_heads_cpu)
        @test Array(dY) ≈ ref_dY rtol=kernel_tol.rtol atol=kernel_tol.atol
    end

    @testset "fixed-routing contiguous in proj" begin
        fixed_token_heads = repeat(reshape(Int32[1, 2, 3, 4], :, 1), 1, 5)
        fixed_problem = pk.make_example_problem(; d_model=8, d_head=8, L=5, token_heads=fixed_token_heads, tile_m=4)
        fixed_Y_gather = CUDA.zeros(Float32, size(fixed_problem.W_in, 1), size(fixed_problem.token_heads, 1) * size(fixed_problem.X, 2))
        fixed_Y_contig = similar(fixed_Y_gather)
        pk.launch_in_proj_fwd!(fixed_Y_gather, fixed_problem.W_in, fixed_problem.X, fixed_problem.dispatch; tile_e=8, tile_d=8)
        pk.launch_in_proj_fwd_contiguous!(fixed_Y_contig, fixed_problem.W_in, fixed_problem.X, fixed_problem.dispatch; tile_e=8, tile_d=8)
        ref_fixed_Y = pk.reference_in_proj_fwd(Array(fixed_problem.W_in), Array(fixed_problem.X), Array(fixed_problem.token_heads))
        @test Array(fixed_Y_gather) ≈ ref_fixed_Y rtol=kernel_tol.rtol atol=kernel_tol.atol
        @test Array(fixed_Y_contig) ≈ ref_fixed_Y rtol=kernel_tol.rtol atol=kernel_tol.atol
    end

    @testset "out proj fwd / in proj bwd dx" begin
        Y = CUDA.rand(Float32, size(W_in, 1), kL)
        Z = CUDA.zeros(Float32, size(W_out, 1), size(X, 2))
        pk.launch_out_proj_fwd!(Z, W_out, Y, dispatch; tile_d=8, tile_l=4, tile_e=8)
        ref_Z = pk.reference_out_proj_fwd(Array(W_out), Array(Y), token_heads_cpu)
        @test Array(Z) ≈ ref_Z rtol=1e-4 atol=1e-4

        dY = CUDA.rand(Float32, size(W_in, 1), kL)
        dX = CUDA.zeros(Float32, size(X)...)
        pk.launch_in_proj_bwd_dx!(dX, W_in, dY, dispatch; tile_d=8, tile_l=4, tile_e=8)
        ref_dX = pk.reference_in_proj_bwd_dx(Array(W_in), Array(dY), token_heads_cpu)
        @test Array(dX) ≈ ref_dX rtol=1e-4 atol=1e-4
    end

    @testset "weight gradients" begin
        Y = CUDA.rand(Float32, size(W_in, 1), kL)
        dY = CUDA.rand(Float32, size(W_in, 1), kL)
        dZ = CUDA.rand(Float32, size(W_out, 1), size(X, 2))

        dW_in = CUDA.zeros(Float32, size(W_in)...)
        dW_out = CUDA.zeros(Float32, size(W_out)...)
        pk.launch_in_proj_bwd_dw!(dW_in, dY, X, dispatch; tile_e=8, tile_d=8, num_m_splits=3)
        pk.launch_out_proj_bwd_dw!(dW_out, dZ, Y, dispatch; tile_d=8, tile_e=4, num_m_splits=3)

        ref_dW_in = pk.reference_in_proj_bwd_dw(Array(dY), Array(X), token_heads_cpu, Int(dispatch.H))
        ref_dW_out = pk.reference_out_proj_bwd_dw(Array(dZ), Array(Y), token_heads_cpu, Int(dispatch.H))
        @test Array(dW_in) ≈ ref_dW_in rtol=1e-4 atol=1e-4
        @test Array(dW_out) ≈ ref_dW_out rtol=1e-4 atol=1e-4
    end

    @testset "end to end gradient sanity" begin
        Y = CUDA.zeros(Float32, size(W_in, 1), kL)
        pk.launch_in_proj_fwd!(Y, W_in, X, dispatch; tile_e=8, tile_d=8)

        dZ = CUDA.rand(Float32, size(W_out, 1), size(X, 2))
        Z = CUDA.zeros(Float32, size(W_out, 1), size(X, 2))
        pk.launch_out_proj_fwd!(Z, W_out, Y, dispatch; tile_d=8, tile_l=4, tile_e=8)

        dY = CUDA.zeros(Float32, size(W_in, 1), kL)
        dX = CUDA.zeros(Float32, size(X)...)
        dW_in = CUDA.zeros(Float32, size(W_in)...)
        dW_out = CUDA.zeros(Float32, size(W_out)...)

        pk.launch_out_proj_bwd_dy!(dY, W_out, dZ, dispatch; tile_e=8, tile_d=8)
        pk.launch_in_proj_bwd_dx!(dX, W_in, dY, dispatch; tile_d=8, tile_l=4, tile_e=8)
        pk.launch_in_proj_bwd_dw!(dW_in, dY, X, dispatch; tile_e=8, tile_d=8, num_m_splits=2)
        pk.launch_out_proj_bwd_dw!(dW_out, dZ, Y, dispatch; tile_d=8, tile_e=4, num_m_splits=2)
        CUDA.synchronize()

        W_in_cpu = Array(W_in)
        W_out_cpu = Array(W_out)
        X_cpu = Array(X)
        dZ_cpu = Array(dZ)
        Y_cpu = pk.reference_in_proj_fwd(W_in_cpu, X_cpu, token_heads_cpu)
        dY_cpu = pk.reference_out_proj_bwd_dy(W_out_cpu, dZ_cpu, token_heads_cpu)
        dX_cpu = pk.reference_in_proj_bwd_dx(W_in_cpu, dY_cpu, token_heads_cpu)
        dW_in_cpu = pk.reference_in_proj_bwd_dw(dY_cpu, X_cpu, token_heads_cpu, Int(dispatch.H))
        dW_out_cpu = pk.reference_out_proj_bwd_dw(dZ_cpu, Y_cpu, token_heads_cpu, Int(dispatch.H))

        @test Array(dY) ≈ dY_cpu rtol=kernel_tol.rtol atol=kernel_tol.atol
        @test Array(dX) ≈ dX_cpu rtol=kernel_tol.rtol atol=kernel_tol.atol
        @test Array(dW_in) ≈ dW_in_cpu rtol=1e-4 atol=1e-4
        @test Array(dW_out) ≈ dW_out_cpu rtol=1e-4 atol=1e-4

        function cpu_loss(Wi, Wo, Xin)
            Yin = pk.reference_in_proj_fwd(Wi, Xin, token_heads_cpu)
            Zout = pk.reference_out_proj_fwd(Wo, Yin, token_heads_cpu)
            return sum(Zout .* dZ_cpu)
        end

        eps = 1f-3
        Wi_plus = copy(W_in_cpu)
        Wi_minus = copy(W_in_cpu)
        Wi_plus[1, 1, 1] += eps
        Wi_minus[1, 1, 1] -= eps
        fd_wi = (cpu_loss(Wi_plus, W_out_cpu, X_cpu) - cpu_loss(Wi_minus, W_out_cpu, X_cpu)) / (2f0 * eps)
        @test fd_wi ≈ Array(dW_in)[1, 1, 1] rtol=5e-2 atol=5e-2

        Wo_plus = copy(W_out_cpu)
        Wo_minus = copy(W_out_cpu)
        Wo_plus[1, 1, 1] += eps
        Wo_minus[1, 1, 1] -= eps
        fd_wo = (cpu_loss(W_in_cpu, Wo_plus, X_cpu) - cpu_loss(W_in_cpu, Wo_minus, X_cpu)) / (2f0 * eps)
        @test fd_wo ≈ Array(dW_out)[1, 1, 1] rtol=5e-2 atol=5e-2

        X_plus = copy(X_cpu)
        X_minus = copy(X_cpu)
        X_plus[1, 1] += eps
        X_minus[1, 1] -= eps
        fd_x = (cpu_loss(W_in_cpu, W_out_cpu, X_plus) - cpu_loss(W_in_cpu, W_out_cpu, X_minus)) / (2f0 * eps)
        @test fd_x ≈ Array(dX)[1, 1] rtol=5e-2 atol=5e-2
    end

    @testset "dense fixed-head baseline" begin
        fixed_token_heads = repeat(reshape(Int32[1, 2, 3, 4], :, 1), 1, 5)
        fixed_problem = pk.make_example_problem(; d_model=8, d_head=8, L=5, token_heads=fixed_token_heads, tile_m=4)
        dense = pk.make_dense_fixed_head_baseline(fixed_problem.W_in, fixed_problem.W_out, fixed_problem.X, fixed_problem.token_heads)
        pk.dense_fixed_heads_fwd!(dense; reduce_tile_d=8, reduce_tile_l=4)

        Y_ref = pk.reference_in_proj_fwd(Array(fixed_problem.W_in), Array(fixed_problem.X), Array(fixed_problem.token_heads))
        Z_ref = pk.reference_out_proj_fwd(Array(fixed_problem.W_out), Y_ref, Array(fixed_problem.token_heads))
        @test Array(dense.Z) ≈ Z_ref rtol=1e-4 atol=1e-4
    end

    @testset "packed fixed-head baseline" begin
        fixed_token_heads = repeat(reshape(Int32[1, 2, 3, 4], :, 1), 1, 5)
        fixed_problem = pk.make_example_problem(; d_model=8, d_head=8, L=5, token_heads=fixed_token_heads, tile_m=4)
        packed = pk.make_packed_fixed_head_baseline(fixed_problem.W_in, fixed_problem.W_out, fixed_problem.X, fixed_problem.token_heads)
        Y_ref = pk.reference_in_proj_fwd(Array(fixed_problem.W_in), Array(fixed_problem.X), Array(fixed_problem.token_heads))
        Z_ref = pk.reference_out_proj_fwd(Array(fixed_problem.W_out), Y_ref, Array(fixed_problem.token_heads))
        d_head = size(fixed_problem.W_in, 1)
        L_fixed = size(fixed_problem.X, 2)
        k_fixed = size(fixed_problem.token_heads, 1)
        Y_cat_ref = vcat([Y_ref[:, slot:k_fixed:end] for slot in 1:k_fixed]...)

        pk.packed_in_proj_gemm!(packed)
        @test Array(packed.Y_cat) ≈ Y_cat_ref rtol=1e-4 atol=1e-4

        fill!(packed.Y_sparse, 0f0)
        pk.packed_in_proj_unpack!(packed; tile_e=d_head, tile_l=8, K=Int32(k_fixed))
        @test Array(packed.Y_sparse) ≈ Y_ref rtol=1e-4 atol=1e-4

        fill!(packed.Y_sparse, 0f0)
        pk.packed_in_proj_practical!(packed; tile_e=d_head, tile_l=8, K=Int32(k_fixed))
        @test Array(packed.Y_sparse) ≈ Y_ref rtol=1e-4 atol=1e-4

        fill!(packed.Z, 0f0)
        pk.packed_out_proj_gemm!(packed)
        @test Array(packed.Z) ≈ Z_ref rtol=1e-4 atol=1e-4

        fill!(packed.Y_cat, 0f0)
        fill!(packed.Z, 0f0)
        pk.packed_out_proj_from_sparse!(packed, CuArray(Y_ref); tile_e=d_head, tile_l=8, K=Int32(k_fixed))
        @test Array(packed.Y_cat) ≈ Y_cat_ref rtol=1e-4 atol=1e-4
        @test Array(packed.Z) ≈ Z_ref rtol=1e-4 atol=1e-4

        fill!(packed.Y_cat, 0f0)
        fill!(packed.Y_sparse, 0f0)
        fill!(packed.Z, 0f0)
        pk.packed_total_practical!(packed; tile_e=d_head, tile_l=8, K=Int32(k_fixed))
        @test Array(packed.Y_sparse) ≈ Y_ref rtol=1e-4 atol=1e-4
        @test Array(packed.Z) ≈ Z_ref rtol=1e-4 atol=1e-4
    end

    @testset "float16 benchmark smoke" begin
        data = pk.prepare(; T=Float16, d_model=16, d_head=8, L=6, H=6, k=4, tile_m=4)
        result = pk.run(data; tile_e=8, tile_d=8, tile_e_small=8, tile_l=4, nruns=1, warmup=0)
        pk.verify(data, result)
        others = pk.run_others(data; tile_d=8, tile_e=8, tile_e_small=8, tile_l=4, nruns=1, warmup=0)
        @test haskey(others, "Packed in-proj GEMM only")
        @test haskey(others, "Packed in-proj + layout")
        @test haskey(others, "Packed total GEMM only")
        @test haskey(others, "Packed total + layout")
    end

    @testset "float16 routed benchmark smoke" begin
        data = pk.prepare(; T=Float16, d_model=16, d_head=8, L=6, H=6, k=4, routing=:random, tile_m=4)
        @test !data.fixed_routing
        result = pk.run(data; tile_e=8, tile_d=8, tile_e_small=8, tile_l=4, nruns=1, warmup=0)
        pk.verify(data, result)
        others = pk.run_others(data; tile_d=8, tile_e=8, tile_e_small=8, tile_l=4, nruns=1, warmup=0)
        @test haskey(others, "Sparse in-proj gather")
        @test !haskey(others, "Sparse in-proj contiguous")
        @test !haskey(others, "Dense fixed-head total")
        @test !haskey(others, "Packed total + layout")
    end

    @testset "scalar k to head lookup" begin
        function head_lookup_kernel(sorted_ids::ct.TileArray{Int32, 1},
                                    token_heads_t::ct.TileArray{Int32, 2},
                                    out::ct.TileArray{Int32, 1},
                                    K::Int32)
            bid = ct.bid(1)
            flat_id = sorted_ids[bid]
            if flat_id == Int32(0)
                ct.store(out, bid, ct.broadcast_to(ct.Tile(Int32(0)), (1,)))
                return nothing
            end
            tok = fld(flat_id - Int32(1), K) + Int32(1)
            slot = rem(flat_id - Int32(1), K) + Int32(1)
            head = token_heads_t[slot, tok]
            ct.store(out, bid, ct.broadcast_to(ct.Tile(head), (1,)))
            return nothing
        end

        out = CUDA.zeros(Int32, Int(dispatch.padded_M))
        ct.launch(head_lookup_kernel, Int(dispatch.padded_M), dispatch.sorted_ids, dispatch.token_heads, out, dispatch.k)
        expected = Int32[
            flat_id == 0 ? 0 : token_heads_cpu[mod1(flat_id, Int(dispatch.k)), fld1(flat_id, Int(dispatch.k))]
            for flat_id in Array(dispatch.sorted_ids)
        ]
        @test Array(out) == expected
    end
end

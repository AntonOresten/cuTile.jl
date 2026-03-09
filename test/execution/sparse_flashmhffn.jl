using CUDA

include(joinpath(@__DIR__, "..", "..", "examples", "sparse_flashmhffn_impl.jl"))

const sf = SparseFlashMHFExample

@testset "sparse flash mhffn" begin
    @testset "forward ones-gated" begin
        data = sf.make_example_problem(; d_head=8, H=4, L=5, k=2, d_inter=16, with_router=false, routing=:random, tile_m=4)
        ones_r = CUDA.ones(Float32, 2, size(data.Q, 2))
        O = CUDA.zeros(Float32, size(data.Q)...)
        sf.sparse_mhffn!(O, data.Q, data.K, data.U, data.V, data.dispatch; R=ones_r, D_E=8, tile_d=8, tile_i=8)
        ref = sf.reference_sparse_mhffn(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(data.token_heads); R=ones(Float32, 2, size(data.Q, 2)), D_E=8)
        @test Array(O) ≈ ref rtol=1e-3 atol=1e-2
    end

    @testset "forward gated" begin
        data = sf.make_example_problem(; d_head=8, H=4, L=5, k=2, d_inter=16, with_router=true, routing=:random, tile_m=4)
        O = CUDA.zeros(Float32, size(data.Q)...)
        sf.sparse_mhffn!(O, data.Q, data.K, data.U, data.V, data.dispatch; R=data.R, D_E=data.D_E, tile_d=8, tile_i=4)
        ref = sf.reference_sparse_mhffn(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(data.token_heads); R=Array(data.R), D_E=data.D_E)
        @test Array(O) ≈ ref rtol=1e-3 atol=1e-2
    end

    @testset "backward ones-gated" begin
        data = sf.make_example_problem(; d_head=8, H=4, L=5, k=2, d_inter=16, with_router=false, routing=:random, tile_m=4)
        ones_r = CUDA.ones(Float32, 2, size(data.Q, 2))
        dO = CUDA.rand(Float32, size(data.Q)...)
        dQ = CUDA.zeros(Float32, size(data.Q)...)
        dK = CUDA.zeros(Float32, size(data.K)...)
        dU = CUDA.zeros(Float32, size(data.U)...)
        dV = CUDA.zeros(Float32, size(data.V)...)
        dR = CUDA.zeros(Float32, 2, size(data.Q, 2))
        sf.∇sparse_mhffn!(dQ, dK, dU, dV, dO, data.Q, data.K, data.U, data.V, data.dispatch; R=ones_r, dR=dR, D_E=8, tile_d=8, tile_i=8)

        ref_dQ, _ = sf.reference_sparse_mhffn_dq(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=ones(Float32, 2, size(data.Q, 2)), D_E=8)
        ref_dK, ref_dU, ref_dV = sf.reference_sparse_mhffn_dkuv(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=ones(Float32, 2, size(data.Q, 2)), D_E=8)
        @test Array(dQ) ≈ ref_dQ rtol=1e-3 atol=1e-2
        @test Array(dK) ≈ ref_dK rtol=1e-3 atol=1e-2
        @test Array(dU) ≈ ref_dU rtol=1e-3 atol=1e-2
        @test Array(dV) ≈ ref_dV rtol=1e-3 atol=1e-2
    end

    @testset "backward gated" begin
        data = sf.make_example_problem(; d_head=8, H=4, L=5, k=2, d_inter=16, with_router=true, routing=:random, tile_m=4)
        dO = CUDA.rand(Float32, size(data.Q)...)
        dQ = CUDA.zeros(Float32, size(data.Q)...)
        dK = CUDA.zeros(Float32, size(data.K)...)
        dU = CUDA.zeros(Float32, size(data.U)...)
        dV = CUDA.zeros(Float32, size(data.V)...)
        dR = CUDA.zeros(Float32, size(data.R)...)
        sf.∇sparse_mhffn!(dQ, dK, dU, dV, dO, data.Q, data.K, data.U, data.V, data.dispatch; R=data.R, dR=dR, D_E=data.D_E, tile_d=8, tile_i=4)

        ref_dQ, ref_dR = sf.reference_sparse_mhffn_dq(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=Array(data.R), D_E=data.D_E)
        ref_dK, ref_dU, ref_dV = sf.reference_sparse_mhffn_dkuv(Array(data.Q), Array(data.K), Array(data.U), Array(data.V), Array(dO), Array(data.token_heads); R=Array(data.R), D_E=data.D_E)
        @test Array(dQ) ≈ ref_dQ rtol=1e-3 atol=1e-2
        @test Array(dK) ≈ ref_dK rtol=1e-3 atol=1e-2
        @test Array(dU) ≈ ref_dU rtol=1e-3 atol=1e-2
        @test Array(dV) ≈ ref_dV rtol=1e-3 atol=1e-2
        @test Array(dR) ≈ ref_dR rtol=1e-3 atol=1e-2
    end

    @testset "fixed-routing smoke" begin
        data = sf.prepare(; T=Float32, d_head=8, H=4, L=8, k=2, d_inter=16, with_router=true, routing=:fixed, tile_m=4)
        result = sf.run(data; tile_d=8, tile_i=4, nruns=1, warmup=0)
        sf.verify(data, result)
        @test length(result.times) == 1
    end
end

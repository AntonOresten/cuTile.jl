using CUDA

@testset "Tiled broadcast" begin
    @testset "1D element-wise" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "fused multi-op" begin
        n = 1024
        A = CUDA.rand(Float32, n) .+ 0.1f0
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(A) .* sin.(ct.Tiled(A))
        @test Array(C) ≈ Array(A) .+ Array(A) .* sin.(Array(A)) rtol=1e-5
    end

    @testset "scalar broadcast" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ 1.0f0
        @test Array(C) ≈ Array(A) .+ 1.0f0
    end

    @testset "2D element-wise" begin
        m, n = 128, 256
        A = CUDA.rand(Float32, m, n)
        B = CUDA.rand(Float32, m, n)
        C = CUDA.zeros(Float32, m, n)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "3D element-wise" begin
        A = CUDA.rand(Float32, 64, 64, 4)
        B = CUDA.rand(Float32, 64, 64, 4)
        C = CUDA.zeros(Float32, 64, 64, 4)
        ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "ct.@. expands to Tiled" begin
        ex = @macroexpand ct.@. C = A + B
        # The macro should produce Tiled() wrapping, not plain dotted calls
        @test occursin("Tiled", string(ex))
    end

    @testset "ct.@. in-place" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + B
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "ct.@. with function" begin
        n = 1024
        A = CUDA.rand(Float32, n) .+ 0.1f0
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + sin(A)
        @test Array(C) ≈ Array(A) .+ sin.(Array(A)) rtol=1e-5
    end

    @testset "ct.@. with scalar" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        C = CUDA.zeros(Float32, n)
        ct.@. C = A + 2.0f0
        @test Array(C) ≈ Array(A) .+ 2.0f0
    end

    @testset "allocating copy" begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = ct.Tiled(A) .+ ct.Tiled(B)
        @test C isa CuArray
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "allocating ct.@." begin
        n = 1024
        A = CUDA.rand(Float32, n)
        B = CUDA.rand(Float32, n)
        C = ct.@. A + B
        @test C isa CuArray
        @test Array(C) ≈ Array(A) .+ Array(B)
    end

    @testset "leading singleton dim" begin
        A = CUDA.rand(Float32, 1, 1024)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .+ 1.0f0
        @test Array(B) ≈ Array(A) .+ 1.0f0
    end

    @testset "double leading singleton" begin
        A = CUDA.rand(Float32, 1, 1, 512)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .* 2.0f0
        @test Array(B) ≈ Array(A) .* 2.0f0
    end

    @testset "small leading dim" begin
        A = CUDA.rand(Float32, 4, 1024)
        B = similar(A)
        ct.Tiled(B) .= ct.Tiled(A) .+ ct.Tiled(A)
        @test Array(B) ≈ 2 .* Array(A)
    end
end

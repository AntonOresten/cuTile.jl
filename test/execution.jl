@testset "execution" begin

using CUDA

@testset "launch" begin

@testset "1D vector add" begin
    function vadd_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "1D vector sub" begin
    function vsub_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a - tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vsub_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) - Array(b)
end

@testset "1D vector mul" begin
    function vmul_1d(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a * tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vmul_1d, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) .* Array(b)
end

@testset "2D matrix add" begin
    function madd_2d(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                     c::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile_a = ct.load(a, (bidx, bidy), (32, 32))
        tile_b = ct.load(b, (bidx, bidy), (32, 32))
        ct.store(c, (bidx, bidy), tile_a + tile_b)
        return
    end

    m, n = 256, 256
    tile_x, tile_y = 32, 32
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(madd_2d, (cld(m, tile_x), cld(n, tile_y)), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "transpose" begin
    function transpose_kernel(x::ct.TileArray{Float32,2}, y::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(x, (bidx, bidy), (32, 32))
        transposed = ct.transpose(tile)
        ct.store(y, (bidy, bidx), transposed)
        return
    end

    m, n = 256, 128
    tile_size = 32
    x = CUDA.rand(Float32, m, n)
    y = CUDA.zeros(Float32, n, m)

    ct.launch(transpose_kernel, (cld(m, tile_size), cld(n, tile_size)), x, y)

    @test Array(y) ≈ transpose(Array(x))
end

end

@testset "Constant parameters" begin

@testset "1D with Constant tile size" begin
    function vadd_const_tile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             c::ct.TileArray{Float32,1}, tile::ct.Constant{Int})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (tile[],))
        tile_b = ct.load(b, pid, (tile[],))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 32
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)

    ct.launch(vadd_const_tile, cld(n, tile_size), a, b, c, ct.Constant(tile_size))

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "2D with Constant tile sizes" begin
    function madd_const_tiles(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                              c::ct.TileArray{Float32,2},
                              tx::ct.Constant{Int}, ty::ct.Constant{Int})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile_a = ct.load(a, (bidx, bidy), (tx[], ty[]))
        tile_b = ct.load(b, (bidx, bidy), (tx[], ty[]))
        ct.store(c, (bidx, bidy), tile_a + tile_b)
        return
    end

    m, n = 256, 256
    tile_x, tile_y = 64, 64
    a = CUDA.rand(Float32, m, n)
    b = CUDA.rand(Float32, m, n)
    c = CUDA.zeros(Float32, m, n)

    ct.launch(madd_const_tiles, (cld(m, tile_x), cld(n, tile_y)), a, b, c,
              ct.Constant(tile_x), ct.Constant(tile_y))

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "data types" begin

@testset "Float64" begin
    function vadd_f64(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1},
                      c::ct.TileArray{Float64,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float64, n)
    b = CUDA.rand(Float64, n)
    c = CUDA.zeros(Float64, n)

    ct.launch(vadd_f64, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

@testset "Float16" begin
    function vadd_f16(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1},
                      c::ct.TileArray{Float16,1})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        ct.store(c, pid, tile_a + tile_b)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float16, n)
    b = CUDA.rand(Float16, n)
    c = CUDA.zeros(Float16, n)

    ct.launch(vadd_f16, cld(n, tile_size), a, b, c)

    @test Array(c) ≈ Array(a) + Array(b)
end

end

@testset "compilation cache" begin
    function cached_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end

    n = 256
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    # First launch triggers compilation
    ct.launch(cached_kernel, cld(n, tile_size), a, b)
    @test Array(b) ≈ Array(a)

    # Second launch should use cached CuFunction
    a2 = CUDA.rand(Float32, n)
    b2 = CUDA.zeros(Float32, n)
    ct.launch(cached_kernel, cld(n, tile_size), a2, b2)
    @test Array(b2) ≈ Array(a2)
end

@testset "TileArray auto-conversion" begin
    # Test that CuArrays are automatically converted to TileArray
    function copy_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(src, pid, (16,))
        ct.store(dst, pid, tile)
        return
    end

    n = 512
    tile_size = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    # Pass CuArrays directly - should auto-convert
    ct.launch(copy_kernel, cld(n, tile_size), src, dst)

    @test Array(dst) ≈ Array(src)
end

end

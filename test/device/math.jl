# math primitives

using CUDA


@testset "bitwise operations" begin

@testset "andi (bitwise AND)" begin
    function bitwise_and_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(&, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_and_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .& Array(b)
end

@testset "ori (bitwise OR)" begin
    function bitwise_or_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                               c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(|, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_or_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .| Array(b)
end

@testset "xori (bitwise XOR)" begin
    function bitwise_xor_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(xor, ta, tb))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(bitwise_xor_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == Array(a) .⊻ Array(b)
end

@testset "shli (shift left)" begin
    function shift_left_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(x -> x << Int32(4), tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x0fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(shift_left_kernel, cld(n, tile_size), a, b)

    @test Array(b) == Array(a) .<< Int32(4)
end

@testset "shri (shift right)" begin
    function shift_right_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(x -> x >> Int32(8), tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(shift_right_kernel, cld(n, tile_size), a, b)

    @test Array(b) == Array(a) .>> Int32(8)
end

@testset "combined bitwise ops" begin
    # (a & b) | (a ^ b) \u2014 exercises all three ops in a single kernel
    function combined_bitwise_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1},
                                     c::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        ta = ct.load(a, pid, (16,))
        tb = ct.load(b, pid, (16,))
        ct.store(c, pid, map(|, map(&, ta, tb), map(xor, ta, tb)))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    c = CUDA.zeros(Int32, n)

    ct.launch(combined_bitwise_kernel, cld(n, tile_size), a, b, c)

    @test Array(c) == (Array(a) .& Array(b)) .| (Array(a) .⊻ Array(b))
end

@testset "bitwise NOT (~)" begin
    function bitwise_not_kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, map(~, tile))
        return
    end

    n = 1024
    tile_size = 16
    a = CuArray(rand(Int32(0):Int32(0x7fff_ffff), n))
    b = CUDA.zeros(Int32, n)

    ct.launch(bitwise_not_kernel, cld(n, tile_size), a, b)

    @test Array(b) == .~Array(a)
end

end


@testset "exp2 with flush_to_zero" begin
    function exp2_ftz_kernel(a::ct.TileArray{Float32, 1}, b::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        result = ct.exp2(tile; flush_to_zero=true)
        ct.store(b, pid, result)
        return
    end

    n = 1024
    tile_size = 16
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(exp2_ftz_kernel, cld(n, tile_size), a, b)

    @test Array(b) ≈ exp2.(Array(a)) rtol=1e-5
end


@testset "truediv" begin

@testset "same shape with ftz and rounding" begin
    function truediv_kernel(a::ct.TileArray{Float32, 2}, b::ct.TileArray{Float32, 2},
                            c::ct.TileArray{Float32, 2})
        pid = ct.bid(1)
        ta = ct.load(a; index=(Int32(1), pid), shape=(64, 1))
        tb = ct.load(b; index=(Int32(1), pid), shape=(64, 1))
        result = ct.truediv(ta, tb; flush_to_zero=true, rounding_mode=ct.Rounding.Approx)
        ct.store(c; index=(Int32(1), pid), tile=result)
        return
    end

    m, n = 64, 32
    a = CUDA.rand(Float32, m, n) .+ 1.0f0
    b = CUDA.rand(Float32, m, n) .+ 1.0f0
    c = CUDA.zeros(Float32, m, n)
    ct.launch(truediv_kernel, n, a, b, c)

    @test Array(c) ≈ Array(a) ./ Array(b) rtol=1e-2
end

@testset "broadcasting" begin
    function truediv_bcast_kernel(a::ct.TileArray{Float32, 2},
                                  c::ct.TileArray{Float32, 2})
        pid = ct.bid(1)
        ta = ct.load(a; index=(Int32(1), pid), shape=(64, 1))
        col_sum = sum(ta; dims=1)  # (1, 1)
        result = ct.truediv(ta, col_sum; flush_to_zero=true, rounding_mode=ct.Rounding.Approx)
        ct.store(c; index=(Int32(1), pid), tile=result)
        return
    end

    m, n = 64, 32
    a = CUDA.rand(Float32, m, n) .+ 1.0f0
    c = CUDA.zeros(Float32, m, n)
    ct.launch(truediv_bcast_kernel, n, a, c)

    a_cpu = Array(a)
    expected = a_cpu ./ sum(a_cpu; dims=1)
    @test Array(c) ≈ expected rtol=1e-2
end

end

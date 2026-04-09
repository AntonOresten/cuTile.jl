# device print tests

using CUDA

@testset "print constant string" begin
    function print_const_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        print("hello world\n")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "hello world"
        ct.launch(print_const_kernel, 1, a)
        CUDA.synchronize()
    end
end

@testset "println with tile" begin
    function print_tile_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("tile=", tile)
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "tile=["
        @check "1.000000"
        ct.launch(print_tile_kernel, 1, a)
        CUDA.synchronize()
    end
end

@testset "print bid (scalar tile)" begin
    function print_bid_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("bid=", bid)
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "bid=1"
        ct.launch(print_bid_kernel, 1, a)
        CUDA.synchronize()
    end
end

@testset "string interpolation" begin
    function interp_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("bid=$bid")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "bid=1"
        ct.launch(interp_kernel, 1, a)
        CUDA.synchronize()
    end
end

@testset "multiple prints" begin
    function multi_print_kernel(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tile = ct.load(a, bid, (16,))
        println("first")
        println("second")
        ct.store(a, bid, tile)
        return
    end

    a = CUDA.ones(Float32, 16)
    @test @filecheck begin
        @check "first"
        @check "second"
        ct.launch(multi_print_kernel, 1, a)
        CUDA.synchronize()
    end
end

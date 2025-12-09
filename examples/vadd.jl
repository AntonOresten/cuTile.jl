# Vector/Matrix addition example - Julia port of cuTile Python's VectorAddition.py sample

using CUDA
import cuTile as ct

# 1D kernels with different tile sizes
function vec_add_1d_tile16(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}) where T
    bid = ct.bid(0)
    a_tile = ct.load(a, bid, (16,))
    b_tile = ct.load(b, bid, (16,))
    ct.store(c, bid, a_tile + b_tile)
    return
end

function vec_add_1d_tile512(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}) where T
    bid = ct.bid(0)
    a_tile = ct.load(a, bid, (512,))
    b_tile = ct.load(b, bid, (512,))
    ct.store(c, bid, a_tile + b_tile)
    return
end

function vec_add_1d_tile1024(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}) where T
    bid = ct.bid(0)
    a_tile = ct.load(a, bid, (1024,))
    b_tile = ct.load(b, bid, (1024,))
    ct.store(c, bid, a_tile + b_tile)
    return
end

# 2D kernels with different tile sizes
function vec_add_2d_tile32(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}) where T
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    a_tile = ct.load(a, (bid_x, bid_y), (32, 32))
    b_tile = ct.load(b, (bid_x, bid_y), (32, 32))
    ct.store(c, (bid_x, bid_y), a_tile + b_tile)
    return
end

function vec_add_2d_tile64(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}) where T
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    a_tile = ct.load(a, (bid_x, bid_y), (64, 64))
    b_tile = ct.load(b, (bid_x, bid_y), (64, 64))
    ct.store(c, (bid_x, bid_y), a_tile + b_tile)
    return
end

function test_add_1d(kernel, ::Type{T}, n, tile; name=nothing) where T
    name = something(name, "1D $(nameof(kernel)) ($n elements, $T)")
    println("--- $name ---")
    a, b = CUDA.rand(T, n), CUDA.rand(T, n)

    argtypes = Tuple{Ptr{T}, Ptr{T}, Ptr{T}}
    cubin = ct.compile(kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, string(nameof(kernel)))

    c = CUDA.zeros(T, n)
    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}, CuPtr{T}}, a, b, c; blocks=cld(n, tile))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function test_add_2d(kernel, ::Type{T}, m, n, tile_x, tile_y; name=nothing) where T
    name = something(name, "2D $(nameof(kernel)) ($m x $n, $T)")
    println("--- $name ---")
    a, b = CUDA.rand(T, m, n), CUDA.rand(T, m, n)

    argtypes = Tuple{Ptr{T}, Ptr{T}, Ptr{T}}
    cubin = ct.compile(kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, string(nameof(kernel)))

    c = CUDA.zeros(T, m, n)
    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}, CuPtr{T}}, a, b, c;
             blocks=(cld(m, tile_x), cld(n, tile_y)))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function main()
    println("--- cuTile Vector/Matrix Addition Examples ---\n")

    # 1D tests with Float32
    test_add_1d(vec_add_1d_tile1024, Float32, 1_024_000, 1024)
    test_add_1d(vec_add_1d_tile512, Float32, 2^20, 512)

    # 1D tests with Float64
    test_add_1d(vec_add_1d_tile512, Float64, 2^18, 512)

    # 1D tests with Float16
    test_add_1d(vec_add_1d_tile1024, Float16, 1_024_000, 1024)

    # 2D tests with Float32
    test_add_2d(vec_add_2d_tile32, Float32, 2048, 1024, 32, 32)
    test_add_2d(vec_add_2d_tile64, Float32, 1024, 2048, 64, 64)

    # 2D tests with Float64
    test_add_2d(vec_add_2d_tile32, Float64, 1024, 512, 32, 32)

    # 2D tests with Float16
    test_add_2d(vec_add_2d_tile64, Float16, 1024, 1024, 64, 64)

    println("\n--- All addition examples completed ---")
end

isinteractive() || main()

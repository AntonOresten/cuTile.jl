# Vector/Matrix addition example - Julia port of cuTile Python's VectorAddition.py sample

using CUDA
import cuTile as ct

# 1D kernel with constant tile size
# Val{TILE} is a ghost type - filtered from parameters, value becomes constant in IR
function vec_add_kernel_1d(a::Ptr{T}, b::Ptr{T}, c::Ptr{T}, ::Val{TILE}) where {T, TILE}
    bid = ct.bid(0)
    a_tile = ct.load(a, bid, (TILE,))
    b_tile = ct.load(b, bid, (TILE,))
    ct.store(c, bid, a_tile + b_tile)
    return
end

# 2D kernel with constant tile sizes
function vec_add_kernel_2d(a::Ptr{T}, b::Ptr{T}, c::Ptr{T},
                           ::Val{TILE_X}, ::Val{TILE_Y}) where {T, TILE_X, TILE_Y}
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    a_tile = ct.load(a, (bid_x, bid_y), (TILE_X, TILE_Y))
    b_tile = ct.load(b, (bid_x, bid_y), (TILE_X, TILE_Y))
    ct.store(c, (bid_x, bid_y), a_tile + b_tile)
    return
end

function test_add_1d(::Type{T}, n, tile; name=nothing) where T
    name = something(name, "1D vec_add ($n elements, $T, tile=$tile)")
    println("--- $name ---")
    a, b = CUDA.rand(T, n), CUDA.rand(T, n)

    # Compile with specific constant tile size
    argtypes = Tuple{Ptr{T}, Ptr{T}, Ptr{T}, Val{tile}}
    cubin = ct.compile(vec_add_kernel_1d, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, "vec_add_kernel_1d")

    c = CUDA.zeros(T, n)
    # Note: Val parameters are ghost types - NOT passed at launch time
    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}, CuPtr{T}}, a, b, c; blocks=cld(n, tile))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function test_add_2d(::Type{T}, m, n, tile_x, tile_y; name=nothing) where T
    name = something(name, "2D vec_add ($m x $n, $T, tiles=$tile_x x $tile_y)")
    println("--- $name ---")
    a, b = CUDA.rand(T, m, n), CUDA.rand(T, m, n)

    # Compile with specific constant tile sizes
    argtypes = Tuple{Ptr{T}, Ptr{T}, Ptr{T}, Val{tile_x}, Val{tile_y}}
    cubin = ct.compile(vec_add_kernel_2d, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, "vec_add_kernel_2d")

    c = CUDA.zeros(T, m, n)
    # Note: Val parameters are ghost types - NOT passed at launch time
    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}, CuPtr{T}}, a, b, c;
             blocks=(cld(m, tile_x), cld(n, tile_y)))

    @assert Array(c) ≈ Array(a) + Array(b)
    println("✓ passed")
end

function main()
    println("--- cuTile Vector/Matrix Addition Examples ---\n")

    # 1D tests with Float32
    test_add_1d(Float32, 1_024_000, 1024)
    test_add_1d(Float32, 2^20, 512)

    # 1D tests with Float64
    test_add_1d(Float64, 2^18, 512)

    # 1D tests with Float16
    test_add_1d(Float16, 1_024_000, 1024)

    # 2D tests with Float32
    test_add_2d(Float32, 2048, 1024, 32, 32)
    test_add_2d(Float32, 1024, 2048, 64, 64)

    # 2D tests with Float64
    test_add_2d(Float64, 1024, 512, 32, 32)

    # 2D tests with Float16
    test_add_2d(Float16, 1024, 1024, 64, 64)

    println("\n--- All addition examples completed ---")
end

isinteractive() || main()

# Matrix transpose example - Julia port of cuTile Python's Transpose.py sample

using CUDA
import cuTile as ct

# Transpose kernels with different tile sizes
function transpose_kernel_32(x::Ptr{T}, y::Ptr{T}) where T
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, (bidx, bidy), (32, 32))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)
    return
end

function transpose_kernel_64(x::Ptr{T}, y::Ptr{T}) where T
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, (bidx, bidy), (64, 64))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)
    return
end

function transpose_kernel_128(x::Ptr{T}, y::Ptr{T}) where T
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, (bidx, bidy), (128, 128))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)
    return
end

function test_transpose(kernel, ::Type{T}, m, n, tm, tn; name=nothing) where T
    name = something(name, "$(nameof(kernel)) ($m x $n, $T)")
    println("--- $name ---")
    x = CUDA.rand(T, m, n)

    argtypes = Tuple{Ptr{T}, Ptr{T}}
    cubin = ct.compile(kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, string(nameof(kernel)))

    grid_x = cld(m, tm)
    grid_y = cld(n, tn)
    y = CUDA.zeros(T, n, m)

    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}}, x, y; blocks=(grid_x, grid_y))

    @assert Array(y) ≈ transpose(Array(x))
    println("✓ passed")
end

function main()
    println("--- cuTile Matrix Transposition Examples ---\n")

    # Float32 tests (like Python's test case 2)
    test_transpose(transpose_kernel_32, Float32, 1024, 512, 32, 32)
    test_transpose(transpose_kernel_64, Float32, 1024, 512, 64, 64)

    # Float64 tests
    test_transpose(transpose_kernel_32, Float64, 1024, 512, 32, 32)
    test_transpose(transpose_kernel_64, Float64, 512, 1024, 64, 64)

    # Float16 tests (like Python's test case 1 with 128x128 tiles)
    test_transpose(transpose_kernel_128, Float16, 1024, 512, 128, 128)
    test_transpose(transpose_kernel_64, Float16, 1024, 1024, 64, 64)

    println("\n--- All transpose examples completed ---")
end

isinteractive() || main()

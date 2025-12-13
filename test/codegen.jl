@testset "codegen" begin

using CUDA
sm_arch = ct.default_sm_arch()

# Helper to disassemble cubin to SASS using cuobjdump
function disasm_sass(cubin::Vector{UInt8})
    mktempdir() do dir
        path = joinpath(dir, "kernel.cubin")
        write(path, cubin)
        read(`cuobjdump -sass $path`, String)
    end
end

@testset "load/store 1D" begin
    function copy_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    cubin = ct.compile(copy_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "copy_kernel")
    @test contains(sass, "LDG")  # Load global (tile load)
    @test contains(sass, "STG")  # Store global (tile store)
end

@testset "load/store 2D" begin
    function copy_2d_kernel(a::Ptr{Float32}, b::Ptr{Float32})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    cubin = ct.compile(copy_2d_kernel, Tuple{Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "copy_2d_kernel")
    @test contains(sass, "LDG")  # Load global (tile load)
    @test contains(sass, "STG")  # Store global (tile store)
end

@testset "add" begin
    function add_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a + tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(add_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "add_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FADD")  # Float add
    @test contains(sass, "STG")   # Tile store
end

@testset "sub" begin
    function sub_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a - tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(sub_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "sub_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FADD")  # Sub is FADD with negated operand
    @test contains(sass, "STG")   # Tile store
end

@testset "mul" begin
    function mul_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
        pid = ct.bid(0)
        tile_a = ct.load(a, pid, (16,))
        tile_b = ct.load(b, pid, (16,))
        result = tile_a * tile_b
        ct.store(c, pid, result)
        return
    end
    cubin = ct.compile(mul_kernel, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "mul_kernel")
    @test contains(sass, "LDG")   # Tile loads
    @test contains(sass, "FMUL")  # Float multiply
    @test contains(sass, "STG")   # Tile store
end

@testset "load from TileArray 1D" begin
    function tilearray_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
        pid = ct.bid(0)
        tile = ct.load(a, pid, (16,))
        ct.store(b, pid, tile)
        return
    end
    spec = ct.ArraySpec{1}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}
    cubin = ct.compile(tilearray_kernel, argtypes; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "tilearray_kernel")
    @test contains(sass, "LDG")  # Tile load
    @test contains(sass, "STG")  # Tile store
end

@testset "load from TileArray 2D" begin
    function tilearray_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2})
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        tile = ct.load(a, (bidx, bidy), (32, 32))
        ct.store(b, (bidx, bidy), tile)
        return
    end
    spec = ct.ArraySpec{2}(16, true)
    argtypes = Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}}
    cubin = ct.compile(tilearray_2d_kernel, argtypes; sm_arch)
    sass = disasm_sass(cubin)
    @test contains(sass, "tilearray_2d_kernel")
    @test contains(sass, "LDG")  # Tile load
    @test contains(sass, "STG")  # Tile store
end

end

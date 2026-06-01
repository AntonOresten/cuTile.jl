using CUDA
using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

spec1d = ct.ArraySpec{1}(16, true)

@testset "codegen" begin

# Float32 -> Float8_E4M3FN (always available; 13.1+)
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E4M3FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float32 -> Float8_E5M2 (always available; 13.1+)
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E5M2}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float32 -> Float8_E8M0FNU works on bytecode 13.2+
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
               bytecode_version=v"13.2") do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float8_E8M0FNU rejected on bytecode 13.1 with a clear version error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.2+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.1")
end

# Float4_E2M1FN requires bytecode 13.3 — rejected at 13.2 with a clear error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float4_E2M1FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.3+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.2")
end

# Whole-tile `reinterpret` between UInt8 and Float4_E2M1FN packs/unpacks two FP4
# per byte: a `Tile{UInt8,(8,)}` unpacks to a `Tile{Float4_E2M1FN,(16,)}`,
# lowering to `cuda_tile.unpack` (13.3+).
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{UInt8,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
               bytecode_version=v"13.3") do a, b
        pid = ct.bid(1)
        bytes = ct.load(a, pid, (8,))            # Tile{UInt8,(8,)}
        @check "unpack"
        fp4 = reinterpret(Float4_E2M1FN, bytes)  # Tile{Float4_E2M1FN,(16,)}
        ct.store(b, pid, convert(ct.Tile{Float32}, fp4))
        return
    end
end

# And the reverse packs FP4 back into bytes via `cuda_tile.pack` (13.3+).
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{UInt8,1,spec1d}};
               bytecode_version=v"13.3") do a, b
        pid = ct.bid(1)
        vals = ct.load(a, pid, (16,))
        fp4 = convert(ct.Tile{Float4_E2M1FN}, vals)  # Tile{Float4_E2M1FN,(16,)}
        @check "pack"
        ct.store(b, pid, reinterpret(UInt8, fp4))    # Tile{UInt8,(8,)}
        return
    end
end

end

# FP8 types are Blackwell-only
@testset "execution" begin
if capability(device()) >= v"10"

# Round-trip Float32 → microfloat → Float32 on values exactly representable
# in the target type — result must match input bit-for-bit.
function rt_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float8_E4M3FN}, tile)))
    return
end
function rt_e5m2(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (16,))
    ct.store(b, pid, convert(ct.Tile{Float32}, convert(ct.Tile{Float8_E5M2}, tile)))
    return
end
# Float8_E4M3FN / Float8_E5M2: 13.1+, always available
representable8 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
                         16.0, 32.0, 64.0, 128.0, 256.0, -1.0, -2.0, -0.5]
let a = CuArray(representable8), b = CUDA.zeros(Float32, length(representable8))
    @cuda backend=cuTile blocks=1 rt_e4m3(a, b)
    @test Array(b) == representable8
    @cuda backend=cuTile blocks=1 rt_e5m2(a, b)
    @test Array(b) == representable8
end

# FMA in FP8: load Float32, convert to FP8, multiply-add in FP8, convert back.
# Uses inputs whose products and sums also stay representable, so the result
# is exact.
function fma_e4m3(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                  c::ct.TileArray{Float32,1}, d::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    ta = convert(ct.Tile{Float8_E4M3FN}, ct.load(a, pid, (16,)))
    tb = convert(ct.Tile{Float8_E4M3FN}, ct.load(b, pid, (16,)))
    tc = convert(ct.Tile{Float8_E4M3FN}, ct.load(c, pid, (16,)))
    ct.store(d, pid, convert(ct.Tile{Float32}, muladd.(ta, tb, tc)))
    return
end
let av = Float32[1.0, 2.0, 0.5, 4.0, 1.5, 2.0, -1.0, -0.5, 3.0, 0.5, 1.0, 2.0, -2.0, 1.0, 0.5, 4.0],
    bv = Float32[2.0, 1.0, 4.0, 0.5, 2.0, 3.0,  2.0,  4.0, 1.0, 2.0, 1.0, 0.5,  2.0, 1.0, 2.0, 1.0],
    cv = Float32[0.0, 1.0, 0.0, 0.0, 1.0, 1.0,  0.0,  0.0, 1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0]
    a, b, c = CuArray(av), CuArray(bv), CuArray(cv)
    d = CUDA.zeros(Float32, length(av))
    @cuda backend=cuTile blocks=1 fma_e4m3(a, b, c, d)
    @test Array(d) == av .* bv .+ cv
end

# Float8_E8M0FNU and Float4_E2M1FN: codegen for `ftof` lowers fine (see the
# codegen tests above) but tileiras refuses to lower a *standalone* f32 ↔
# microfloat conversion on Blackwell — these formats only have meaningful
# hardware paths as the scale/operand dtypes of block-scaled mma, or packed
# into a byte-wide tile via `reinterpret` (below).

# Float4_E2M1FN moves through global memory packed two-per-byte: `reinterpret`
# unpacks a `UInt8` tile into FP4 (doubling the leading dim) and packs it back.

# Pure pack/unpack round-trip: reinterpreting UInt8 → FP4 → UInt8 only
# reinterprets bits, so the bytes must come back unchanged for any input.
function rt_pack(a::ct.TileArray{UInt8,1}, b::ct.TileArray{UInt8,1})
    pid = ct.bid(1)
    bytes = ct.load(a, pid, (8,))
    fp4 = reinterpret(Float4_E2M1FN, bytes)   # unpack: (8,) UInt8 → (16,) FP4
    ct.store(b, pid, reinterpret(UInt8, fp4)) # pack:   (16,) FP4 → (8,) UInt8
    return
end
let av = UInt8[0x00, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde],
    b = CUDA.zeros(UInt8, 8)
    a = CuArray(av)
    @cuda backend=cuTile blocks=1 rt_pack(a, b)
    @test Array(b) == av
end

# Value round-trip through FP4 stored as UInt8: convert Float32 → FP4, pack to
# UInt8 and store; then load UInt8, unpack to FP4 and convert back to Float32.
# All inputs are exactly representable in E2M1 (magnitudes 0,0.5,1,1.5,2,3,4,6),
# so the result must match bit-for-bit.
function pack_fp4(src::ct.TileArray{Float32,1}, dst::ct.TileArray{UInt8,1})
    pid = ct.bid(1)
    vals = ct.load(src, pid, (16,))
    fp4 = convert(ct.Tile{Float4_E2M1FN}, vals)
    ct.store(dst, pid, reinterpret(UInt8, fp4))  # pack: (16,) FP4 → (8,) UInt8
    return
end
function unpack_fp4(src::ct.TileArray{UInt8,1}, dst::ct.TileArray{Float32,1})
    pid = ct.bid(1)
    bytes = ct.load(src, pid, (8,))
    fp4 = reinterpret(Float4_E2M1FN, bytes)      # unpack: (8,) UInt8 → (16,) FP4
    ct.store(dst, pid, convert(ct.Tile{Float32}, fp4))
    return
end
let representable4 = Float32[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.5]
    src = CuArray(representable4)
    packed = CUDA.zeros(UInt8, 8)
    out = CUDA.zeros(Float32, 16)
    @cuda backend=cuTile blocks=1 pack_fp4(src, packed)
    @cuda backend=cuTile blocks=1 unpack_fp4(packed, out)
    @test Array(out) == representable4
end

# N-D reinterpret: pack/unpack are rank-1, but whole-tile `reinterpret` flattens
# (via reshape) so it works on any rank. Round-trip a 2-D Float32 tile through
# 2-D FP4 and 2-D packed UInt8: the leading (column-major) dim absorbs the 2× /
# ½ scaling — (8,2) FP4 ↔ (4,2) UInt8.
function rt_fp4_2d(src::ct.TileArray{Float32,2}, dst::ct.TileArray{Float32,2})
    pid = ct.bid(1)
    fp4 = convert(ct.Tile{Float4_E2M1FN}, ct.load(src, pid, (8, 2)))  # (8,2) FP4
    bytes = reinterpret(UInt8, fp4)                                   # (4,2) UInt8
    fp4b = reinterpret(Float4_E2M1FN, bytes)                         # (8,2) FP4
    ct.store(dst, pid, convert(ct.Tile{Float32}, fp4b))
    return
end
let m = reshape(Float32[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.0,
                        -2.0, -3.0, -4.0, -6.0, 0.5, 1.0, 2.0, 3.0], 8, 2)
    src = CuArray(m)
    out = CUDA.zeros(Float32, 8, 2)
    @cuda backend=cuTile blocks=1 rt_fp4_2d(src, out)
    @test Array(out) == m
end

end
end

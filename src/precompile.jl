using PrecompileTools: @setup_workload, @compile_workload

# Load REPL so that interactive use doesn't invalidate cuTile.jl (REPL.jl's
# `OptimizationParams(::REPLInterpreter)` and other AbsInt methods cause
# invalidation of cuTileInterpreter versions)
import REPL

@setup_workload begin
    # Drive the Julia → Tile IR compilation pipeline at precompile time so the
    # first user-visible kernel launch doesn't have to JIT-compile the entire
    # interpreter, IR structurizer, and bytecode emitter.
    #
    # We call the driver functions directly (`emit_julia` → `emit_structured`
    # → `emit_tile`) rather than going through `code_tiled`, because the
    # latter writes the bytecode to a temp file so it can shell out to
    # `cuda-tile-translate` for disassembly. We don't want to materialize
    # files at precompile time — we only want to exercise the codegen path.
    #
    # `cufunction` itself can't run at precompile time because it dispatches
    # on the current GPU's compute capability via `check_tile_ir_support`.
    # Instead we loop over a representative set of shader models so the
    # bytecode emitter is primed for whatever device the user ends up on.
    function vadd_1d(a::TileArray{T,1}, b::TileArray{T,1},
                     c::TileArray{T,1}, tile) where {T}
        bid = cuTile.bid(1)
        a_tile = cuTile.load(a; index=bid, shape=(tile,))
        b_tile = cuTile.load(b; index=bid, shape=(tile,))
        cuTile.store(c; index=bid, tile=a_tile + b_tile)
        return
    end

    function vadd_2d(a::TileArray{T,2}, b::TileArray{T,2},
                     c::TileArray{T,2}, tx, ty) where {T}
        bx = cuTile.bid(1); by = cuTile.bid(2)
        a_tile = cuTile.load(a; index=(bx, by), shape=(tx, ty))
        b_tile = cuTile.load(b; index=(bx, by), shape=(tx, ty))
        cuTile.store(c; index=(bx, by), tile=a_tile + b_tile)
        return
    end

    function vadd_gather(a::TileArray{T,1}, b::TileArray{T,1},
                         c::TileArray{T,1}, tile) where {T}
        bid = cuTile.bid(1)
        offsets = cuTile.arange(tile)
        base = cuTile.Tile((bid - Int32(1)) * Int32(tile))
        indices = cuTile.broadcast_to(base, (tile,)) .+ offsets
        a_tile = cuTile.gather(a, indices)
        b_tile = cuTile.gather(b, indices)
        cuTile.scatter(c, indices, a_tile + b_tile)
        return
    end

    spec1d = ArraySpec{1}(16, true)
    spec2d = ArraySpec{2}(16, true)

    # Drive the full driver chain for a single (kernel, argtypes) signature.
    # Loops over a small set of representative shader models so the per-arch
    # bytecode emission paths are precompiled regardless of which GPU the
    # user eventually runs on. The bytecode is generated in-memory and
    # immediately discarded — we don't materialize files.
    function precompile_kernel(@nospecialize(f), @nospecialize(tt))
        world = Base.get_world_counter()
        stripped, const_argtypes = process_const_argtypes(f, tt)
        mi = lookup_method_instance(f, stripped; world)
        # Ampere/Ada (8.0) and Blackwell (10.0) cover all supported archs;
        # bytecode emission keys off `sm_arch` for entry-hint encoding only,
        # so two representative values are enough to prime that path.
        for sm_arch in (v"8.0", v"10.0")
            # Use a real `TileCacheKey` so the workload's specializations of
            # `emit_*!` and `write_bytecode!` match the launch path's cache type.
            key = TileCacheKey(sm_arch, DEFAULT_BYTECODE_VERSION, nothing, nothing, nothing)
            cache = CacheView{CuTileResults}(key, world)
            ir, rettype = emit_julia(cache, mi; const_argtypes)
            sci, rettype, kernel_meta = emit_structured(ir, rettype)
            opts = CGOpts((sm_arch=sm_arch, opt_level=nothing,
                           num_ctas=nothing, occupancy=nothing,
                           bytecode_version=DEFAULT_BYTECODE_VERSION))
            emit_tile(sci, rettype, kernel_meta;
                      name=sanitize_name(string(mi.def.name)),
                      opts, cache, const_argtypes)
        end
        return
    end

    @compile_workload begin
        # 1D vec_add — load/add/store across float types. Different `T`s share
        # the cuTileInterpreter typeinf path, so this primarily covers the
        # bytecode-emission specializations on tile element type.
        for T in (Float32, Float16)
            tt = Tuple{TileArray{T, 1, spec1d},
                       TileArray{T, 1, spec1d},
                       TileArray{T, 1, spec1d},
                       Constant{Int, 1024}}
            precompile_kernel(vadd_1d, tt)
        end

        # 2D vec_add — multi-dim block IDs and shapes.
        let tt = Tuple{TileArray{Float32, 2, spec2d},
                       TileArray{Float32, 2, spec2d},
                       TileArray{Float32, 2, spec2d},
                       Constant{Int, 32}, Constant{Int, 32}}
            precompile_kernel(vadd_2d, tt)
        end

        # Gather/scatter path — arange, broadcast_to, gather, scatter, and
        # the contiguous_gather assume infrastructure.
        let tt = Tuple{TileArray{Float32, 1, spec1d},
                       TileArray{Float32, 1, spec1d},
                       TileArray{Float32, 1, spec1d},
                       Constant{Int, 1024}}
            precompile_kernel(vadd_gather, tt)
        end

        # No-Constant path. Covers `const_argtypes::Nothing` specializations
        # of the codegen-pipeline closures (write_bytecode!, emit_tile, etc.)
        # — the launch path uses the Nothing variant whenever the kernel has
        # no `Constant{T,V}` arguments (e.g. `@cuda backend=cuTile identity(nothing)`).
        precompile_kernel(identity, Tuple{Nothing})
    end

    # Explicit precompile of the launch path. The workload above covers the
    # codegen pipeline starting at `emit_julia`, but the launch entry
    # (`cufunction_compile`) is not exercised by the workload — its MI is
    # precompiled here so the first user-visible launch doesn't pay its JIT cost.
    precompile(Tuple{typeof(cufunction_compile),
                     typeof(identity), Type{Tuple{Nothing}},
                     Type{Tuple{Nothing}}, Nothing, TileCacheKey})
end

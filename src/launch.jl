# Host-side kernel launch.
#
# Compiles a Julia function with `TileArray` arguments to Tile IR bytecode,
# runs `tileiras` to lower bytecode → CUBIN, loads the cubin into the active
# CUDA context, and launches it via `cudacall`. Compilation is cached per
# `(MethodInstance, sm_arch, opt_level, num_ctas, occupancy, bytecode_version)`.

using CUDACore: CUDACore, CuArray, CuModule, CuFunction, cudacall, device, capability,
                AbstractBackend, AbstractKernel, kernel_convert, kernel_compile
using CUDA_Compiler_jll

using Adapt: Adapt, adapt

"""
    KernelAdaptor

`Adapt.jl` adaptor used to convert host-side launch arguments into their
kernel-side counterparts. `AbstractArray`s become `TileArray`s; `Type`
values become `Constant`s. User-defined structs containing arrays compose
naturally via `Adapt.adapt_structure`.

This is the cuTile analogue of `CUDACore.KernelAdaptor`.
"""
struct KernelAdaptor end

Adapt.adapt_storage(::KernelAdaptor, arr::AbstractArray) = TileArray(arr)
Adapt.adapt_storage(::KernelAdaptor, t::Type) = Constant(t)

# Adapt's default `adapt_structure(to, ::PermutedDimsArray)` recurses by
# rebuilding `PermutedDimsArray(adapt(parent), perm)`. We can't follow that
# pattern because `TileArray` isn't `<:AbstractArray` — strided-wrapper
# state is absorbed into its `sizes`/`strides` fields directly. Short-circuit
# the recursion so the whole wrapper becomes a single TileArray.
Adapt.adapt_structure(::KernelAdaptor, arr::PermutedDimsArray) = TileArray(arr)

"""
    cuTileconvert(x)

Convert a launch argument to its kernel-side form via `Adapt.adapt` with
`KernelAdaptor()`. Mirrors `CUDACore.cudaconvert`.
"""
cuTileconvert(x) = adapt(KernelAdaptor(), x)


#=============================================================================
 Backend registration — plugs cuTile into CUDACore's `@cuda` dispatch protocol.
=============================================================================#

"""
    TileBackend()

cuTile backend for `@cuda backend=...`. Routes the call through
[`cuTile.cufunction`](@ref) (Tile IR bytecode → tileiras → CUBIN) and
returns a [`TileKernel`](@ref) for launch.

```julia
@cuda backend=cuTile.TileBackend() blocks=N my_kernel(a, b, c)
```
"""
struct TileBackend <: AbstractBackend end

CUDACore.kernel_convert(::TileBackend, x) = cuTileconvert(x)

CUDACore.kernel_compile(::TileBackend, f::F, tt::TT=Tuple{}; kwargs...) where {F,TT} =
    cufunction(f, tt; kwargs...)


#=============================================================================
 Toolkit / device validation (cached: once per `(capability, cuda_version)`).
=============================================================================#

const _tile_ir_support_cache = Dict{Tuple{VersionNumber, VersionNumber}, VersionNumber}()
const _tile_ir_support_lock = ReentrantLock()

function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)
    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))
    return proc, log
end

"""
    check_tile_ir_support()

Validate that the current CUDA toolkit supports Tile IR on the active device.
Result is cached per `(capability, cuda_version)` — checking compute capability
and toolkit version on every kernel launch is wasteful.
"""
function check_tile_ir_support()
    if !CUDA_Compiler_jll.is_available()
        error("CUDA_Compiler_jll is not available; cannot compile Tile IR kernels")
    end

    cuda_ver = CUDA_Compiler_jll.cuda_version
    cap = capability(device())
    key = (cap, cuda_ver)

    cached = Base.@lock _tile_ir_support_lock get(_tile_ir_support_cache, key, nothing)
    cached !== nothing && return cached

    sm_str = format_sm_arch(cap)
    if cap >= v"10.0"       # Blackwell
        cuda_ver >= v"13.1" ||
            error("Tile IR on Blackwell ($sm_str) requires CUDA ≥ 13.1, got $cuda_ver")
    elseif cap >= v"9.0"    # Hopper — not supported
        error("Tile IR is not supported on Hopper ($sm_str)")
    elseif cap >= v"8.0"    # Ampere / Ada
        cuda_ver >= v"13.2" ||
            error("Tile IR on Ampere/Ada ($sm_str) requires CUDA ≥ 13.2, got $cuda_ver")
    else
        error("Tile IR is not supported on compute capability $cap ($sm_str)")
    end

    bytecode_version = VersionNumber(cuda_ver.major, cuda_ver.minor)
    Base.@lock _tile_ir_support_lock _tile_ir_support_cache[key] = bytecode_version
    return bytecode_version
end


#=============================================================================
 Argument-type unwrapping for cufunction.
=============================================================================#

"""
    unwrap_argtypes(f, tt) -> (argtypes::Type{<:Tuple}, const_argtypes::Union{Vector{Any},Nothing})

Compile-time-specialized derivation of:
- `argtypes::Type{<:Tuple}` — concrete dispatch tuple for `method_instance(f, argtypes)`,
  with `Constant{T,V}` slots unwrapped to `T`.
- `const_argtypes::Vector{Any}` — `[CC.Const(f), ...args]` with `Constant{T,V}` slots
  replaced by `CC.Const(V)`, for const-prop inference. `nothing` when no `Constant`
  arguments are present (skips the const-seeding pipeline entirely).

`@generated` so the unwrapped `Tuple` type and the `Constant`-vs-not branching
fold to constants at the call site. Only the `Vector{Any}` allocation and the
`CC.Const(...)` boxes for runtime values survive to runtime.
"""
@generated function unwrap_argtypes(@nospecialize(f), ::Type{TT}) where TT <: Tuple
    unwrapped = map(t -> t <: Constant ? constant_eltype(t) : t, TT.parameters)
    argtypes_T = Tuple{unwrapped...}
    has_consts = any(t -> t <: Constant, TT.parameters)
    has_consts || return :(($argtypes_T, nothing))

    cats_exprs = Any[:(CC.Const(f))]
    for t in TT.parameters
        if t <: Constant
            push!(cats_exprs, :(CC.Const($(t.parameters[2]))))
        else
            push!(cats_exprs, t)
        end
    end
    return :(($argtypes_T, Any[$(cats_exprs...)]))
end


#=============================================================================
 Compilation: bytecode → CUBIN → CuFunction.
=============================================================================#

"""
    emit_binary!(cache, mi, ci, res; const_argtypes=nothing) -> Vector{UInt8}

Cached binary phase: compile Tile IR bytecode to CUBIN using tileiras.
"""
function emit_binary!(cache::CacheView, mi::Core.MethodInstance,
                      ci::Core.CodeInstance, res::CuTileResults;
                      const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Recurse first — emit_structured! at the bottom of the chain fires
    # `compile_hook` for `@device_code_*` reflection, which must run on every
    # launch even when downstream artifacts are fully cached.
    bytecode = emit_tile!(cache, mi, ci, res; const_argtypes)

    res.cuda_bin !== nothing && return res.cuda_bin

    opts = cache.owner[2]

    # Resolve opt_level here (not in emit_tile) because it's a tileiras flag, not bytecode.
    # num_ctas/occupancy are resolved in emit_tile because they're encoded in bytecode.
    _, _, kernel_meta = res.julia_ir
    opt_level = something(resolve_hint(opts.opt_level, kernel_meta, :opt_level, opts.sm_arch), 3)

    # Run tileiras to produce CUBIN
    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"
    compiled = false
    try
        write(input_path, bytecode)
        cmd = addenv(`$(CUDA_Compiler_jll.tileiras()) $input_path -o $output_path --gpu-name $(format_sm_arch(opts.sm_arch)) -O$(opt_level) --lineinfo`,
                     "CUDA_ROOT" => CUDA_Compiler_jll.artifact_dir)
        proc, log = run_and_collect(cmd)
        if !success(proc)
            reason = proc.termsignal > 0 ? "tileiras received signal $(proc.termsignal)" :
                                           "tileiras exited with code $(proc.exitcode)"
            msg = "Failed to compile Tile IR ($reason)"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(input_path)"
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $(input_path)`)
            end
            error(msg)
        end
        compiled = true
        res.cuda_bin = read(output_path)
    finally
        compiled && rm(input_path, force=true)
        rm(output_path, force=true)
    end

    return res.cuda_bin
end

"""
    emit_function!(cache, mi, ci, res; const_argtypes=nothing) -> CuFunction

Cached function phase: load CUBIN into GPU memory as a CuFunction.
"""
function emit_function!(cache::CacheView, mi::Core.MethodInstance,
                        ci::Core.CodeInstance, res::CuTileResults;
                        const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    cubin = emit_binary!(cache, mi, ci, res; const_argtypes)

    res.cuda_func !== nothing && return res.cuda_func

    kernel_name = sanitize_name(string(mi.def.name))
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, kernel_name)
    res.cuda_func = cufunc
    return cufunc
end


#=============================================================================
 TileKernel + cufunction: hoisted compilation step.

 Mirrors the `cufunction(f, tt) -> HostKernel` pattern in CUDACore. Once
 obtained, calling `(::TileKernel)(args...; blocks=…)` skips the MI lookup
 and CompilerCaching dispatch — only argument flatten + `cudacall` runs.
=============================================================================#

"""
    TileKernel{F, TT}

A compiled cuTile kernel. Returned by [`cuTile.cufunction`](@ref) and the
target of `(::TileKernel)(args...; blocks, …)` calls. Concrete subtype of
`CUDACore.AbstractKernel`.
"""
struct TileKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::CuFunction
end

"""
    cuTile.cufunction(f, tt=Tuple{}; sm_arch=nothing, opt_level=nothing,
                      num_ctas=nothing, occupancy=nothing, name=nothing) -> TileKernel

Compile `f` for the cuTile backend. `tt` is the tuple of *converted*
argument types (i.e. after `cuTileconvert`/`Adapt.adapt(KernelAdaptor(), …)`).
Compilation is cached; calling `cufunction` repeatedly with the same
`(f, tt, opts)` is O(1) after the first compile.

Mirrors `CUDACore.cufunction` but produces a [`TileKernel`](@ref). Caching
is delegated to CompilerCaching: the resulting `TileKernel` is stored in
the `CuTileResults` attached to the underlying Julia `CodeInstance`, so
invalidation rides on Julia's normal CI lifecycle.
"""
function cufunction(@nospecialize(f), tt::Type{<:Tuple}=Tuple{};
                    sm_arch::Union{VersionNumber, Nothing}=nothing,
                    opt_level::Union{Int, Nothing}=nothing,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing,
                    name::Union{String, Nothing}=nothing)
    bytecode_version = check_tile_ir_support()
    resolved_sm_arch = sm_arch !== nothing ? sm_arch : default_sm_arch()

    opts = (sm_arch=resolved_sm_arch, opt_level=opt_level,
            num_ctas=num_ctas, occupancy=occupancy,
            bytecode_version=bytecode_version)

    # Single pass over `tt.parameters`: build the unwrapped argtypes tuple
    # (Constant{T,V} → T for MI lookup) and the const_argtypes vector
    # (Constant{T,V} → CC.Const(V) for inference) together. cufunction
    # specializes on `tt`, so this loop unrolls per kernel signature.
    argtypes, const_argtypes = unwrap_argtypes(f, tt)

    world = Base.get_world_counter()
    mi = method_instance(f, argtypes; world)
    mi === nothing && throw(MethodError(f, argtypes))
    if !Base.isdispatchtuple(mi.specTypes)
        sig = Base.signature_type(f, argtypes)
        mi = CC.specialize_method(mi.def, sig, mi.sparam_vals)::Core.MethodInstance
    end

    cache = CacheView{CuTileResults}((:cuTile, opts), world)

    # Single resolution of (ci, res) up front — threaded through the emit_*!
    # chain so each phase only does its own short-circuit, not redundant
    # cache lookups. The cached compilation results are attached to the
    # underlying `CodeInstance` via CompilerCaching; the `TileKernel` wrapper
    # rides along in the same `CuTileResults`, so kernel-instance lifecycle
    # follows the CI's instead of needing a separate global Dict.
    ci, res = ensure_compiled(cache, mi, const_argtypes)

    # Always walk the emit chain (each phase short-circuits on its own cached
    # field, but `emit_structured!` also fires `compile_hook` for reflection,
    # which has to run on every launch even when the cube/cufunc is cached).
    cufunc = emit_function!(cache, mi, ci, res; const_argtypes)

    res.tile_kernel !== nothing && return res.tile_kernel::TileKernel{Core.Typeof(f), tt}
    kernel = TileKernel{Core.Typeof(f), tt}(f, cufunc)
    res.tile_kernel = kernel
    return kernel
end

# Tile IR has a 24-bit grid limit per dimension.
const _MAX_GRID_DIM = (1 << 24) - 1

# `convert=Val(...)` is the AbstractKernel callable convention from CUDACore;
# `@cuda` passes `convert=Val(false)` because args were already converted at
# expansion time. We always treat args as already-converted — direct
# `kernel(args...)` calls without the macro should pass converted args.
function (k::TileKernel)(args::Vararg{Any, N}; blocks, threads=1,
                         convert=Val(false), kwargs...) where {N}
    state = KernelState()

    # Flatten args for cudacall — Tile IR uses flat scalar parameter signatures,
    # so each TileArray expands to (ptr, sizes..., strides...). Constant returns
    # () so ghost types disappear. Trailing `flatten(state)` matches the implicit
    # KernelState slot at the end of the bytecode kernel signature.
    flat_args = (Iterators.flatten(map(flatten, args))..., flatten(state)...)
    flat_types = Tuple{map(typeof, flat_args)...}

    grid_dims = blocks isa Integer ? (blocks,) : blocks
    for (i, dim) in enumerate(grid_dims)
        if dim > _MAX_GRID_DIM
            error("Grid[$i] exceeds 24-bit limit: max=$_MAX_GRID_DIM, got=$dim. " *
                  "Use multiple kernel launches for larger workloads.")
        end
    end

    # Note: threads=1 lets the driver use the cubin's EIATTR_REQNTID metadata
    # which specifies the actual thread count (typically 128 for Tile kernels).
    cudacall(k.fun, flat_types, flat_args...; blocks=grid_dims, threads, kwargs...)
    return nothing
end


#=============================================================================
 launch: high-level convenience wrapper, retained as the function-call entry
 point alongside `@cuda backend=cuTile.TileBackend() …`.
=============================================================================#

"""
    launch(f, grid, args...; sm_arch=nothing, opt_level=nothing,
           num_ctas=nothing, occupancy=nothing, name=nothing)

Compile and launch a Tile IR kernel. `args` are converted via
`cuTileconvert` (CuArray → TileArray, Type → Constant). Equivalent to
`@cuda backend=cuTile.TileBackend() blocks=grid f(args...)` modulo
slight kwarg naming.

# Example
```julia
using CUDA, cuTile

a = CUDA.zeros(Float32, 1024); b = CUDA.ones(Float32, 1024); c = similar(a)

function vadd_kernel(a::cuTile.TileArray{Float32,1}, b::cuTile.TileArray{Float32,1},
                     c::cuTile.TileArray{Float32,1})
    pid = cuTile.bid(0)
    ta = cuTile.load(a, (pid,), (16,))
    tb = cuTile.load(b, (pid,), (16,))
    cuTile.store(c, (pid,), ta + tb)
    return
end

cuTile.launch(vadd_kernel, 64, a, b, c)
```
"""
function launch(@nospecialize(f), grid, args...;
                sm_arch::Union{VersionNumber, Nothing}=nothing,
                opt_level::Union{Int, Nothing}=nothing,
                num_ctas::Union{Int, Nothing}=nothing,
                occupancy::Union{Int, Nothing}=nothing,
                name::Union{String, Nothing}=nothing)
    converted = map(cuTileconvert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    kernel = cufunction(f, tt; sm_arch, opt_level, num_ctas, occupancy, name)
    kernel(converted...; blocks=grid)
    return nothing
end

"""
    default_sm_arch() -> VersionNumber

Get the compute capability of the current CUDA device as a VersionNumber.
Returns e.g. `v"12.0"` for compute capability 12.0.
"""
default_sm_arch() = capability(device())

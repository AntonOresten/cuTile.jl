"""
    DiskCache

Content-addressable disk cache for compiled Tile IR → CUBIN artifacts.

cuTile's in-memory cache (on `CodeInstance` via `CompilerCaching`) only
covers a single session. This submodule adds an LMDB-backed cache that
persists across sessions, so the second run of a kernel skips `tileiras`
entirely. Modeled on JuliaLang/julia#61527 (LLVM `objcache`) and
cuTile Python's SQLite cache (`cuda.tile._cache`).

The implementation talks to LMDB directly via `LMDB_jll`. The public
surface is intentionally narrow (`open`, `close`, `get`, `put!`,
`compute_key`, plus the lazy `global_cache` accessor) so we can swap the
backend to `LMDB.jl` later without touching call sites.

# Layout
- A single LMDB env at `\$(scratchspace)/disk_cache/`, where
  `\$(scratchspace)` is the Scratch.jl-managed directory for cuTile
  (typically `\$DEPOT/scratchspaces/<cuTile-UUID>/disk_cache`). The
  directory contains exactly two files (`data.mdb`, `lock.mdb`); wiping
  the cache means `rm -rf` of that directory or `Scratch.delete_scratch!`.
- A single (unnamed) main database inside the env.
- Key = `sha256(SCHEMA ‖ toolkit_version ‖ sm_arch ‖ opt_level ‖ bytecode)`.

The toolkit version lives in the hash, not the path, so a single env
holds entries from any number of toolkit versions safely. Old-toolkit
entries become unreachable garbage when the toolkit is upgraded, but
they're bounded by the env's `mapsize` cap. See [Eviction](#Eviction)
below for the strategy when the cap is hit.

The combination of content-addressable keys and per-version envs makes
stale hits structurally impossible.
"""
module DiskCache

import LMDB_jll
using Scratch: @get_scratch!
using SHA: sha256

# ===========================================================================
# Minimal LMDB binding
# ===========================================================================

const MDB_RDONLY      = Cuint(0x00020000)
const MDB_NOTLS       = Cuint(0x00200000)
const MDB_NORDAHEAD   = Cuint(0x00800000)
const MDB_NOOVERWRITE = Cuint(0x00000010)

const MDB_SUCCESS     = Cint(0)
const MDB_KEYEXIST    = Cint(-30799)
const MDB_NOTFOUND    = Cint(-30798)

struct MDB_val
    mv_size::Csize_t
    mv_data::Ptr{Cvoid}
end

@inline liblmdb() = LMDB_jll.liblmdb

errstr(ret::Cint) =
    unsafe_string(ccall((:mdb_strerror, liblmdb()), Cstring, (Cint,), ret))

@inline function check(ret::Cint, what)
    iszero(ret) && return nothing
    error("LMDB $what failed: $(errstr(ret))")
end

# ===========================================================================
# Cache handle
# ===========================================================================

"""
    Cache

Opaque handle to an opened LMDB-backed disk cache. Created via [`open`](@ref),
released via [`close`](@ref). Safe to share across threads (LMDB serializes
writers internally; readers use `MDB_NOTLS` so they're decoupled from
the OS thread).
"""
mutable struct Cache
    env::Ptr{Cvoid}     # MDB_env*
    dbi::Cuint          # main DB handle, valid for the env's lifetime
    path::String

    function Cache(env::Ptr{Cvoid}, dbi::Cuint, path::AbstractString)
        c = new(env, dbi, String(path))
        finalizer(close, c)
        return c
    end
end

isopen(cache::Cache) = cache.env != C_NULL

"""
    close(cache::Cache)

Release the underlying LMDB environment. Idempotent.
"""
function close(cache::Cache)
    cache.env == C_NULL && return
    ccall((:mdb_env_close, liblmdb()), Cvoid, (Ptr{Cvoid},), cache.env)
    cache.env = C_NULL
    return
end

"""
    open(path; mapsize = 1<<30, maxreaders = 510) -> Cache

Open or create the disk cache at `path`. The directory is created if
missing. `mapsize` is the maximum on-disk size in bytes (LMDB grows the
map sparsely up to this limit). `maxreaders` caps concurrent reader
transactions.
"""
function open(path::AbstractString; mapsize::Integer = (Csize_t(1) << 30),
              maxreaders::Integer = 510)
    mkpath(path)

    env_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_env_create, liblmdb()), Cint, (Ref{Ptr{Cvoid}},), env_ref),
          "mdb_env_create")
    env = env_ref[]

    try
        check(ccall((:mdb_env_set_maxreaders, liblmdb()), Cint,
                    (Ptr{Cvoid}, Cuint), env, Cuint(maxreaders)),
              "mdb_env_set_maxreaders")
        check(ccall((:mdb_env_set_mapsize, liblmdb()), Cint,
                    (Ptr{Cvoid}, Csize_t), env, Csize_t(mapsize)),
              "mdb_env_set_mapsize")

        # MDB_NOTLS: read txns aren't tied to the OS thread that opened them
        # (Julia tasks may migrate). MDB_NORDAHEAD: lookups are random-access,
        # so OS read-ahead is wasted I/O.
        flags = MDB_NOTLS | MDB_NORDAHEAD
        check(ccall((:mdb_env_open, liblmdb()), Cint,
                    (Ptr{Cvoid}, Cstring, Cuint, Cushort),
                    env, path, flags, Cushort(0o644)),
              "mdb_env_open($(repr(path)))")

        dbi = open_main_db!(env)
        return Cache(env, dbi, path)
    catch
        ccall((:mdb_env_close, liblmdb()), Cvoid, (Ptr{Cvoid},), env)
        rethrow()
    end
end

# Get a reusable handle to the env's main (unnamed) DB. The handle becomes
# valid in subsequent transactions only after the opening txn commits, so
# we always do this through a fresh dummy write txn.
function open_main_db!(env::Ptr{Cvoid})
    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb()), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                env, C_NULL, Cuint(0), txn_ref),
          "mdb_txn_begin (init)")
    txn = txn_ref[]

    dbi_ref = Ref{Cuint}(0)
    ret = ccall((:mdb_dbi_open, liblmdb()), Cint,
                (Ptr{Cvoid}, Ptr{Cchar}, Cuint, Ref{Cuint}),
                txn, Ptr{Cchar}(C_NULL), Cuint(0), dbi_ref)
    if !iszero(ret)
        ccall((:mdb_txn_abort, liblmdb()), Cvoid, (Ptr{Cvoid},), txn)
        check(ret, "mdb_dbi_open (main)")
    end

    check(ccall((:mdb_txn_commit, liblmdb()), Cint, (Ptr{Cvoid},), txn),
          "mdb_txn_commit (init)")

    return dbi_ref[]
end

# ===========================================================================
# get / put!
# ===========================================================================

"""
    get(cache, key) -> Union{Vector{UInt8}, Nothing}

Look up `key` in the cache. Returns a freshly-allocated copy of the
value on hit, or `nothing` on miss. The copy is necessary because LMDB
hands back a pointer into the memory-mapped data file, which a future
writer is allowed to reuse.
"""
function get(cache::Cache, key::Vector{UInt8})
    cache.env != C_NULL || error("DiskCache.get on closed cache")

    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb()), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, MDB_RDONLY, txn_ref),
          "mdb_txn_begin (read)")
    txn = txn_ref[]

    try
        key_val  = Ref(MDB_val(Csize_t(length(key)), pointer(key)))
        data_val = Ref(MDB_val(Csize_t(0), C_NULL))

        ret = GC.@preserve key ccall(
            (:mdb_get, liblmdb()), Cint,
            (Ptr{Cvoid}, Cuint, Ref{MDB_val}, Ref{MDB_val}),
            txn, cache.dbi, key_val, data_val)

        if ret == MDB_NOTFOUND
            return nothing
        end
        check(ret, "mdb_get")

        sz = Int(data_val[].mv_size)
        out = Vector{UInt8}(undef, sz)
        unsafe_copyto!(pointer(out), Ptr{UInt8}(data_val[].mv_data), sz)
        return out
    finally
        ccall((:mdb_txn_abort, liblmdb()), Cvoid, (Ptr{Cvoid},), txn)
    end
end

"""
    put!(cache, key, value)

Insert `key => value` into the cache. Existing entries are not
overwritten — under content addressing, a key collision means the values
are identical (or, vanishingly, a SHA-256 collision); either way, the
first writer wins.
"""
function put!(cache::Cache, key::Vector{UInt8}, value::Vector{UInt8})
    cache.env != C_NULL || error("DiskCache.put! on closed cache")

    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb()), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, Cuint(0), txn_ref),
          "mdb_txn_begin (write)")
    txn = txn_ref[]

    committed = false
    try
        key_val = Ref(MDB_val(Csize_t(length(key)),   pointer(key)))
        val_val = Ref(MDB_val(Csize_t(length(value)), pointer(value)))

        ret = GC.@preserve key value ccall(
            (:mdb_put, liblmdb()), Cint,
            (Ptr{Cvoid}, Cuint, Ref{MDB_val}, Ref{MDB_val}, Cuint),
            txn, cache.dbi, key_val, val_val, MDB_NOOVERWRITE)
        if ret != MDB_KEYEXIST
            check(ret, "mdb_put")
        end

        check(ccall((:mdb_txn_commit, liblmdb()), Cint, (Ptr{Cvoid},), txn),
              "mdb_txn_commit (write)")
        committed = true
    finally
        committed || ccall((:mdb_txn_abort, liblmdb()), Cvoid, (Ptr{Cvoid},), txn)
    end
    return
end

# ===========================================================================
# Key derivation
# ===========================================================================

# Bumped manually when the cache key derivation changes shape. cuTile
# bytecode format changes are picked up automatically because the bytecode
# bytes themselves are mixed into the hash. The toolkit version is encoded
# in the env path, not here.
const _CACHE_SCHEMA = UInt32(1)

"""
    compute_key(bytecode, sm_arch, opt_level, toolkit_version) -> Vector{UInt8}

Derive a 32-byte content-addressable cache key for a Tile IR compilation.
The key covers the bytecode plus every input that changes the resulting
CUBIN: target arch, opt level, and the `tileiras` toolkit version.

A different toolkit version produces a different key for the same
bytecode, so old-toolkit entries simply never match on lookup.
"""
function compute_key(bytecode::Vector{UInt8}, sm_arch::VersionNumber,
                     opt_level::Integer, toolkit_version::VersionNumber)
    io = IOBuffer()
    write(io, hton(_CACHE_SCHEMA))
    write_pstring!(io, string(toolkit_version))
    write_pstring!(io, string(sm_arch))
    write(io, hton(UInt32(opt_level)))
    write(io, bytecode)
    return sha256(take!(io))
end

@inline function write_pstring!(io::IO, s::AbstractString)
    bytes = codeunits(s)
    write(io, hton(UInt32(length(bytes))))
    write(io, bytes)
end

# ===========================================================================
# Process-wide singletons (one Cache per toolkit version)
# ===========================================================================

const _global_cache             = Ref{Union{Cache, Nothing}}(nothing)
const _global_cache_initialized = Ref(false)
const _global_cache_lock        = ReentrantLock()

"""
    global_cache() -> Union{Cache, Nothing}

Return the lazily-initialized process-wide disk cache, or `nothing` if
initialization failed. The cache lives in cuTile's Scratch.jl-managed
scratchspace (so Pkg can clean it up when cuTile is uninstalled).
Failures are remembered; subsequent calls don't keep retrying.
"""
function global_cache()
    _global_cache_initialized[] && return _global_cache[]
    Base.@lock _global_cache_lock begin
        if !_global_cache_initialized[]
            _global_cache[] = _try_init()
            _global_cache_initialized[] = true
        end
    end
    return _global_cache[]
end

function _try_init()
    try
        # @get_scratch! resolves to cuTile's package UUID via moduleroot,
        # so the path is $DEPOT/scratchspaces/<cuTile-UUID>/disk_cache/.
        root = @get_scratch!("disk_cache")
        return open(root)
    catch err
        @debug "cuTile disk cache disabled" exception=(err, catch_backtrace())
        return nothing
    end
end

# ===========================================================================
# Eviction (designed, not yet implemented)
# ===========================================================================
#
# When `mdb_put` returns `MDB_MAP_FULL` (the env hit its mapsize cap),
# we need to free space. This module currently has no eviction logic;
# `put!` will surface MDB_MAP_FULL as an error which the launch site
# silently swallows (so launches keep working, just without caching new
# entries). The plan when we need to enable it:
#
# 1. Value layout: prepend an 8-byte little-endian `atime_ns` to every
#    stored blob. `compute_key` doesn't change. `get` strips the prefix
#    after copying.
# 2. atime refresh: on hit, if `time_ns() - atime > REFRESH_THRESHOLD`
#    (e.g. 1 day), rewrite the entry with a fresh atime in a tiny
#    write txn. Throttling avoids write-amplification on hot kernels.
# 3. Reactive eviction: catch MDB_MAP_FULL in `put!`, run
#    `evict_lru!(target_utilization=0.75)`, then retry the put once.
# 4. evict_lru!: cursor-walk all entries collecting (key, atime),
#    sort by atime ascending, delete oldest until env utilization is
#    below the target. O(N) per eviction; eviction is rare so this is
#    fine. Use `mdb_env_info` + `mdb_stat` to compute utilization.
#
# Migration: existing entries lacking the atime prefix can be detected
# (size < 8 or sentinel byte) and treated as "atime = 0" so they evict
# first. Or just bump _CACHE_SCHEMA, which strands old entries (they
# never match) and lets the next eviction cycle clean them up.

end # module DiskCache

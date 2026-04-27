using CUDA
using Random

# Bucket values in [lo, hi) into `n` bins; returns per-bin counts.
# Used for distribution-shape sanity checks (avoids the strict-uniqueness flake).
function bin_counts(v, n; lo=0.0, hi=1.0)
    counts = zeros(Int, n)
    span = hi - lo
    for x in v
        idx = clamp(floor(Int, (Float64(x) - lo) / span * n) + 1, 1, n)
        counts[idx] += 1
    end
    counts
end

# Natural [lo, hi) span per RNG output type, for the bin-counts check.
rand_span(::Type{T}) where {T<:AbstractFloat} = (0.0, 1.0)
rand_span(::Type{T}) where {T<:Unsigned}      = (0.0, Float64(typemax(T)) + 1.0)
rand_span(::Type{T}) where {T<:Signed}        = (Float64(typemin(T)), Float64(typemax(T)) + 1.0)

@testset "device rand" begin

@testset "typed rand surfaces (T=$T, dims=$dims)" for (T, dims) in
        ((Float32, (16,)), (Float32, (32,)),
         (UInt32, (16,)), (Int32,  (16,)),
         (UInt16, (16,)), (Int16,  (16,)),
         (UInt8,  (16,)), (Int8,   (16,)),
         (UInt64, (16,)), (Int64,  (16,)),
         (Float16, (16,)), (ct.BFloat16, (16,)), (Float64, (16,)))

    # Typed `rand` surfaces, polymorphic over (T, dims): scalar form, NTuple-typed
    # tile form, and variadic-Integer + explicit-RNG form. Each surface writes to
    # its own output array; the explicit-RNG path also exercises `DeviceRNG()`
    # stream allocation distinct from the default stream.
    function k_typed_surfaces(o1::ct.TileArray{T, 1},
                              o2::ct.TileArray{T, 1},
                              o3::ct.TileArray{T, 1},
                              ::Type{T}, dims::NTuple{N, Int}) where {T, N}
        pid = ct.bid(1)
        rng = ct.DeviceRNG(); Random.seed!(rng, 1)
        o1[pid] = rand(T)
        ct.store(o2, pid, rand(T, dims))
        ct.store(o3, pid, rand(rng, T, prod(dims)))
        return
    end

    n_blocks = 64
    m = prod(dims)
    o1 = CUDA.zeros(T, n_blocks)
    o2 = CUDA.zeros(T, n_blocks * m)
    o3 = CUDA.zeros(T, n_blocks * m)
    ct.launch(k_typed_surfaces, n_blocks, o1, o2, o3, T, ct.Constant(dims))

    for v in (Array(o1), Array(o2), Array(o3))
        @test eltype(v) === T
        if T <: AbstractFloat
            # Floats land in [0, 1]; Float32 keeps the (0, 1] convention but
            # Float16/BFloat16 can underflow Float32's 2^-33 minimum to 0.
            @test all(zero(T) .<= v .<= one(T))
        end
    end

    # Distribution-shape check: 4 bins on the type's natural range, ~N/4
    # expected per bin. With N=1024 the `> N/4 - 100` lower bound gives
    # >7σ margin (P(failure per bin) ≈ 3e-13).
    lo, hi = rand_span(T)
    counts = bin_counts(Array(o2), 4; lo, hi)
    @test minimum(counts) > length(o2) ÷ 4 - 100
end

@testset "untyped rand defaults to Float32" begin
    # Untyped surfaces all default to Float32 (overlay choice diverges from
    # stdlib's Float64). Bare `rand()`, `rand(NTuple)`, variadic + explicit RNG,
    # and the multi-dim variadic + reshape path are all routed through the
    # Float32 default.
    function k_untyped_surfaces(o1::ct.TileArray{Float32, 1},
                                o2::ct.TileArray{Float32, 1},
                                o3::ct.TileArray{Float32, 1},
                                o4::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        rng = ct.DeviceRNG(); Random.seed!(rng, 1)
        o1[pid] = rand()
        ct.store(o2, pid, rand((16,)))
        ct.store(o3, pid, rand(rng, 16))
        ct.store(o4, pid, ct.reshape(rand(Float32, 4, 4), (16,)))
        return
    end

    n_blocks = 64
    o1 = CUDA.zeros(Float32, n_blocks)
    o2 = CUDA.zeros(Float32, n_blocks * 16)
    o3 = CUDA.zeros(Float32, n_blocks * 16)
    o4 = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k_untyped_surfaces, n_blocks, o1, o2, o3, o4)
    for v in (Array(o1), Array(o2), Array(o3), Array(o4))
        @test all(0f0 .< v .< 1f0)
    end
    counts = bin_counts(Array(o2), 4; lo=0.0, hi=1.0)
    @test minimum(counts) > length(o2) ÷ 4 - 100
end

#=============================================================================
 Multi-stream / state-threading correctness in one kernel
=============================================================================#

@testset "multi-RNG kernel" begin
    # Single kernel exercising:
    #  - default RNG with two consecutive tile reads (counter advances)
    #  - default RNG inside a `for` loop (counter threaded through ForOp)
    #  - two unseeded `DeviceRNG()` streams (stream-ID mix in `rng_key` keeps
    #    them independent even with a shared `KernelState.seed`)
    #  - one seeded `DeviceRNG()` stream (explicit seed overrides KernelState)
    #
    # Outputs are UInt32 so the strict uniqueness and pairwise-disjoint asserts
    # below are flake-free (2^32 buckets vs Float32's 2^24 in (0, 1)).
    function k_multi(d1::ct.TileArray{UInt32, 1},
                     d2::ct.TileArray{UInt32, 1},
                     a_out::ct.TileArray{UInt32, 1},
                     b_out::ct.TileArray{UInt32, 1},
                     c_out::ct.TileArray{UInt32, 1},
                     loop_out::ct.TileArray{UInt32, 1})
        pid = ct.bid(1)

        # Default RNG: no in-kernel seed → KernelState.seed reaches it. Two
        # consecutive tile reads must produce disjoint output (counter advance).
        ct.store(d1, pid, rand(UInt32, (16,)))
        ct.store(d2, pid, rand(UInt32, (16,)))

        # Two unseeded DeviceRNGs: stream-ID mix in `rng_key` keeps them
        # uncorrelated even though they share the per-launch KernelState.seed.
        a = ct.DeviceRNG()
        b = ct.DeviceRNG()
        ct.store(a_out, pid, rand(a, UInt32, (16,)))
        ct.store(b_out, pid, rand(b, UInt32, (16,)))

        # Seeded DeviceRNG: the explicit seed overrides the KernelState seed,
        # so this output is byte-identical across launches.
        c = ct.DeviceRNG()
        Random.seed!(c, UInt32(0xDEADBEEF))
        ct.store(c_out, pid, rand(c, UInt32, (16,)))

        # Default RNG inside a `for` loop: `rng_state_pass!` threads the counter
        # through the ForOp so each iteration sees a distinct counter.
        for i in Int32(1):Int32(4)
            loop_out[(pid - Int32(1)) * Int32(4) + i] = rand(UInt32)
        end
        return
    end

    n_blocks = 8
    alloc() = (d1   = CUDA.zeros(UInt32, n_blocks * 16),
               d2   = CUDA.zeros(UInt32, n_blocks * 16),
               a    = CUDA.zeros(UInt32, n_blocks * 16),
               b    = CUDA.zeros(UInt32, n_blocks * 16),
               c    = CUDA.zeros(UInt32, n_blocks * 16),
               loop = CUDA.zeros(UInt32, n_blocks * 4))
    out1 = alloc()
    ct.launch(k_multi, n_blocks, out1.d1, out1.d2, out1.a, out1.b, out1.c, out1.loop)
    arr1 = map(Array, out1)

    # Within a launch: every output's elements are fully unique. The loop
    # output's full uniqueness verifies per-iteration counter divergence.
    for v in arr1
        @test length(unique(v)) == length(v)
    end

    # Stream isolation: every pair of output sets is disjoint.
    slot_names = (:d1, :d2, :a, :b, :c, :loop)
    for i in 1:length(slot_names), j in (i + 1):length(slot_names)
        @test isempty(intersect(Set(arr1[slot_names[i]]), Set(arr1[slot_names[j]])))
    end

    # Cross-launch: KernelState.seed is fresh per launch, so unseeded streams
    # diverge — proving the host seed reaches the default stream, both
    # `DeviceRNG()` streams, and the in-loop default usage. The
    # explicitly-seeded `c` stream stays byte-identical.
    out2 = alloc()
    ct.launch(k_multi, n_blocks, out2.d1, out2.d2, out2.a, out2.b, out2.c, out2.loop)
    arr2 = map(Array, out2)

    @test arr1.d1   != arr2.d1
    @test arr1.d2   != arr2.d2
    @test arr1.a    != arr2.a
    @test arr1.b    != arr2.b
    @test arr1.loop != arr2.loop
    @test arr1.c    == arr2.c
end

#=============================================================================
 In-kernel `Random.seed!`
=============================================================================#

@testset "in-kernel `Random.seed!` is deterministic across launches" begin
    function k(out::ct.TileArray{Float32, 1}, seed::UInt32)
        Random.seed!(Random.default_rng(), seed)
        pid = ct.bid(1)
        ct.store(out, pid, rand(Float32, (16,)))
        return
    end
    n_blocks = 16
    out1 = CUDA.zeros(Float32, n_blocks * 16)
    out2 = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, out1, UInt32(42))
    ct.launch(k, n_blocks, out2, UInt32(42))
    @test Array(out1) == Array(out2)
end

@testset "in-kernel `Random.seed!` matches host RNG output" begin
    # In-kernel `Random.seed!(default_rng(), s)` and host-side
    # `cuTile.RNG(s, 0)` both feed the same threaded `(seed, counter)`
    # state, so a single-block draw of equal size must be byte-identical.
    function k(out::ct.TileArray{Float32, 1})
        Random.seed!(Random.default_rng(), UInt32(42))
        pid = ct.bid(1)
        ct.store(out, pid, rand(Float32, (512,)))
        return
    end
    out = CUDA.zeros(Float32, 512); ct.launch(k, 1, out)
    expected = Array(Random.rand(ct.RNG(UInt32(42), UInt32(0)), Float32, 512))
    @test Array(out) == expected
end

end


# Distribution config used to share the randn/randexp tests below. `edges`
# are 3 quartile-or-equivalent CDF cuts; `probs` are the matching per-bin
# probabilities; `mean` is the population mean; `in_range` is the per-element
# validity predicate (e.g. exponential support is `[0, ∞)`).
const RAND_DISTS = (
    (label = "randn",
     f     = Random.randn,
     f!    = Random.randn!,
     edges = (-1.0, 0.0, 1.0),
     probs = (0.1587, 0.3413, 0.3413, 0.1587),
     mean  = 0.0,
     in_range = isfinite),

    (label = "randexp",
     f     = Random.randexp,
     f!    = Random.randexp!,
     edges = (-log(0.75), -log(0.5), -log(0.25)),
     probs = (0.25, 0.25, 0.25, 0.25),
     mean  = 1.0,
     in_range = >=(0.0)),
)

function bin_counts(v, edges)
    counts = zeros(Int, length(edges) + 1)
    for x in v
        idx = something(findfirst(>=(Float64(x)), edges), length(edges) + 1)
        counts[idx] += 1
    end
    counts
end

# Per-bin tolerance: 10% of N. Smallest bin under either CDF (p≈0.16 for
# normal, p=0.25 for exp) clears 5σ at N≥1024, so the check is flake-free
# across the per-launch-randomized default-stream seed.
function check_dist(v, dist; N = length(v), tol_factor = 0.10)
    counts = bin_counts(v, dist.edges)
    for (i, p) in enumerate(dist.probs)
        @test abs(counts[i] - N * p) < N * tol_factor
    end
end

# Mean tolerance: SE/√N at N=1024 is ≤ 0.031; 5σ ≈ 0.16. Narrow floats lose
# precision in the BM/log path, so widen for those.
mean_tol(::Type{T}) where {T} =
    T <: Union{Float16, ct.BFloat16} ? 0.20 : 0.15

@testset "device $(dist.label)" for dist in RAND_DISTS
    f, in_range = dist.f, dist.in_range

    @testset "typed surfaces (T=$T, dims=$dims)" for (T, dims) in
            ((Float32, (16,)), (Float32, (32,)), (Float32, (64,)),
             (Float64, (16,)), (Float16, (16,)), (ct.BFloat16, (16,)))
        # `o1`: scalar form. `o2`: tile form via the default stream.
        # `o3`: tile form via an explicit `DeviceRNG` (different stream).
        function k(o1, o2, o3, ::Type{T_}, dims_::NTuple{N, Int}) where {T_, N}
            pid = ct.bid(1)
            rng = ct.DeviceRNG(); Random.seed!(rng, 1)
            o1[pid] = f(T_)
            ct.store(o2, pid, f(T_, dims_))
            ct.store(o3, pid, f(rng, T_, prod(dims_)))
            return
        end

        n_blocks = 64
        m = prod(dims)
        o1 = CUDA.zeros(T, n_blocks)
        o2 = CUDA.zeros(T, n_blocks * m)
        o3 = CUDA.zeros(T, n_blocks * m)
        ct.launch(k, n_blocks, o1, o2, o3, T, ct.Constant(dims))

        for v in (Array(o1), Array(o2), Array(o3))
            @test eltype(v) === T
            @test all(isfinite, v)
            @test all(in_range, v)
        end

        # Distribution shape + mean on the larger draws (o1's N=64 is too
        # small for shape testing).
        for v in (Array(o2), Array(o3))
            check_dist(v, dist)
            @test abs(sum(Float64, v) / length(v) - dist.mean) < mean_tol(T)
        end
    end

    @testset "untyped surface defaults to Float32" begin
        function k(o::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            ct.store(o, pid, ct.reshape(f(4, 4), (16,)))
            return
        end
        n_blocks = 64
        o = CUDA.zeros(Float32, n_blocks * 16)
        ct.launch(k, n_blocks, o)
        v = Array(o)
        @test eltype(v) === Float32
        @test all(in_range, v)
        check_dist(v, dist)
    end

    @testset "in-kernel `Random.seed!` matches host RNG output" begin
        # Single-block draw with an in-kernel `Random.seed!(default_rng(),s)`
        # plumbs the same `(seed, counter)` as `cuTile.RNG(s, 0)`, so the
        # outputs must be byte-identical.
        function k(out::ct.TileArray{Float32, 1})
            Random.seed!(Random.default_rng(), UInt32(42))
            pid = ct.bid(1)
            ct.store(out, pid, f(Float32, (512,)))
            return
        end
        out = CUDA.zeros(Float32, 512); ct.launch(k, 1, out)
        @test Array(out) == Array(f(ct.RNG(UInt32(42), UInt32(0)), Float32, 512))
    end
end


@testset "host $(dist.label)" for dist in RAND_DISTS
    f, f!, in_range = dist.f, dist.f!, dist.in_range
    N = 4096

    @testset "$(dist.label)! basics + counter advance" begin
        rng = ct.RNG(42)
        A = CUDA.zeros(Float32, N)
        f!(rng, A)
        v = Array(A)
        @test all(isfinite, v)
        @test all(in_range, v)
        @test rng.counter == UInt32(N)
        @test abs(sum(v) / N - dist.mean) < 0.1
    end

    @testset "determinism + Philox re-keying" begin
        # Same seed → byte-identical output.
        @test Array(f(ct.RNG(123), Float32, N)) == Array(f(ct.RNG(123), Float32, N))

        # Different seeds → uncorrelated streams. Set-disjoint isn't safe
        # for either distribution at this N due to Float32 birthday flake
        # (especially randexp, whose output concentrates near 0); use
        # element-wise equality instead — uncorrelated streams collide in
        # ≤ N²/2^24 ≈ 1 position.
        c = Array(f(ct.RNG(42),  Float32, N))
        d = Array(f(ct.RNG(100), Float32, N))
        @test sum(c .== d) < N ÷ 100
    end

    @testset "consecutive disjoint; seed! resets counter" begin
        # Two back-to-back draws on the same RNG use disjoint counter
        # ranges, so the underlying Philox outputs (and post-transform
        # samples) are disjoint up to Float32 birthday flake.
        rng = ct.RNG(7)
        a = Array(f(rng, Float32, N))
        b = Array(f(rng, Float32, N))
        @test sum(a .== b) < N ÷ 100

        Random.seed!(rng, 7)
        @test rng.counter == UInt32(0)
    end

    @testset "T-coverage" for T in (Float16, ct.BFloat16, Float32, Float64)
        v = Array(f(ct.RNG(13), T, N))
        @test eltype(v) === T
        @test all(in_range, v)
        @test all(isfinite, v)
        @test abs(sum(Float64, v) / N - dist.mean) < mean_tol(T)
        check_dist(v, dist)
    end

    @testset "ct.$(dist.label) / ct.$(dist.label)! aliases default to Float32" begin
        ct_f, ct_f! = dist.label == "randn" ? (ct.randn, ct.randn!) :
                                              (ct.randexp, ct.randexp!)
        @test eltype(ct_f(N)) === Float32
        @test eltype(ct_f(Float32, N)) === Float32
        B = CUDA.zeros(Float32, N); ct_f!(B)
        @test all(in_range, Array(B))
    end

    @testset "arbitrary length (partial last tile)" begin
        # `store_partition_view` clips OOB writes, so `length(A)` can be
        # any value. The host advances by `n_blocks * RAND_FILL_TILE`,
        # not `length(A)`, so consecutive partial-length calls remain
        # disjoint up to Float32 birthday flake.
        rng = ct.RNG(0)
        A = CUDA.zeros(Float32, 513)
        f!(rng, A)
        v = Array(A)
        @test all(isfinite, v)
        @test rng.counter == UInt32(2 * cuTile.RAND_FILL_TILE)

        rng2 = ct.RNG(0)
        a = Array(f(rng2, Float32, 100))
        b = Array(f(rng2, Float32, 100))
        @test sum(a .== b) < 100 ÷ 10
    end
end


@testset "host rand" begin
    N = 2048

    @testset "rand! basics + auto-counter advance" begin
        # UInt32 output for the strict-uniqueness check (Float32 in-range
        # coverage lives below in the ct.rand alias testset).
        rng = ct.RNG(42)
        A = CUDA.zeros(UInt32, N)
        Random.rand!(rng, A)
        v = Array(A)
        @test length(unique(v)) == N
        @test rng.counter == UInt32(N)
    end

    @testset "determinism + Philox re-keying" begin
        # Same seed → byte-identical output. Robust regardless of T.
        a = Array(Random.rand(ct.RNG(123), Float32, N))
        b = Array(Random.rand(ct.RNG(123), Float32, N))
        @test a == b

        # Different seeds re-key Philox (rather than offsetting the counter),
        # so no shift produces equality. The shifted-equality check is robust
        # for Float32 (it requires N-k consecutive matching values, which is
        # vastly more stringent than birthday collisions). The set-disjoint
        # check is on UInt32 to avoid Float32 birthday flake (~22% at N=2048).
        c_f = Array(Random.rand(ct.RNG(42),  Float32, N))
        d_f = Array(Random.rand(ct.RNG(100), Float32, N))
        @test !any(k -> c_f[k+1:N] == d_f[1:N-k], 1:64)

        c_u = Array(Random.rand(ct.RNG(42),  UInt32, N))
        d_u = Array(Random.rand(ct.RNG(100), UInt32, N))
        @test isempty(intersect(Set(c_u), Set(d_u)))
    end

    @testset "consecutive rand! disjoint; seed! resets counter" begin
        rng = ct.RNG(7)
        A1 = Array(Random.rand(rng, UInt32, N))
        A2 = Array(Random.rand(rng, UInt32, N))
        @test isempty(intersect(Set(A1), Set(A2)))

        Random.seed!(rng, 7)
        @test rng.counter == UInt32(0)
    end

    @testset "UInt32 output: sample mean" begin
        rng = ct.RNG(99)
        v = Array(Random.rand(rng, UInt32, N))
        @test eltype(v) === UInt32
        # SE/mean ≈ 1.25% at N=2048; 5% tolerance is ~4σ.
        target = Float64(typemax(UInt32)) / 2
        @test abs(sum(Float64.(v))/length(v) - target) / target < 0.05
    end

    @testset "ct.rand / ct.rand! aliases default to Float32" begin
        # Float32 in-range check; this is the host wrapper's only Float32
        # property test (the device Float32 path is covered by the surfaces
        # testsets above).
        @test all(0f0 .< Array(ct.rand(Float32, N)) .< 1f0)
        B = CUDA.zeros(Float32, N); ct.rand!(B)
        @test all(0f0 .< Array(B) .< 1f0)
        @test eltype(ct.rand(N)) === Float32
    end

    @testset "host rand! covers RandTypes (T=$T)" for T in
            (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
             Float16, ct.BFloat16, Float32, Float64)
        # End-to-end host fill for every supported element type. Verifies the
        # generic `rand_fill_kernel` specializes correctly per T and produces
        # values in the type's natural range (in-range is trivial for
        # integers, [0, 1] for floats).
        v = Array(Random.rand(ct.RNG(13), T, N))
        @test eltype(v) === T
        if T <: AbstractFloat
            @test all(zero(T) .<= v .<= one(T))
        end
        lo, hi = rand_span(T)
        counts = bin_counts(v, 4; lo, hi)
        @test minimum(counts) > N ÷ 4 - 100
    end

    @testset "arbitrary length (partial last tile)" begin
        # `store_partition_view` clips OOB writes, so `length(A)` can be any
        # value. The host advance rounds up to a full RAND_FILL_TILE per
        # block, so consecutive partial-length calls remain disjoint.
        # UInt32 output keeps the strict-uniqueness and set-disjoint checks
        # flake-free.
        rng = ct.RNG(0)
        A = CUDA.zeros(UInt32, 513)
        Random.rand!(rng, A)
        v = Array(A)
        @test length(unique(v)) == 513
        @test rng.counter == UInt32(2 * cuTile.RAND_FILL_TILE)

        rng2 = ct.RNG(0)
        A1 = Array(Random.rand(rng2, UInt32, 100))
        A2 = Array(Random.rand(rng2, UInt32, 100))
        @test isempty(intersect(Set(A1), Set(A2)))
    end

    @testset "advance_counter! bumps seed on UInt32 wrap" begin
        rng = ct.RNG(UInt32(7), typemax(UInt32) - UInt32(3))
        cuTile.advance_counter!(rng, UInt32(10))
        @test rng.counter == UInt32(6)
        @test rng.seed    == UInt32(8)
    end
end

# Shared TileArray field-access decoder.
#
# `TileArray` carries its static facts (alignment, contiguity, divisibility)
# in the `ArraySpec` type parameter, but the runtime fields (`ptr`, `sizes`,
# `strides`) are opaque to Julia's compiler. The dataflow analyses therefore
# need to recognise the `getfield` chains that read those fields and project
# the spec's facts onto the appropriate lattice.
#
# Three analyses do this — divisibility, bounds, constant — each previously
# duplicated the same chain walker. This file collects it in one place:
# `decode_tilearray_field` walks the `getfield(...)` operands and returns a
# `TileArrayFieldRef` describing which field of which `ArraySpec` is being
# read; each analysis then projects to its own lattice.
#
# A future PartialStruct-style refactor would push this further: structured
# lattice values seeded at `init_arg`, generic getfield/tuple transfer rules.
# That generalises to user-written `Core.tuple → getfield` patterns inside
# kernels, but at the cost of recursive lattice values across the framework.
# Until a workload demands that, the per-analysis projection here is the
# pragmatic shape — TileArray is the only type whose facts live in a type
# parameter rather than in the lattice itself.

"""
    TileArrayFieldRef

The result of decoding a `getfield` chain rooted at a TileArray-typed
`Argument`. Carries:

- `spec::ArraySpec` — the spec from the TileArray type parameter.
- `field::Symbol` — `:ptr`, `:sizes`, or `:strides`.
- `index::Union{Int,Nothing}` — `nothing` for whole-tuple / pointer reads
  (`getfield(arg, :sizes)`, `getfield(arg, :ptr)`); a 1-based positive
  integer for element reads (`getfield(getfield(arg, :sizes), i)`).
"""
struct TileArrayFieldRef
    spec::ArraySpec
    field::Symbol
    index::Union{Int, Nothing}
end

"""
    decode_tilearray_field(block, ops) -> Union{TileArrayFieldRef, Nothing}

Decode the operands of a `Base.getfield(...)` call as a TileArray field
reference. Caller is expected to have already established that the current
call is `Base.getfield`. Recognises two shapes:

- `getfield(arg::TileArray, :ptr | :sizes | :strides)` — returns the spec
  with `index = nothing`.
- `getfield(getfield(arg::TileArray, :sizes | :strides), i)` — returns the
  spec with `field` set to the inner `:sizes` / `:strides` and `index = i`.

Returns `nothing` for any other shape (non-TileArray base, missing spec,
non-integer / non-positive element index, opaque inner producer).
"""
function decode_tilearray_field(block::Block, ops)
    length(ops) >= 2 || return nothing
    field = ops[2] isa QuoteNode ? ops[2].value : ops[2]
    obj = ops[1]

    obj_T = value_type(block, obj)
    obj_T = obj_T === nothing ? Any : CC.widenconst(obj_T)

    # First-level: getfield(arg::TileArray, :ptr | :sizes | :strides)
    if obj_T <: TileArray
        spec = array_spec(obj_T)
        spec === nothing && return nothing
        field isa Symbol || return nothing
        return TileArrayFieldRef(spec, field, nothing)
    end

    # Second-level: getfield(getfield(arg::TileArray, :sizes|:strides), i).
    # Walk back through `obj`'s defining call to find the inner getfield.
    obj isa SSAValue || return nothing
    inner_def = lookup_def_call(block, obj)
    inner_def === nothing && return nothing
    inner_func, inner_ops = inner_def
    inner_func === Base.getfield || return nothing
    length(inner_ops) >= 2 || return nothing

    inner_field = inner_ops[2] isa QuoteNode ? inner_ops[2].value : inner_ops[2]
    (inner_field === :sizes || inner_field === :strides) || return nothing

    inner_T = value_type(block, inner_ops[1])
    inner_T = inner_T === nothing ? Any : CC.widenconst(inner_T)
    inner_T <: TileArray || return nothing
    spec = array_spec(inner_T)
    spec === nothing && return nothing

    field isa Integer || return nothing
    idx = Int(field)
    idx >= 1 || return nothing

    return TileArrayFieldRef(spec, inner_field, idx)
end

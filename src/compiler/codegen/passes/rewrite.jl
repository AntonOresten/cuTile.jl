# Declarative IR Rewrite Pattern Framework
#
# Inspired by MLIR's PDLL. Patterns compile into pattern/rewrite node trees.
# The framework handles matching (recursive SSA def-chain walking) and rewrite
# application. Cleanup of dead code is delegated to the pipeline's dce_pass!.
#
# Usage:
#   rules = RewriteRule[
#       @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
#               Intrinsics.fma(~x, ~y, ~z)
#       @rewrite Core.Intrinsics.slt_int(~x, ~y) =>
#               Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))
#   ]
#   rewrite_patterns!(sci, rules)

using Core: SSAValue

#=============================================================================
 Pattern & Rewrite Nodes
=============================================================================#

abstract type PatternNode end
struct PCall <: PatternNode; func::Any; operands::Vector{PatternNode}; end
struct PBind <: PatternNode; name::Symbol; end
struct PTypedBind <: PatternNode; name::Symbol; type::Type; end
struct POneUse <: PatternNode; inner::PatternNode; end

abstract type RewriteNode end
struct RCall <: RewriteNode; func::Any; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end
struct RConst <: RewriteNode; val::Any; end

"""
    RFunc(func)

Imperative rewrite node (MLIR-inspired). The function is called with
`(sci, block, inst, match, ctx)` and returns `true` if the rewrite was applied,
`false` to skip this rule and try the next one. `ctx` is the `MatchContext`
providing def-chain lookups via `ctx.defs`.
"""
struct RFunc <: RewriteNode; func::Function; end

struct RewriteRule
    lhs::PCall
    rhs::RewriteNode
    guard::Union{Function, Nothing}  # (match, ctx) -> Bool, or nothing
end
RewriteRule(lhs::PCall, rhs::RewriteNode) = RewriteRule(lhs, rhs, nothing)

root_func(rule::RewriteRule) = rule.lhs.func

#=============================================================================
 @rewrite / @rewriter Macros
=============================================================================#

"""
    @rewrite lhs => rhs
    @rewrite(lhs => rhs, guard)

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls,
`~x` binds (repeated names require equality), `~x::T` binds with type constraint,
`one_use(pat)` requires single use. RHS: `func(args...)` emits calls,
`~x` references bindings, `\$(expr)` injects a literal constant.
Optional `guard` is a function `(match, ctx) -> Bool` checked after pattern match.
"""
macro rewrite(ex, guard=nothing)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    g = guard === nothing ? :nothing : guard
    esc(:(RewriteRule($(_compile_lhs(ex.args[2])), $(_compile_rhs(ex.args[3])), $g)))
end

"""
    @rewriter lhs => func

Declarative pattern with imperative rewrite. LHS uses the same pattern syntax as
`@rewrite`. RHS is a function `(sci, block, inst, match, ctx) -> Bool` that
performs the rewrite and returns `true`, or returns `false` to skip and try the
next rule. Matched bindings are in `match.bindings`; def-chain lookups via
`ctx.defs`.
"""
macro rewriter(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewriter expects: lhs => func")
    esc(:(RewriteRule($(_compile_lhs(ex.args[2])), RFunc($(ex.args[3])))))
end

function _compile_lhs(ex)
    ex isa Expr && ex.head === :call || error("@rewrite LHS: expected call, got $ex")
    f = ex.args[1]
    if f === :~
        inner = ex.args[2]
        if inner isa Expr && inner.head === :(::)
            return :(PTypedBind($(QuoteNode(inner.args[1])), $(inner.args[2])))
        end
        return :(PBind($(QuoteNode(inner))))
    end
    f === :one_use && return :(POneUse($(_compile_lhs(ex.args[2]))))
    :(PCall($f, PatternNode[$(_compile_lhs.(ex.args[2:end])...)]))
end

function _compile_rhs(ex)
    if ex isa Expr && ex.head === :$
        return :(RConst($(ex.args[1])))
    end
    ex isa Expr && ex.head === :call || error("@rewrite RHS: expected call or \$const, got $ex")
    f = ex.args[1]
    f === :~ && return :(RBind($(QuoteNode(ex.args[2]))))
    :(RCall($f, RewriteNode[$(_compile_rhs.(ex.args[2:end])...)]))
end

#=============================================================================
 Matching
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}
    matched_ssas::Vector{Int}
end

struct DefEntry
    block::Block
    inst::Instruction
    func::Any
end

"""Operands of a DefEntry, read from the live instruction."""
function _def_operands(entry::DefEntry)
    call = resolve_call(entry.inst)
    call === nothing && return Any[]
    _, ops = call
    return ops
end

struct MatchContext
    entry::Block      # root block of the StructuredIRCode (for value_type lookups)
    defs::Dict{Int, DefEntry}
    use_index  # UseIndex from IRStructurizer
end

function MatchContext(sci::StructuredIRCode)
    defs = Dict{Int, DefEntry}()
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(inst)
            call === nothing && continue
            func, _ = call
            defs[inst.ssa_idx] = DefEntry(block, inst, func)
        end
    end
    MatchContext(sci.entry, defs, uses(sci.entry))
end

_use_count(ctx::MatchContext, val::SSAValue) =
    haskey(ctx.use_index, val) ? length(ctx.use_index[val]) : 0

# Codegen no-ops that pattern matching traces through transparently.
_is_transparent(func) = func === Intrinsics.to_scalar ||
                         func === Intrinsics.from_scalar ||
                         func === Intrinsics.broadcast

"""Merge bindings, requiring repeated names to bind the same value (=== equality)."""
function _merge_bindings!(dest::Dict{Symbol,Any}, src::Dict{Symbol,Any})
    for (k, v) in src
        if haskey(dest, k)
            dest[k] === v || return false
        else
            dest[k] = v
        end
    end
    return true
end

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::PCall,
                       block::Block=ctx.entry)
    val isa SSAValue || return nothing
    entry = get(ctx.defs, val.id, nothing)
    entry === nothing && return nothing

    if entry.func === pat.func
        ops = _def_operands(entry)
        if length(ops) == length(pat.operands)
            result = MatchResult(Dict{Symbol,Any}(), Int[val.id])
            for (op, sub) in zip(ops, pat.operands)
                m = pattern_match(ctx, op, sub, entry.block)
                m === nothing && return nothing
                _merge_bindings!(result.bindings, m.bindings) || return nothing
                append!(result.matched_ssas, m.matched_ssas)
            end
            return result
        end
    end

    # Trace through single-use transparent ops to find the underlying operation
    if _is_transparent(entry.func)
        _use_count(ctx, val) == 1 || return nothing
        ops = _def_operands(entry)
        isempty(ops) && return nothing
        if entry.func === Intrinsics.broadcast
            inner = ops[1]
            if inner isa SSAValue
                inner_entry = get(ctx.defs, inner.id, nothing)
                if inner_entry !== nothing
                    it, ot = value_type(inner_entry.inst), value_type(entry.inst)
                    it <: Tile && ot <: Tile || return nothing
                    size(it) == size(ot) || return nothing
                end
            end
        end
        result = pattern_match(ctx, ops[1], pat, entry.block)
        result === nothing && return nothing
        push!(result.matched_ssas, val.id)
        return result
    end
    return nothing
end

pattern_match(ctx::MatchContext, @nospecialize(val), pat::PBind, block::Block=ctx.entry) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), Int[])

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::PTypedBind,
                       block::Block=ctx.entry)
    T = value_type(block, val)
    T === nothing && return nothing
    CC.widenconst(T) <: pat.type || return nothing
    MatchResult(Dict{Symbol,Any}(pat.name => val), Int[])
end

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::POneUse,
                       block::Block=ctx.entry)
    val isa SSAValue && _use_count(ctx, val) == 1 || return nothing
    pattern_match(ctx, val, pat.inner, block)
end

#=============================================================================
 Rewrite Application
=============================================================================#

"""Resolve an RHS operand, inserting sub-calls before `ref` as needed."""
_resolve_rhs(block, ref, op::RBind, bindings, typ) = bindings[op.name]
_resolve_rhs(block, ref, op::RConst, bindings, typ) = op.val
function _resolve_rhs(block, ref, op::RCall, bindings, typ)
    operands = Any[_resolve_rhs(block, ref, sub, bindings, typ) for sub in op.operands]
    SSAValue(insert_before!(block, ref, Expr(:call, op.func, operands...), typ))
end

function _apply_rewrite!(sci, block, inst, rule, match, ctx, consumed)
    if rule.rhs isa RFunc
        rule.rhs.func(sci, block, inst, match, ctx) || return false
        return true
    elseif rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root
        replace_uses!(sci.entry, SSAValue(inst), match.bindings[rule.rhs.name])
        delete!(block, inst)
        _cleanup_dead_operands!(sci, inst.ssa_idx, ctx, consumed)
    else
        # Substitution: delete matched intermediates, replace root statement in-place.
        for dead_ssa in match.matched_ssas
            dead_ssa == inst.ssa_idx && continue
            entry = ctx.defs[dead_ssa]
            delete!(entry.block, entry.inst)
        end
        typ = value_type(inst)
        operands = Any[_resolve_rhs(block, SSAValue(inst), op, match.bindings, typ)
                       for op in rule.rhs.operands]
        pos = findfirst(==(inst.ssa_idx), block.body.ssa_idxes)
        block.body.stmts[pos] = Expr(:call, rule.rhs.func, operands...)
    end
end

"""Recursively erase dead pure operands after an instruction is deleted (MLIR-style)."""
function _cleanup_dead_operands!(sci, ssa_idx, ctx, consumed)
    entry = get(ctx.defs, ssa_idx, nothing)
    entry === nothing && return
    for op in _def_operands(entry)
        op isa SSAValue || continue
        op_entry = get(ctx.defs, op.id, nothing)
        op_entry === nothing && continue
        _is_transparent(op_entry.func) || continue
        op.id in consumed && continue
        isempty(uses(sci.entry, op)) || continue
        push!(consumed, op.id)
        delete!(op_entry.block, op_entry.inst)
        _cleanup_dead_operands!(sci, op.id, ctx, consumed)
    end
end

#=============================================================================
 Driver
=============================================================================#

"""
    rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule})

Apply declarative rewrite rules to the structured IR. Dead code left behind
is cleaned up by the pipeline's `dce_pass!`.
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule})
    ctx = MatchContext(sci)
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    consumed = Set{Int}()
    for block in eachblock(sci)
        for inst in collect(instructions(block))
            inst.ssa_idx in consumed && continue
            call = resolve_call(inst)
            call === nothing && continue
            applicable = get(dispatch, call[1], nothing)
            applicable === nothing && continue
            for rule in applicable
                m = pattern_match(ctx, SSAValue(inst), rule.lhs)
                m === nothing && continue
                any(s in consumed for s in m.matched_ssas) && continue
                rule.guard !== nothing && !rule.guard(m, ctx) && continue
                result = _apply_rewrite!(sci, block, inst, rule, m, ctx, consumed)
                result === false && continue  # RFunc declined, try next rule
                # RBind only deletes the root; intermediates stay live and
                # must remain matchable. Substitution deletes intermediates
                # so they must be consumed to prevent the driver revisiting.
                if rule.rhs isa RBind
                    push!(consumed, inst.ssa_idx)
                else
                    union!(consumed, m.matched_ssas)
                end
                break
            end
        end
    end
end

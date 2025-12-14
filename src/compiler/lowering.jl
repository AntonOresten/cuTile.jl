# Lowering: Julia IR -> Structured IR
#
# Provides the main entry point for converting Julia CodeInfo to StructuredCodeInfo.

#=============================================================================
 Main Entry Point
=============================================================================#

"""
    lower_to_structured_ir(code::CodeInfo) -> StructuredCodeInfo

Convert Julia CodeInfo to StructuredCodeInfo.

For straight-line code (no control flow), this creates a single block containing
all statements. For code with control flow, it builds a control tree and converts
it to nested structured control flow.
"""
function lower_to_structured_ir(code::CodeInfo)
    stmts = code.code
    n = length(stmts)

    if n == 0
        # Empty function
        entry = Block(1)
        return StructuredCodeInfo(code, entry)
    end

    # Check if the code is straight-line (no control flow)
    if is_straight_line(code)
        return lower_straight_line(code)
    else
        return lower_with_control_flow(code)
    end
end

"""
    lower_to_structured_ir(target::TileTarget) -> StructuredCodeInfo

Convert a TileTarget to StructuredCodeInfo.
"""
function lower_to_structured_ir(target::TileTarget)
    lower_to_structured_ir(target.ci)
end

#=============================================================================
 Straight-Line Code
=============================================================================#

"""
    is_straight_line(code::CodeInfo) -> Bool

Check if the code contains no control flow (no branches, just return at end).
"""
function is_straight_line(code::CodeInfo)
    stmts = code.code
    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            return false
        elseif stmt isa GotoIfNot
            return false
        elseif stmt isa ReturnNode
            # Return is only allowed as the last statement for straight-line
            if i != length(stmts)
                return false
            end
        end
    end
    return true
end

"""
    lower_straight_line(code::CodeInfo) -> StructuredCodeInfo

Lower straight-line code (no control flow) to structured IR.
Simply places all non-control-flow statements in a single block.
"""
function lower_straight_line(code::CodeInfo)
    stmts = code.code
    n = length(stmts)

    entry = Block(1)

    for i in 1:n
        stmt = stmts[i]
        if stmt isa ReturnNode
            # Set as terminator, don't include in stmts
            entry.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot)
            push!(entry.stmts, i)
        end
    end

    return StructuredCodeInfo(code, entry)
end

#=============================================================================
 Control Flow Code
=============================================================================#

"""
    lower_with_control_flow(code::CodeInfo) -> StructuredCodeInfo

Lower code with control flow using block-level analysis and loop detection.
Falls back to a flat representation if restructuring fails.
"""
function lower_with_control_flow(code::CodeInfo)
    try
        # Build block-level CFG and detect loops
        blocks = build_block_cfg(code)
        loops = find_loops(blocks)

        if !isempty(loops)
            # Use block-level structuring with loop detection
            return lower_with_loops(code, blocks, loops)
        else
            # No loops - use original approach for if-then-else patterns
            cfg = julia_cfg(code)
            control_tree = build_control_tree(cfg, code)
            return control_tree_to_structured_ir(control_tree, code)
        end
    catch e
        # Fallback: create flat IR preserving all statements
        @warn "Control flow restructuring failed, using flat representation" exception=e
        return lower_flat(code)
    end
end

"""
    lower_with_loops(code::CodeInfo, blocks::Vector{BlockInfo}, loops::Vector{LoopInfo}) -> StructuredCodeInfo

Lower code that contains loops using block-level analysis.
"""
function lower_with_loops(code::CodeInfo, blocks::Vector{BlockInfo}, loops::Vector{LoopInfo})
    stmts = code.code

    # For now, handle the common case: single loop with optional pre/post code
    # Entry block → Loop (header + body) → Exit block
    @assert length(loops) == 1 "Multiple loops not yet supported"
    loop = loops[1]

    entry = Block(1)
    block_id = Ref(2)

    # Process blocks before the loop (entry blocks)
    entry_blocks = filter(bi -> bi ∉ loop.blocks, 1:loop.header-1)
    for bi in entry_blocks
        collect_block_stmts!(entry, blocks[bi], code)
    end

    # Create the LoopOp
    loop_op = build_loop_op(code, blocks, loop, block_id)
    push!(entry.nested, loop_op)

    # Process blocks after the loop (exit blocks)
    # The loop's exit blocks contain the code after the loop
    for exit_bi in sort(collect(loop.exit_blocks))
        block = blocks[exit_bi]
        for si in block.range
            stmt = stmts[si]
            if stmt isa ReturnNode
                entry.terminator = stmt
            elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
                push!(entry.stmts, si)
            end
        end
    end

    return StructuredCodeInfo(code, entry)
end

"""
    build_loop_op(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo, block_id::Ref{Int}) -> LoopOp

Build a LoopOp from loop information.
"""
function build_loop_op(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo, block_id::Ref{Int})
    stmts = code.code
    header_block = blocks[loop.header]

    # Find phi nodes in header - these become loop-carried values and results
    init_values = IRValue[]
    carried_values = IRValue[]
    block_args = BlockArg[]
    result_vars = SSAValue[]  # The SSAValues that the loop produces (the phi nodes)

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            # The phi's SSAValue becomes a result of the loop
            push!(result_vars, SSAValue(si))
            # Extract initial value (from entry edge) and carried value (from back edge)
            phi = stmt
            edges = phi.edges
            values = phi.values

            # Determine which edges are from inside the loop (back edges) vs outside (entry edges)
            # We do this by checking the predecessor blocks of the header:
            # - Predecessors IN the loop are back-edge sources (latches)
            # - Predecessors NOT in the loop are entry sources
            entry_preds = filter(p -> p ∉ loop.blocks, header_block.preds)
            latch_preds = filter(p -> p ∈ loop.blocks, header_block.preds)

            entry_val = nothing
            carried_val = nothing

            # The phi edges are Julia's internal block numbers
            # We need to map them to understand which values are entry vs carried
            # Simple heuristic: entry value is typically a literal or comes from outside the loop
            # For a while loop with init=0, the first edge is usually the entry
            for (edge_idx, _edge) in enumerate(edges)
                if isassigned(values, edge_idx)
                    val = values[edge_idx]

                    # Heuristic: if value is an SSAValue from inside the loop, it's the carried value
                    # If it's a literal or from outside the loop, it's the entry value
                    if val isa SSAValue
                        val_stmt = val.id
                        stmt_to_blk = stmt_to_block_map(blocks, length(stmts))
                        if val_stmt > 0 && val_stmt <= length(stmts)
                            val_block = stmt_to_blk[val_stmt]
                            if val_block ∈ loop.blocks
                                # Value defined inside loop - this is carried value
                                carried_val = val
                            else
                                # Value defined outside loop - this is entry value
                                entry_val = convert_phi_value(val)
                            end
                        else
                            entry_val = convert_phi_value(val)
                        end
                    else
                        # Literal value - assume this is the entry/initial value
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            if entry_val !== nothing
                push!(init_values, entry_val)
            end
            if carried_val !== nothing
                push!(carried_values, carried_val)
            end

            # Create block argument for this phi
            phi_type = code.ssavaluetypes[si]
            arg = BlockArg(length(block_args) + 1, phi_type)
            push!(block_args, arg)
        end
    end

    # Build loop body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.stmts, si)
        end
    end

    # Find the condition for loop exit (from GotoIfNot in header)
    condition = nothing
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            break
        end
    end

    # Collect body block statements (from latch blocks)
    for latch_bi in loop.latches
        if latch_bi != loop.header
            latch_block = blocks[latch_bi]
            for si in latch_block.range
                stmt = stmts[si]
                if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                    push!(body.stmts, si)
                end
            end
        end
    end

    # Create the conditional structure inside the loop body
    # The loop condition determines continue vs break
    if condition !== nothing
        cond_value = convert_phi_value(condition)

        # Then branch: continue the loop (condition was true)
        then_block = Block(block_id[])
        block_id[] += 1
        then_block.terminator = ContinueOp(carried_values)

        # Else branch: exit the loop (condition was false)
        else_block = Block(block_id[])
        block_id[] += 1
        # The result is the current loop-carried value (before update)
        # For the bar(x,y) example, we want to yield %2 (the acc before checking)
        result_values = IRValue[]
        for (i, arg) in enumerate(block_args)
            push!(result_values, arg)
        end
        else_block.terminator = BreakOp(result_values)

        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(body.nested, if_op)
    else
        # No condition found - unconditional loop (shouldn't happen for while loops)
        body.terminator = ContinueOp(carried_values)
    end

    return LoopOp(init_values, body, result_vars)
end

"""
    find_last_stmt_in_julia_block(code::CodeInfo, julia_block::Integer) -> Int

Find the last statement index that belongs to the given Julia basic block number.
Julia block numbers are shown in the IR (e.g., #1, #2, #3).
"""
function find_last_stmt_in_julia_block(code::CodeInfo, julia_block::Integer)
    # Julia IR uses block numbers that correspond to statement ranges
    # We need to find where block N ends

    stmts = code.code
    n = length(stmts)

    # Find block starts by scanning for control flow targets
    block_starts = [1]
    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            if stmt.label <= n && stmt.label ∉ block_starts
                push!(block_starts, stmt.label)
            end
        elseif stmt isa GotoIfNot
            if stmt.dest <= n && stmt.dest ∉ block_starts
                push!(block_starts, stmt.dest)
            end
            if i + 1 <= n && (i + 1) ∉ block_starts
                push!(block_starts, i + 1)
            end
        elseif stmt isa PhiNode
            # Phi nodes reference block numbers in their edges
        end
        # After GotoNode or ReturnNode, next statement starts a new block
        if (stmt isa GotoNode || stmt isa ReturnNode) && i + 1 <= n
            if (i + 1) ∉ block_starts
                push!(block_starts, i + 1)
            end
        end
    end

    sort!(unique!(block_starts))

    # Map Julia block number to statement range
    if julia_block < 1 || julia_block > length(block_starts)
        return 0
    end

    start_stmt = block_starts[julia_block]
    if julia_block < length(block_starts)
        end_stmt = block_starts[julia_block + 1] - 1
    else
        end_stmt = n
    end

    return end_stmt
end

"""
    convert_phi_value(val) -> IRValue

Convert a phi node value to an IRValue.
"""
function convert_phi_value(val)
    if val isa SSAValue
        return val
    elseif val isa Argument
        return val
    elseif val isa Int
        # Literal integer - we'll need to handle this specially
        # For now, create a pseudo-SSAValue (this is a hack)
        return SSAValue(0)  # Indicates literal 0
    elseif val isa QuoteNode
        # Quoted literal
        return SSAValue(0)
    else
        # Other types - try to convert
        return SSAValue(0)
    end
end

"""
    collect_block_stmts!(block::Block, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a Block.
"""
function collect_block_stmts!(block::Block, info::BlockInfo, code::CodeInfo)
    stmts = code.code
    for si in info.range
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push!(block.stmts, si)
        end
    end
end

"""
    lower_flat(code::CodeInfo) -> StructuredCodeInfo

Create a flat structured IR representation without restructuring.
Used as fallback when restructuring fails.
"""
function lower_flat(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    entry = Block(1)

    for i in 1:n
        stmt = stmts[i]
        if stmt isa ReturnNode
            entry.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot)
            push!(entry.stmts, i)
        end
    end

    # If no explicit return was set, check for the last statement
    if entry.terminator === nothing && n > 0
        last_stmt = stmts[n]
        if last_stmt isa ReturnNode
            entry.terminator = last_stmt
        end
    end

    return StructuredCodeInfo(code, entry)
end

#=============================================================================
 Validation
=============================================================================#

"""
    validate_structured_ir(sci::StructuredCodeInfo) -> Bool

Validate that the structured IR is well-formed.
Returns true if valid, throws an error otherwise.
"""
function validate_structured_ir(sci::StructuredCodeInfo)
    code = sci.code
    n = length(code.code)

    # Collect all referenced statement indices
    referenced = Set{Int}()
    each_stmt(sci.entry) do idx
        if idx < 1 || idx > n
            error("Invalid statement index $idx (code has $n statements)")
        end
        if idx in referenced
            error("Statement $idx referenced multiple times")
        end
        push!(referenced, idx)
    end

    # Note: Not all statements need to be referenced (e.g., control flow statements
    # become implicit in the structure)

    return true
end

#=============================================================================
 Debugging Utilities
=============================================================================#

"""
    dump_structured_ir(sci::StructuredCodeInfo)

Print the structured IR for debugging.
"""
function dump_structured_ir(sci::StructuredCodeInfo)
    print_structured_ir(stdout, sci)
end

"""
    dump_julia_ir(code::CodeInfo)

Print the Julia IR for debugging.
"""
function dump_julia_ir(code::CodeInfo)
    println("Julia IR:")
    for (i, stmt) in enumerate(code.code)
        ssatype = code.ssavaluetypes[i]
        println("  %$i = $stmt :: $ssatype")
    end
end

"""
    compare_ir(code::CodeInfo, sci::StructuredCodeInfo)

Compare Julia IR and structured IR side-by-side for debugging.
"""
function compare_ir(code::CodeInfo, sci::StructuredCodeInfo)
    println("=== Julia IR ===")
    dump_julia_ir(code)
    println()
    println("=== Structured IR ===")
    dump_structured_ir(sci)
end

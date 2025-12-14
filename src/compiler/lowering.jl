# Lowering: Julia IR -> Structured IR
#
# Provides convenience functions for converting Julia CodeInfo to StructuredCodeInfo.
# The actual restructuring is done by structurize!() in restructuring.jl.

#=============================================================================
 Main Entry Point
=============================================================================#

"""
    lower_to_structured_ir(code::CodeInfo) -> StructuredCodeInfo

Convert Julia CodeInfo to StructuredCodeInfo with structured control flow.

This is a convenience function that creates a StructuredCodeInfo and
calls structurize!() to convert control flow to structured ops.
"""
function lower_to_structured_ir(code::CodeInfo)
    sci = StructuredCodeInfo(code)
    structurize!(sci)
    return sci
end

"""
    lower_to_structured_ir(target::TileTarget) -> StructuredCodeInfo

Convert a TileTarget to StructuredCodeInfo.
"""
function lower_to_structured_ir(target::TileTarget)
    lower_to_structured_ir(target.ci)
end

#=============================================================================
 Validation
=============================================================================#

"""
    validate_structured_ir(sci::StructuredCodeInfo) -> Bool

Validate that the structured IR is well-formed (no duplicate statement references,
all indices in bounds). This is a basic sanity check, not a control flow check.

For verifying that control flow is properly structured, use
`validate_structured_control_flow(sci)` instead.
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

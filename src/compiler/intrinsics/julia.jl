# high-level intrinsics
#
# These handle Julia built-ins and map them to Tile IR intrinsics.


## Julia intrinsics

# We handle Julia Core.Intrinsics here for two reasons:
#
# 1. IRStructurizer support: add_int, slt_int, sle_int, ult_int are used by
#    IRStructurizer for control flow transformations.
#
# 2. JuliaLang/julia#60582 workaround: When certain functions are inlined, Julia
#    uses non-overlay methods, causing Core intrinsics to appear instead of our own.

function emit_intrinsic!(ctx::CGCtx, func::Core.IntrinsicFunction, args)
    # Integer arithmetic
    if func === Core.Intrinsics.add_int
        emit_intrinsic!(ctx, Intrinsics.addi, args)
    elseif func === Core.Intrinsics.sub_int
        emit_intrinsic!(ctx, Intrinsics.subi, args)
    elseif func === Core.Intrinsics.mul_int
        emit_intrinsic!(ctx, Intrinsics.muli, args)
    elseif func === Core.Intrinsics.sdiv_int
        emit_intrinsic!(ctx, Intrinsics.divi, [args..., SignednessSigned])
    elseif func === Core.Intrinsics.udiv_int
        emit_intrinsic!(ctx, Intrinsics.divi, [args..., SignednessUnsigned])
    elseif func === Core.Intrinsics.srem_int
        emit_intrinsic!(ctx, Intrinsics.remi, [args..., SignednessSigned])
    elseif func === Core.Intrinsics.urem_int
        emit_intrinsic!(ctx, Intrinsics.remi, [args..., SignednessUnsigned])
    elseif func === Core.Intrinsics.neg_int
        emit_intrinsic!(ctx, Intrinsics.negi, args)

    # Integer comparisons
    elseif func === Core.Intrinsics.eq_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpEqual, SignednessSigned])
    elseif func === Core.Intrinsics.ne_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpNotEqual, SignednessSigned])
    elseif func === Core.Intrinsics.slt_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessSigned])
    elseif func === Core.Intrinsics.sle_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThanOrEqual, SignednessSigned])
    elseif func === Core.Intrinsics.ult_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThan, SignednessUnsigned])
    elseif func === Core.Intrinsics.ule_int
        emit_intrinsic!(ctx, Intrinsics.cmpi, [args..., CmpLessThanOrEqual, SignednessUnsigned])

    # Bitwise operations
    elseif func === Core.Intrinsics.and_int
        emit_intrinsic!(ctx, Intrinsics.andi, args)
    elseif func === Core.Intrinsics.or_int
        emit_intrinsic!(ctx, Intrinsics.ori, args)
    elseif func === Core.Intrinsics.xor_int
        emit_intrinsic!(ctx, Intrinsics.xori, args)
    elseif func === Core.Intrinsics.shl_int
        emit_intrinsic!(ctx, Intrinsics.shli, args)
    elseif func === Core.Intrinsics.ashr_int
        emit_intrinsic!(ctx, Intrinsics.shri, [args..., SignednessSigned])
    elseif func === Core.Intrinsics.lshr_int
        emit_intrinsic!(ctx, Intrinsics.shri, [args..., SignednessUnsigned])
    elseif func === Core.Intrinsics.not_int
        # not_int(x) = xor_int(x, -1)
        # We need to emit a constant -1 of the same type and xor with it
        emit_not_int!(ctx, args)

    # Integer conversions
    elseif func === Core.Intrinsics.sext_int
        emit_int_extension!(ctx, args, SignednessSigned)
    elseif func === Core.Intrinsics.zext_int
        emit_int_extension!(ctx, args, SignednessUnsigned)
    elseif func === Core.Intrinsics.trunc_int
        emit_int_truncation!(ctx, args)
    elseif func === Core.Intrinsics.bitcast
        emit_bitcast!(ctx, args)

    # Float-int conversions
    elseif func === Core.Intrinsics.sitofp
        emit_int_to_float!(ctx, args, SignednessSigned)
    elseif func === Core.Intrinsics.uitofp
        emit_int_to_float!(ctx, args, SignednessUnsigned)
    elseif func === Core.Intrinsics.fptosi
        emit_float_to_int!(ctx, args, SignednessSigned)
    elseif func === Core.Intrinsics.fptoui
        emit_float_to_int!(ctx, args, SignednessUnsigned)

    # Float-float conversions
    elseif func === Core.Intrinsics.fpext
        emit_float_conversion!(ctx, args)
    elseif func === Core.Intrinsics.fptrunc
        emit_float_conversion!(ctx, args)

    # Floating-point arithmetic
    elseif func === Core.Intrinsics.add_float
        emit_intrinsic!(ctx, Intrinsics.addf, args)
    elseif func === Core.Intrinsics.sub_float
        emit_intrinsic!(ctx, Intrinsics.subf, args)
    elseif func === Core.Intrinsics.mul_float
        emit_intrinsic!(ctx, Intrinsics.mulf, args)
    elseif func === Core.Intrinsics.div_float
        emit_intrinsic!(ctx, Intrinsics.divf, args)
    elseif func === Core.Intrinsics.neg_float
        emit_intrinsic!(ctx, Intrinsics.negf, args)
    elseif func === Core.Intrinsics.abs_float
        emit_intrinsic!(ctx, Intrinsics.absf, args)
    elseif func === Core.Intrinsics.fma_float
        emit_intrinsic!(ctx, Intrinsics.fma, args)
    elseif func === Core.Intrinsics.muladd_float
        emit_intrinsic!(ctx, Intrinsics.fma, args)
    elseif func === Core.Intrinsics.sqrt_llvm
        emit_intrinsic!(ctx, Intrinsics.sqrt, args)
    elseif func === Core.Intrinsics.ceil_llvm
        emit_intrinsic!(ctx, Intrinsics.ceil, args)
    elseif func === Core.Intrinsics.floor_llvm
        emit_intrinsic!(ctx, Intrinsics.floor, args)

    # Floating-point comparisons
    elseif func === Core.Intrinsics.eq_float
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpEqual])
    elseif func === Core.Intrinsics.ne_float
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpNotEqual])
    elseif func === Core.Intrinsics.lt_float
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpLessThan])
    elseif func === Core.Intrinsics.le_float
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpLessThanOrEqual])

    # Fast-math variants (treat same as regular for now)
    elseif func === Core.Intrinsics.add_float_fast
        emit_intrinsic!(ctx, Intrinsics.addf, args)
    elseif func === Core.Intrinsics.sub_float_fast
        emit_intrinsic!(ctx, Intrinsics.subf, args)
    elseif func === Core.Intrinsics.mul_float_fast
        emit_intrinsic!(ctx, Intrinsics.mulf, args)
    elseif func === Core.Intrinsics.div_float_fast
        emit_intrinsic!(ctx, Intrinsics.divf, args)
    elseif func === Core.Intrinsics.neg_float_fast
        emit_intrinsic!(ctx, Intrinsics.negf, args)
    elseif func === Core.Intrinsics.sqrt_llvm_fast
        emit_intrinsic!(ctx, Intrinsics.sqrt, args)
    elseif func === Core.Intrinsics.eq_float_fast
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpEqual])
    elseif func === Core.Intrinsics.ne_float_fast
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpNotEqual])
    elseif func === Core.Intrinsics.lt_float_fast
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpLessThan])
    elseif func === Core.Intrinsics.le_float_fast
        emit_intrinsic!(ctx, Intrinsics.cmpf, [args..., CmpLessThanOrEqual])

    # TODO: No Tile IR equivalent yet
    # - bswap_int, ctlz_int, ctpop_int, cttz_int (bit manipulation)
    # - copysign_float, flipsign_int (sign manipulation)
    # - rint_llvm, trunc_llvm (float rounding)
    # - checked_* (overflow-checked arithmetic)
    # - pointer operations (add_ptr, sub_ptr, pointerref, pointerset)
    # - atomic operations (handled separately via Intrinsics)
    # - have_fma (compile-time query)

    else
        error("Unhandled Julia intrinsic: $func")
    end
end

"""
Emit integer extension (sext_int or zext_int).
args[1] is the target type, args[2] is the value to extend.
"""
function emit_int_extension!(ctx::CGCtx, args, signedness::Signedness)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for integer extension")
    emit_intrinsic!(ctx, Intrinsics.exti, [args[2], target_type, signedness])
end

"""
Emit integer truncation (trunc_int).
args[1] is the target type, args[2] is the value to truncate.
"""
function emit_int_truncation!(ctx::CGCtx, args)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for integer truncation")
    emit_intrinsic!(ctx, Intrinsics.trunci, [args[2], target_type])
end

"""
Emit bitcast.
args[1] is the target type, args[2] is the value to bitcast.
"""
function emit_bitcast!(ctx::CGCtx, args)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for bitcast")
    emit_intrinsic!(ctx, Intrinsics.bitcast, [args[2], target_type])
end

"""
Emit integer to float conversion (sitofp or uitofp).
args[1] is the target float type, args[2] is the integer value.
"""
function emit_int_to_float!(ctx::CGCtx, args, signedness::Signedness)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for int-to-float conversion")
    emit_intrinsic!(ctx, Intrinsics.itof, [args[2], target_type, signedness])
end

"""
Emit float to integer conversion (fptosi or fptoui).
args[1] is the target integer type, args[2] is the float value.
"""
function emit_float_to_int!(ctx::CGCtx, args, signedness::Signedness)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for float-to-int conversion")
    emit_intrinsic!(ctx, Intrinsics.ftoi, [args[2], target_type, signedness])
end

"""
Emit float-to-float conversion (fpext or fptrunc).
args[1] is the target float type, args[2] is the source float value.
"""
function emit_float_conversion!(ctx::CGCtx, args)
    target_type = get_constant(ctx, args[1])
    target_type === nothing && error("Cannot resolve target type for float conversion")
    emit_intrinsic!(ctx, Intrinsics.ftof, [args[2], target_type])
end

"""
Emit bitwise NOT (not_int) as xor with -1.
"""
function emit_not_int!(ctx::CGCtx, args)
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source value for not_int")

    # Create constant -1 of same type
    jltype = unwrap_type(source.jltype)
    minus_one = jltype(-1)

    # Emit xor(x, -1)
    emit_intrinsic!(ctx, Intrinsics.xori, [args[1], minus_one])
end


## Built-in functions

# We cannot overlay built-in functions

function emit_intrinsic!(ctx::CGCtx, ::typeof(===), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for ===")
    rhs === nothing && error("Cannot resolve RHS operand for ===")

    result_type_id = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type_id, lhs_v, rhs_v;
                              predicate=CmpEqual, signedness=SignednessSigned)

    CGVal(result_v, result_type_id, Bool, Int[])
end

emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args) = nothing

emit_intrinsic!(ctx::CGCtx, ::typeof(Base.donotdelete), args) = nothing


## Other

## XXX: Tile constructor
function emit_intrinsic!(ctx::CGCtx, func::Type{<:Tile}, args)
    # Emit the scalar value
    source = emit_value!(ctx, args[1])

    # Get element type from the constructor type (Tile{T, S})
    # If func is fully parametric, extract T; otherwise infer from source
    elem_type = if func !== Tile && length(func.parameters) >= 1
        func.parameters[1]
    elseif source.constant !== nothing
        typeof(source.constant)
    else
        unwrap_type(source.jltype)
    end

    # Return as 0D tile type with element type from the constructor
    result_jltype = Tile{elem_type, ()}
    CGVal(source.v, source.type_id, result_jltype, source.shape)
end

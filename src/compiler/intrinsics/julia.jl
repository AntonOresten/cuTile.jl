# Julia built-in intrinsics
#
# Handles Julia built-ins that survive into the StructuredIRCode and are NOT
# lowered by rewrite_passes! (because they have no direct cuTile equivalent
# or are compile-time-only constructs).
#
# Julia Core.Intrinsics (add_int, sub_int, slt_int, etc.) and Core.ifelse /
# === are lowered to cuTile Intrinsics by rewrite_passes! and should not
# appear here.

# built-in: tuple (ghost — no runtime representation)
emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

# built-in: isa (compile-time type check, emitted as a tile constant)
function emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args)
    length(args) >= 2 || return nothing
    T = @something get_constant(ctx, args[2]) return nothing
    val_type = CC.widenconst(argextype(ctx, args[1]))
    emit_constant!(ctx, val_type <: T, Tile{Bool, Tuple{}})
end

# built-in: donotdelete (keep-alive barrier — no Tile IR emission)
emit_intrinsic!(ctx::CGCtx, ::typeof(donotdelete), args) = nothing

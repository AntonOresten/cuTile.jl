# Floating-point mode scope intrinsics (@fpmode)

@intrinsic fpmode_begin(rounding_mode, flush_to_zero)
@intrinsic fpmode_end()

tfunc(𝕃, ::typeof(Intrinsics.fpmode_begin), @nospecialize args...) = Nothing
efunc(::typeof(Intrinsics.fpmode_begin), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fpmode_begin), args)
    rm_arg = @something get_constant(ctx, args[1]) nothing
    ftz_arg = @something get_constant(ctx, args[2]) nothing

    parent = current_fpmode(ctx)
    rm = rm_arg isa Rounding.T ? convert_enum(RoundingMode, Integer(rm_arg)) : parent.rounding_mode
    ftz = ftz_arg isa Bool ? ftz_arg : parent.flush_to_zero

    push!(ctx.fpmode_stack, FPMode(rm, ftz))
    ghost_value(Nothing)
end

tfunc(𝕃, ::typeof(Intrinsics.fpmode_end)) = Nothing
efunc(::typeof(Intrinsics.fpmode_end), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fpmode_end), args)
    isempty(ctx.fpmode_stack) || pop!(ctx.fpmode_stack)
    ghost_value(Nothing)
end

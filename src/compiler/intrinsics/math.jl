# Floating-point math


## TODO: cuda_tile.atan2


## cuda_tile.ceil

@eval Intrinsics begin
    """Element-wise ceiling. Compiled to cuda_tile.ceil."""
    @noinline function ceil(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.ceil), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for ceil()")

    result = encode_CeilOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.cosh

@eval Intrinsics begin
    """Element-wise hyperbolic cosine. Compiled to cuda_tile.cosh."""
    @noinline function cosh(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cosh), args)
    cb = ctx.cb
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for cosh()")

    result = encode_CosHOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.cos

@eval Intrinsics begin
    """Element-wise cosine. Compiled to cuda_tile.cos."""
    @noinline function cos(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cos), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for cos()")

    result = encode_CosOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.exp2

@eval Intrinsics begin
    """Element-wise base-2 exponential. Compiled to cuda_tile.exp2."""
    @noinline function exp2(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for exp2()")

    result = encode_Exp2Op!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.exp

@eval Intrinsics begin
    """Element-wise exponential. Compiled to cuda_tile.exp."""
    @noinline function exp(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.exp), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for exp()")

    result = encode_ExpOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.floor

@eval Intrinsics begin
    """Element-wise floor. Compiled to cuda_tile.floor."""
    @noinline function floor(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.floor), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for floor()")

    result = encode_FloorOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.fma

@eval Intrinsics begin
    """Element-wise fused multiply-add. Compiled to cuda_tile.fma."""
    @noinline function fma(a::Tile{T, S}, b::Tile{T, S}, c::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(a, b, c)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.fma), args)
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for fma")

    result = encode_FmaOp!(cb, lhs.type_id, lhs.v, rhs.v, acc.v)
    CGVal(result, lhs.type_id, lhs.jltype, lhs.shape)
end

## cuda_tile.log2

@eval Intrinsics begin
    """Element-wise base-2 logarithm. Compiled to cuda_tile.log2."""
    @noinline function log2(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log2), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for log2()")

    result = encode_Log2Op!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end

## cuda_tile.log

@eval Intrinsics begin
    """Element-wise natural logarithm. Compiled to cuda_tile.log."""
    @noinline function log(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.log), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for log()")

    result = encode_LogOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.maxf

@eval Intrinsics begin
    @noinline maxf(x::T, y::T) where {T<:AbstractFloat} = ifelse(x > y || isnan(x), x, y)
    @noinline maxf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (Base.donotdelete(a, b); Tile{T, S}())
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    emit_binop!(ctx, args, encode_MaxFOp!)
end


## cuda_tile.minf

@eval Intrinsics begin
    @noinline minf(x::T, y::T) where {T<:AbstractFloat} = ifelse(x < y || isnan(x), x, y)
    @noinline minf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (Base.donotdelete(a, b); Tile{T, S}())
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.minf), args)
    emit_binop!(ctx, args, encode_MinFOp!)
end


## cuda_tile.pow

@eval Intrinsics begin
    """Element-wise power. Compiled to cuda_tile.pow."""
    @noinline function pow(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(a, b)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.pow), args)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for pow")

    result_v = encode_PowOp!(cb, lhs.type_id, lhs.v, rhs.v)

    CGVal(result_v, lhs.type_id, lhs.jltype, lhs.shape)
end


## cuda_tile.remf

@eval Intrinsics begin
    """Element-wise remainder. Compiled to cuda_tile.remf."""
    @noinline function remf(a::Tile{T, S}, b::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(a, b)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.remf), args)
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for remf")

    result = encode_RemFOp!(cb, lhs.type_id, lhs.v, rhs.v)

    CGVal(result, lhs.type_id, lhs.jltype, lhs.shape)
end

## cuda_tile.rsqrt

@eval Intrinsics begin
    """Element-wise reciprocal square root. Compiled to cuda_tile.rsqrt."""
    @noinline function rsqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.rsqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for rsqrt()")

    result = encode_RsqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.sinh

@eval Intrinsics begin
    """Element-wise hyperbolic sine. Compiled to cuda_tile.sinh."""
    @noinline function sinh(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sinh), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sinh()")

    result = encode_SinHOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.sin

@eval Intrinsics begin
    """Element-wise sine. Compiled to cuda_tile.sin."""
    @noinline function sin(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sin), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sin()")

    result = encode_SinOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.sqrt

@eval Intrinsics begin
    """Element-wise square root. Compiled to cuda_tile.sqrt."""
    @noinline function sqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.sqrt), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sqrt()")

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.tanh

@eval Intrinsics begin
    """Element-wise hyperbolic tangent. Compiled to cuda_tile.tanh."""
    @noinline function tanh(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tanh), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for tanh()")

    result = encode_TanHOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end


## cuda_tile.tan

@eval Intrinsics begin
    """Element-wise tangent. Compiled to cuda_tile.tan."""
    @noinline function tan(tile::Tile{T, S}) where {T <: AbstractFloat, S}
        Base.donotdelete(tile)
        Tile{T, S}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.tan), args)
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for tan()")

    result = encode_TanOp!(cb, source.type_id, source.v)
    CGVal(result, source.type_id, source.jltype, source.shape)
end

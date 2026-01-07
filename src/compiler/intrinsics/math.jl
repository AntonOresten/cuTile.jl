# Floating-point math


## TODO: cuda_tile.atan2


## TODO: cuda_tile.ceil


## TODO: cuda_tile.cosh


## TODO: cuda_tile.cos


## TODO: cuda_tile.exp2


## TODO: cuda_tile.exp


## TODO: cuda_tile.floor


## TODO: cuda_tile.fma


## TODO: cuda_tile.log2


## TODO: cuda_tile.log


## cuda_tile.maxf

@eval Intrinsics begin
    # NOTE: Must perform actual computation because overlay methods can execute
    # during constant propagation (Julia bug #47349).
    @noinline maxf(x::T, y::T) where {T<:AbstractFloat} = ifelse(x > y || isnan(x), x, y)
    @noinline maxf(a::Tile{T, S}, b::Tile{T, S}) where {T<:AbstractFloat, S} = (Base.donotdelete(a, b); Tile{T, S}())
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.maxf), args)
    emit_binop!(ctx, args, encode_MaxFOp!)
end


## cuda_tile.minf

@eval Intrinsics begin
    # NOTE: Must perform actual computation because overlay methods can execute
    # during constant propagation (Julia bug #47349).
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


## TODO: cuda_tile.remf


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

    result = encode_RSqrtOp!(cb, source.type_id, source.v)

    CGVal(result, source.type_id, source.jltype, source.shape)
end


## TODO: cuda_tile.sinh


## TODO: cuda_tile.sin


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


## TODO: cuda_tile.tanh


## TODO: cuda_tile.tan

# Method overlays for cuTile compilation
#
# Defines alternative implementations of Base functions that map
# to Tile IR operations instead of Julia intrinsics.

macro overlay(ex)
    esc(:(Base.Experimental.@consistent_overlay cuTileMethodTable Base.@assume_effects :foldable $ex))
end


#=============================================================================
 Type conversions
=============================================================================#

# Type tuples for metaprogramming specific overlays
# Generic overlays don't take precedence over Core's Int64(x::BuiltinInts) etc.
const SignedInts = (Int8, Int16, Int32, Int64)
const UnsignedInts = (UInt8, UInt16, UInt32, UInt64)
const Floats = (Float16, Float32, Float64)

# Integer to integer (specific type pairs for promotion/truncation)
for T in SignedInts, S in SignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, SignednessSigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

for T in UnsignedInts, S in UnsignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, SignednessUnsigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

# Integer extension/truncation (via rem) - T and S both used in body
@overlay Base.rem(x::T, ::Type{S}) where {T <: Signed, S <: Signed} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, SignednessSigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

@overlay Base.rem(x::T, ::Type{S}) where {T <: Unsigned, S <: Unsigned} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, SignednessUnsigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

# Float to float (specific type pairs)
for T in Floats, S in Floats
    T === S && continue
    @eval @overlay $T(x::$S) = Intrinsics.ftof(x, $T)
end

# Integer to float (specific type pairs)
for F in Floats
    for I in SignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, SignednessSigned)
    end
    for I in UnsignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, SignednessUnsigned)
    end
    @eval @overlay $F(x::Bool) = Intrinsics.itof(x, $F, SignednessUnsigned)
end

# Float to integer (via unsafe_trunc)
for F in Floats
    for I in SignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, SignednessSigned)
    end
    for I in UnsignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, SignednessUnsigned)
    end
end


#=============================================================================
 Integer operations
=============================================================================#

# NOTE: some integer arithmetic operations are NOT overlaid because
#       the IRStructurizer needs to see them to convert `while` loops into `for` loops.

#@overlay Base.:+(x::T, y::T) where {T <: ScalarInt} = Intrinsics.addi(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarInt} = Intrinsics.subi(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarInt} = Intrinsics.muli(x, y)
@overlay Base.:-(x::ScalarInt) = Intrinsics.negi(x)
@overlay Base.div(x::T, y::T) where {T <: Signed} = Intrinsics.divi(x, y, SignednessSigned)
@overlay Base.div(x::T, y::T) where {T <: Unsigned} = Intrinsics.divi(x, y, SignednessUnsigned)

@overlay Base.:(==)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, CmpEqual, SignednessSigned)
@overlay Base.:(!=)(x::T, y::T) where {T <: ScalarInt} = Intrinsics.cmpi(x, y, CmpNotEqual, SignednessSigned)
#@overlay Base.:<(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, CmpLessThan, SignednessSigned)
#@overlay Base.:<(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, CmpLessThan, SignednessUnsigned)
#@overlay Base.:<=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(x, y, CmpLessThanOrEqual, SignednessSigned)
#@overlay Base.:<=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(x, y, CmpLessThanOrEqual, SignednessUnsigned)
@overlay Base.:>(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, CmpLessThan, SignednessSigned)
@overlay Base.:>(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, CmpLessThan, SignednessUnsigned)
@overlay Base.:>=(x::T, y::T) where {T <: Signed} = Intrinsics.cmpi(y, x, CmpLessThanOrEqual, SignednessSigned)
@overlay Base.:>=(x::T, y::T) where {T <: Unsigned} = Intrinsics.cmpi(y, x, CmpLessThanOrEqual, SignednessUnsigned)

@overlay Base.rem(x::T, y::T) where {T <: Signed} = Intrinsics.remi(x, y, SignednessSigned)
@overlay Base.rem(x::T, y::T) where {T <: Unsigned} = Intrinsics.remi(x, y, SignednessUnsigned)

@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, SignednessSigned)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, SignednessUnsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, SignednessSigned)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, SignednessUnsigned)


#=============================================================================
 Floating-point operations
=============================================================================#

@overlay Base.:+(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.addf(x, y)
@overlay Base.:-(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.subf(x, y)
@overlay Base.:*(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.mulf(x, y)
@overlay Base.:/(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.divf(x, y)
@overlay Base.:-(x::ScalarFloat) = Intrinsics.negf(x)

@overlay Base.:<(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpLessThan)
@overlay Base.:<=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpLessThanOrEqual)
@overlay Base.:>(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpGreaterThan)
@overlay Base.:>=(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpGreaterThanOrEqual)
@overlay Base.:(==)(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, y, CmpEqual)

@overlay Base.abs(x::ScalarFloat) = Intrinsics.absf(x)

@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y)


#=============================================================================
 Bitwise operations
=============================================================================#

@overlay Base.:&(x::T, y::T) where {T <: ScalarInt} = Intrinsics.andi(x, y)
@overlay Base.:|(x::T, y::T) where {T <: ScalarInt} = Intrinsics.ori(x, y)
@overlay Base.xor(x::T, y::T) where {T <: ScalarInt} = Intrinsics.xori(x, y)

# Bitwise NOT via XOR with all 1s - T used in body
@overlay Base.:~(x::T) where {T <: Signed} = Intrinsics.xori(x, T(-1))
@overlay Base.:~(x::T) where {T <: Unsigned} = Intrinsics.xori(x, ~T(0))

# Boolean negation
@overlay Base.:!(x::Bool) = Intrinsics.xori(x, true)

# Shifts
@overlay Base.:<<(x::ScalarInt, y::Integer) = Intrinsics.shli(x, y)
@overlay Base.:>>(x::Signed, y::Integer) = Intrinsics.shri(x, y, SignednessSigned)
@overlay Base.:>>(x::Unsigned, y::Integer) = Intrinsics.shri(x, y, SignednessUnsigned)
@overlay Base.:>>>(x::ScalarInt, y::Integer) = Intrinsics.shri(x, y, SignednessUnsigned)

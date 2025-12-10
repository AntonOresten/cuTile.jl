# Operation encoders for Tile IR bytecode
# Each function encodes a specific operation into the code builder's buffer

include("writer.jl")

# Opcode constants (from Python's encodings.py)
module Opcode
    const AbsFOp = 0
    const AbsIOp = 1
    const AddFOp = 2
    const AddIOp = 3
    const AndIOp = 4
    const AssertOp = 5
    const AssumeOp = 6
    const AtomicCASPtrOp = 7
    const AtomicRMWPtrOp = 8
    const BarrierOp = 9
    const BitcastOp = 10
    const BroadcastOp = 11
    const CeilOp = 12
    const ClampFOp = 13
    const ConditionOp = 14
    const ContinueOp = 15
    const ConstantOp = 16
    const CosOp = 17
    # ... (skipping some for brevity)
    const DivFOp = 22
    const DivSIOp = 23
    const DivUIOp = 24
    # ...
    const ForOp = 39
    # ...
    const GetNumTileBlocksOp = 46  # Note: out of alphabetical order in binary format
    const GetTileBlockIdOp = 48
    # ...
    const IfOp = 51
    # ...
    const LoadViewTkoOp = 62
    # ...
    const LoopOp = 64
    # ...
    const MakePartitionViewOp = 66
    const MakeTensorViewOp = 67
    const MakeTokenOp = 68
    # ...
    const MulFOp = 76
    const MulIOp = 78
    # ...
    const ReturnOp = 92
    # ...
    const PermuteOp = 83
    # ...
    const StoreViewTkoOp = 102
    const SubFOp = 103
    const SubIOp = 104
    # ...
    const TruncIOp = 106
    # ...
    const YieldOp = 110
end

# Enums for operation attributes

@enum RoundingMode begin
    RoundingNearestEven = 0
    RoundingZero = 1
    RoundingNegativeInf = 2
    RoundingPositiveInf = 3
    RoundingApprox = 4
    RoundingFull = 5
    RoundingNearestIntToZero = 6
end

@enum IntegerOverflow begin
    OverflowNone = 0
    OverflowNSW = 1
    OverflowNUW = 2
    OverflowNW = 3
end

@enum MemoryOrderingSemantics begin
    MemoryWeak = 0
    MemoryRelaxed = 1
    MemoryAcquire = 2
    MemoryRelease = 3
    MemoryAcqRel = 4
end

@enum MemoryScope begin
    ScopeTLBlock = 0
    ScopeDevice = 1
    ScopeSystem = 2
end

@enum Signedness begin
    SignednessUnsigned = 0
    SignednessSigned = 1
end

# Helper to encode enum as single byte
function encode_enum!(buf::Vector{UInt8}, e::Enum)
    push!(buf, UInt8(Int(e)))
end

#=============================================================================
 Attribute Tags (for tagged attributes like AssumePredicate)
=============================================================================#

module AttributeTag
    const Integer = 0x01
    const Float = 0x02
    const Bool = 0x03
    const Type = 0x04
    const String = 0x05
    const Array = 0x06
    const DenseElements = 0x07
    const DivBy = 0x08
    const SameElements = 0x09
    const Dictionary = 0x0a
    const OptimizationHints = 0x0b
    const Bounded = 0x0c
end

#=============================================================================
 AssumePredicate types for AssumeOp
=============================================================================#

"""
    AssumePredicate

Abstract type for assume predicates used with AssumeOp.
"""
abstract type AssumePredicate end

"""
    DivBy(divisor; every=nothing, along=nothing)

Predicate asserting a value is divisible by `divisor`.
Optional `every` and `along` specify dimensional constraints.
"""
struct DivBy <: AssumePredicate
    divisor::Int
    every::Union{Int, Nothing}
    along::Union{Int, Nothing}
end
DivBy(divisor::Int) = DivBy(divisor, nothing, nothing)

"""
    Bounded(; lb=nothing, ub=nothing)

Predicate asserting a value is within bounds [lb, ub].
Either bound can be nothing (unbounded).
"""
struct Bounded <: AssumePredicate
    lb::Union{Int, Nothing}
    ub::Union{Int, Nothing}
end
Bounded(; lb=nothing, ub=nothing) = Bounded(lb, ub)

"""
    SameElements(values)

Predicate asserting all elements of a tile have the same value from `values`.
Used for broadcast/splat optimization hints.
"""
struct SameElements <: AssumePredicate
    values::Vector{Int}
end

"""
    encode_assume_predicate!(cb, pred::AssumePredicate)

Encode an assume predicate as a tagged attribute.
"""
function encode_assume_predicate!(cb::CodeBuilder, pred::DivBy)
    push!(cb.buf, AttributeTag.DivBy)
    encode_varint!(cb.buf, pred.divisor)
    flags = (pred.every !== nothing ? 0x01 : 0x00) |
            (pred.along !== nothing ? 0x02 : 0x00)
    push!(cb.buf, flags)
    if pred.every !== nothing
        encode_signed_varint!(cb.buf, pred.every)
    end
    if pred.along !== nothing
        encode_signed_varint!(cb.buf, pred.along)
    end
end

function encode_assume_predicate!(cb::CodeBuilder, pred::Bounded)
    push!(cb.buf, AttributeTag.Bounded)
    flags = (pred.lb !== nothing ? 0x01 : 0x00) |
            (pred.ub !== nothing ? 0x02 : 0x00)
    push!(cb.buf, flags)
    if pred.lb !== nothing
        encode_signed_varint!(cb.buf, pred.lb)
    end
    if pred.ub !== nothing
        encode_signed_varint!(cb.buf, pred.ub)
    end
end

function encode_assume_predicate!(cb::CodeBuilder, pred::SameElements)
    push!(cb.buf, AttributeTag.SameElements)
    encode_varint!(cb.buf, length(pred.values))
    for v in pred.values
        encode_signed_varint!(cb.buf, v)
    end
end

#=============================================================================
 Essential operations for vadd
=============================================================================#

"""
    encode_ConstantOp!(cb, result_type, value_bytes) -> Value

Create a constant value.
Opcode: 16
"""
function encode_ConstantOp!(cb::CodeBuilder, result_type::TypeId, value_bytes::Vector{UInt8})
    encode_varint!(cb.buf, Opcode.ConstantOp)
    encode_typeid!(cb.buf, result_type)
    encode_opattr_dense!(cb, value_bytes)
    return new_op!(cb)
end

"""
    encode_AssumeOp!(cb, result_type, value, predicate) -> Value

Create an assume operation that annotates a value with a predicate.
The predicate tells the compiler it can assume certain properties about
the value (e.g., divisibility, bounds) for optimization.

Opcode: 6
"""
function encode_AssumeOp!(cb::CodeBuilder, result_type::TypeId, value::Value, predicate::AssumePredicate)
    encode_varint!(cb.buf, Opcode.AssumeOp)
    encode_typeid!(cb.buf, result_type)
    encode_assume_predicate!(cb, predicate)
    encode_operand!(cb.buf, value)
    return new_op!(cb)
end

"""
    encode_GetTileBlockIdOp!(cb, x_type, y_type, z_type) -> (Value, Value, Value)

Get the 3D block ID.
Opcode: 48
"""
function encode_GetTileBlockIdOp!(cb::CodeBuilder, x_type::TypeId, y_type::TypeId, z_type::TypeId)
    encode_varint!(cb.buf, Opcode.GetTileBlockIdOp)
    encode_typeid!(cb.buf, x_type)
    encode_typeid!(cb.buf, y_type)
    encode_typeid!(cb.buf, z_type)
    return new_op!(cb, 3)
end

"""
    encode_GetNumTileBlocksOp!(cb, x_type, y_type, z_type) -> (Value, Value, Value)

Get the 3D grid dimensions.
Opcode: 49
"""
function encode_GetNumTileBlocksOp!(cb::CodeBuilder, x_type::TypeId, y_type::TypeId, z_type::TypeId)
    encode_varint!(cb.buf, Opcode.GetNumTileBlocksOp)
    encode_typeid!(cb.buf, x_type)
    encode_typeid!(cb.buf, y_type)
    encode_typeid!(cb.buf, z_type)
    return new_op!(cb, 3)
end

"""
    encode_MakeTensorViewOp!(cb, result_type, base, dynamic_shape, dynamic_strides) -> Value

Create a tensor view from a base pointer and dynamic shape/strides.
Opcode: 67
"""
function encode_MakeTensorViewOp!(cb::CodeBuilder, result_type::TypeId,
                                   base::Value,
                                   dynamic_shape::Vector{Value},
                                   dynamic_strides::Vector{Value})
    encode_varint!(cb.buf, Opcode.MakeTensorViewOp)
    # Variadic result types (just one)
    encode_typeid_seq!(cb.buf, [result_type])
    # Operands
    encode_operand!(cb.buf, base)
    encode_sized_operands!(cb.buf, dynamic_shape)
    encode_sized_operands!(cb.buf, dynamic_strides)
    return new_op!(cb)
end

"""
    encode_MakePartitionViewOp!(cb, result_type, tensor_view) -> Value

Create a partition view from a tensor view.
Opcode: 66
"""
function encode_MakePartitionViewOp!(cb::CodeBuilder, result_type::TypeId, tensor_view::Value)
    encode_varint!(cb.buf, Opcode.MakePartitionViewOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, tensor_view)
    return new_op!(cb)
end

"""
    encode_MakeTokenOp!(cb, result_type) -> Value

Create an initial memory ordering token.
Opcode: 68
"""
function encode_MakeTokenOp!(cb::CodeBuilder, result_type::TypeId)
    encode_varint!(cb.buf, Opcode.MakeTokenOp)
    encode_typeid!(cb.buf, result_type)
    return new_op!(cb)
end

"""
    encode_LoadViewTkoOp!(cb, tile_type, token_type, view, index; kwargs...) -> (Value, Value)

Load a tile from a partition view with token ordering.
Opcode: 62
"""
function encode_LoadViewTkoOp!(cb::CodeBuilder,
                               tile_type::TypeId,
                               token_type::TypeId,
                               view::Value,
                               index::Vector{Value};
                               token::Union{Value, Nothing}=nothing,
                               memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                               memory_scope::Union{MemoryScope, Nothing}=nothing,
                               optimization_hints::Union{Vector{UInt8}, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.LoadViewTkoOp)
    # Variadic result types
    encode_typeid_seq!(cb.buf, [tile_type, token_type])

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if token !== nothing
        flags |= 4
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        append!(cb.buf, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, view)
    encode_sized_operands!(cb.buf, index)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb, 2)
end

"""
    encode_StoreViewTkoOp!(cb, token_type, tile, view, index; kwargs...) -> Value

Store a tile to a partition view with token ordering.
Opcode: 102
"""
function encode_StoreViewTkoOp!(cb::CodeBuilder,
                                token_type::TypeId,
                                tile::Value,
                                view::Value,
                                index::Vector{Value};
                                token::Union{Value, Nothing}=nothing,
                                memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                                memory_scope::Union{MemoryScope, Nothing}=nothing,
                                optimization_hints::Union{Vector{UInt8}, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.StoreViewTkoOp)
    # Variadic result types (just token)
    encode_typeid_seq!(cb.buf, [token_type])

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if token !== nothing
        flags |= 4
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        append!(cb.buf, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, tile)
    encode_operand!(cb.buf, view)
    encode_sized_operands!(cb.buf, index)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb)
end

#=============================================================================
 Arithmetic operations
=============================================================================#

"""
    encode_AddFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float addition.
Opcode: 2
"""
function encode_AddFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.AddFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_AddIOp!(cb, result_type, lhs, rhs; overflow) -> Value

Integer addition.
Opcode: 3
"""
function encode_AddIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.AddIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_SubFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float subtraction.
Opcode: 103
"""
function encode_SubFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.SubFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MulFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float multiplication.
Opcode: 73
"""
function encode_MulFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.MulFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MulIOp!(cb, result_type, lhs, rhs; overflow) -> Value

Integer multiplication.
Opcode: 74
"""
function encode_MulIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.MulIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_TruncIOp!(cb, result_type, source; overflow) -> Value

Truncate integer to smaller width.
Opcode: 106
"""
function encode_TruncIOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                          overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.TruncIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Shape/layout operations
=============================================================================#

"""
    encode_PermuteOp!(cb, result_type, source, permutation) -> Value

Permute the dimensions of a tile according to the given permutation.
For 2D transpose, permutation would be [1, 0].
Opcode: 83
"""
function encode_PermuteOp!(cb::CodeBuilder, result_type::TypeId, source::Value,
                           permutation::Vector{Int})
    encode_varint!(cb.buf, Opcode.PermuteOp)
    encode_typeid!(cb.buf, result_type)
    # Encode permutation as dense int32 array attribute
    encode_dense_int32_array!(cb, permutation)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Control flow operations
=============================================================================#

"""
    encode_ReturnOp!(cb, operands)

Return from function.
Opcode: 92
"""
function encode_ReturnOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.ReturnOp)
    # Variadic result types (none)
    encode_typeid_seq!(cb.buf, TypeId[])
    # Operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

"""
    encode_YieldOp!(cb, operands)

Yield from a nested region (loop body, if branch, etc.).
Opcode: 110
"""
function encode_YieldOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.YieldOp)
    encode_sized_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

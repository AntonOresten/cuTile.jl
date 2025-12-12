# Codegen: Julia IR -> Tile IR bytecode
#
# Pattern matches on Julia SSA IR nodes and emits corresponding Tile IR operations.

include("target.jl")

using Core: SlotNumber


"""
    is_ghost_type(T) -> Bool

Check if a type is a ghost type (zero-size singleton that has no runtime representation).
Ghost types like Val{V} are filtered from kernel parameters since their values
are embedded in the specialized method.
"""
function is_ghost_type(@nospecialize(T))
    try
        isbitstype(T) && sizeof(T) == 0
    catch
        false
    end
end

"""
    emit_kernel!(writer, target; name, is_entry=true) -> Vector{UInt8}

Compile a TileTarget to Tile IR bytecode.
"""
function emit_kernel!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                      target::TileTarget;
                      name::String = string(target.mi.def.name),
                      is_entry::Bool = true)
    tr = Translation(writer)
    tt = tr.type_table

    # Build parameter list, handling ghost types and struct destructuring
    # Ghost types (like Val{V}) have no runtime representation - their values
    # are baked into the specialized IR as constants (QuoteNodes)
    # Struct types (like TileArray) are destructured into flat parameters
    param_types = TypeId[]
    # Track: (orig_arg_idx, field_or_nothing) for each flat param
    param_mapping = Tuple{Int, Union{Nothing, Symbol}}[]

    for (i, argtype) in enumerate(target.argtypes)
        argtype_unwrapped = unwrap_type(argtype)
        if is_ghost_type(argtype_unwrapped)
            # Ghost type - no runtime representation, skip as parameter
            continue
        elseif should_destructure(argtype_unwrapped)
            # Destructure struct into flat parameters, one entry per field
            # Get base type for field iteration (handles UnionAll like TileArray{T, N})
            base_type = get_base_type(argtype_unwrapped)
            for fi in 1:fieldcount(base_type)
                fname = fieldname(base_type, fi)
                ftype = fieldtype(base_type, fi)
                # For TileArray{T, N}, get N from type parameters
                ndims = 2  # Default for 2D arrays
                if argtype_unwrapped isa UnionAll || argtype_unwrapped isa DataType
                    # Try to extract N from type parameters
                    if hasfield(typeof(argtype_unwrapped), :parameters)
                        params = argtype_unwrapped.parameters
                        if length(params) >= 2 && params[2] isa Integer
                            ndims = params[2]
                        end
                    elseif argtype_unwrapped isa UnionAll
                        # For UnionAll, check the body
                        body = argtype_unwrapped.body
                        while body isa UnionAll
                            body = body.body
                        end
                        if length(body.parameters) >= 2 && body.parameters[2] isa Integer
                            ndims = body.parameters[2]
                        end
                    end
                end
                # Determine the actual count for tuple fields
                if fname === :sizes || fname === :strides
                    fcount = ndims
                    elem_type = Int32  # sizes and strides are Int32
                else
                    fcount = flat_field_count(ftype)
                    # Get the element type - for Ptr, extract the pointed type
                    if ftype isa UnionAll && ftype.body.name.name === :Ptr
                        # Get element type from the first TileArray parameter
                        if argtype_unwrapped isa UnionAll
                            body = argtype_unwrapped.body
                            while body isa UnionAll
                                body = body.body
                            end
                            if length(body.parameters) >= 1
                                ptr_elem_type = body.parameters[1]
                                if ptr_elem_type isa TypeVar
                                    # Get actual type from outer type
                                    ptr_elem_type = Float32  # Default
                                    params = argtype_unwrapped.parameters
                                    if length(params) >= 1 && (params[1] isa Type || params[1] isa DataType)
                                        ptr_elem_type = params[1]
                                    end
                                end
                                elem_type = Ptr{ptr_elem_type}
                            else
                                elem_type = ftype <: Tuple ? eltype(ftype) : ftype
                            end
                        else
                            elem_type = ftype <: Tuple ? eltype(ftype) : ftype
                        end
                    else
                        elem_type = ftype <: Tuple ? eltype(ftype) : ftype
                    end
                end
                for _ in 1:fcount
                    push!(param_types, tile_type_for_julia!(tr, elem_type))
                    push!(param_mapping, (i, fname))
                end
            end
            # Store the original type for this destructured arg
            tr.arg_types[i] = argtype_unwrapped
        else
            # Regular parameter
            push!(param_types, tile_type_for_julia!(tr, argtype_unwrapped))
            push!(param_mapping, (i, nothing))
        end
    end

    # Determine return types
    result_types = TypeId[]
    if target.rettype !== Nothing && target.rettype !== Union{}
        push!(result_types, tile_type_for_julia!(tr, target.rettype))
    end

    # Create the function
    cb = add_function!(writer, func_buf, name, param_types, result_types; is_entry)
    tr.code_builder = cb

    # Set up argument values
    arg_values = make_block_args!(cb, length(param_types))

    # Build the unified arg_flat_values map
    # Collect values by (arg_idx, field) key
    field_values = Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}()

    for (param_idx, val) in enumerate(arg_values)
        key = param_mapping[param_idx]
        if !haskey(field_values, key)
            field_values[key] = Value[]
        end
        push!(field_values[key], val)
    end

    # Store in Translation and set up slot/argument mappings
    for (key, values) in field_values
        arg_idx, field = key
        tr.arg_flat_values[key] = values

        if field === nothing
            # Regular argument - also set up traditional slot/arg mappings
            @assert length(values) == 1
            val = values[1]
            tr[SlotNumber(arg_idx + 1)] = val
            tr[Argument(arg_idx + 1)] = val
            set_julia_type!(tr, val, target.argtypes[arg_idx])
        end
    end

    # Ghost type arguments don't need explicit handling - their constant values
    # flow through Julia's IR as QuoteNodes and are handled by emit_constant!

    # Create token for memory ordering (like Python's make_token())
    token_type = Token(tt)
    tr.token = encode_MakeTokenOp!(cb, token_type)

    # Emit each statement
    code_stmts = code(target)
    types = ssatypes(target)

    for (i, stmt) in enumerate(code_stmts)
        result_type = types[i]
        emit_statement!(tr, target, stmt, i, result_type)
    end

    finalize_function!(func_buf, cb, writer.debug_info)
end

"""
    emit_statement!(tr, target, stmt, idx, result_type)

Emit bytecode for a single SSA statement.
"""
function emit_statement!(tr::Translation, target::TileTarget,
                         @nospecialize(stmt), idx::Int, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # Handle different statement types
    if stmt isa ReturnNode
        emit_return!(tr, stmt)
    elseif stmt isa GotoNode
        # Simple goto - Tile IR uses structured control flow, skip for now
        # (will need restructuring pass for real control flow)
    elseif stmt isa GotoIfNot
        # Conditional goto - needs restructuring
        error("Control flow not yet supported: GotoIfNot")
    elseif stmt isa Expr
        val = emit_expr!(tr, target, stmt, idx, result_type)
        if val !== nothing
            tr[SSAValue(idx)] = val
        end
    elseif stmt isa GlobalRef
        # Global reference - might be a function or constant
        # In unoptimized IR, these are loaded as values but we don't emit anything
        # They're resolved when used in calls
    elseif stmt isa QuoteNode
        # Quoted literal - emit as constant
        val = emit_constant!(tr, stmt.value, result_type)
        if val !== nothing
            tr[SSAValue(idx)] = val
        end
    elseif stmt isa SlotNumber
        # Slot reference - copy the slot's value to this SSA position
        # This allows later SSA references to find the value
        slot_val = tr[stmt]
        if slot_val !== nothing
            tr[SSAValue(idx)] = slot_val
        end
    elseif stmt === nothing
        # No-op
    else
        @warn "Unhandled statement type" typeof(stmt) stmt
    end
end

"""
    emit_return!(tr, node::ReturnNode)

Emit a return operation.
"""
function emit_return!(tr::Translation, node::ReturnNode)
    cb = tr.code_builder

    # Check if val is defined (unreachable returns don't have val)
    if !isdefined(node, :val) || node.val === nothing || (node.val isa GlobalRef && node.val.name === :nothing)
        encode_ReturnOp!(cb, Value[])
    else
        val = resolve_value(tr, node.val)
        if val !== nothing
            encode_ReturnOp!(cb, [val])
        else
            # Try to emit as constant
            encode_ReturnOp!(cb, Value[])
        end
    end
end

"""
    emit_expr!(tr, target, expr::Expr, idx, result_type) -> Union{Value, Nothing}

Emit bytecode for an expression.
"""
function emit_expr!(tr::Translation, target::TileTarget,
                    expr::Expr, idx::Int, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if expr.head === :call
        return emit_call!(tr, target, expr, result_type)
    elseif expr.head === :invoke
        return emit_invoke!(tr, target, expr, result_type)
    elseif expr.head === :(=)
        # Assignment expression: (_slot = value)
        # In unoptimized IR, this stores a value to a slot
        lhs = expr.args[1]
        rhs = expr.args[2]

        # Emit the RHS
        val = emit_rhs!(tr, target, rhs, idx, result_type)

        # Store to the slot if LHS is a SlotNumber
        if lhs isa SlotNumber && val !== nothing
            tr[lhs] = val
        end

        return val
    elseif expr.head === :new
        # Struct construction - skip for now
        return nothing
    elseif expr.head === :foreigncall
        error("Foreign calls not supported in Tile IR")
    elseif expr.head === :boundscheck
        # Bounds checking - skip in GPU code
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

"""
    emit_rhs!(tr, target, rhs, idx, result_type) -> Union{Value, Nothing}

Emit bytecode for the right-hand side of an assignment.
"""
function emit_rhs!(tr::Translation, target::TileTarget,
                   @nospecialize(rhs), idx::Int, @nospecialize(result_type))
    if rhs isa Expr
        return emit_expr!(tr, target, rhs, idx, result_type)
    elseif rhs isa SSAValue
        return tr[rhs]
    elseif rhs isa SlotNumber
        return tr[rhs]
    elseif rhs isa GlobalRef
        # Global reference being assigned - resolve but don't emit
        return nothing
    elseif rhs isa QuoteNode
        return emit_constant!(tr, rhs.value, result_type)
    else
        # Literal value - emit as constant
        return emit_constant!(tr, rhs, result_type)
    end
end

"""
    emit_call!(tr, target, expr::Expr, result_type) -> Union{Value, Nothing}

Emit bytecode for a function call.
"""
function emit_call!(tr::Translation, target::TileTarget,
                    expr::Expr, @nospecialize(result_type))
    args = expr.args
    f = args[1]

    # Check if this is a kwcall
    func = resolve_function_in_ir(f, target)

    if func === Core.kwcall
        # kwcall(kwargs_namedtuple, func, positional_args...)
        # args[1] = Core.kwcall
        # args[2] = SSA value or expr for kwargs NamedTuple
        # args[3] = the actual function
        # args[4:end] = positional arguments
        return emit_kwcall!(tr, target, args, result_type)
    end

    call_args = args[2:end]

    result = emit_known_call!(tr, target, func, call_args, result_type)
    result === missing && error("Unknown function call: $func with args $call_args")
    return result
end

"""
    emit_kwcall!(tr, target, args, result_type) -> Union{Value, Nothing}

Emit bytecode for a keyword argument function call (Core.kwcall).
"""
function emit_kwcall!(tr::Translation, target::TileTarget,
                      args::Vector{Any}, @nospecialize(result_type))
    # args[1] = Core.kwcall
    # args[2] = kwargs NamedTuple (or SSA value)
    # args[3] = the actual function
    # args[4:end] = positional arguments

    kwargs_ref = args[2]
    func_ref = args[3]
    positional_args = args[4:end]

    # Resolve the actual function
    func = resolve_function_in_ir(func_ref, target)

    # Extract kwargs from the NamedTuple construction
    kwargs = extract_kwargs_from_ir(kwargs_ref, target)

    # For cuTile.load and cuTile.store, convert kwargs to the expected format
    if func === load
        return emit_load_kwcall!(tr, target, positional_args, kwargs, result_type)
    elseif func === store
        return emit_store_kwcall!(tr, target, positional_args, kwargs, result_type)
    else
        # Try calling the function with combined args
        # This is a fallback - most functions should be handled explicitly
        combined_args = vcat(positional_args, collect(values(kwargs)))
        result = emit_known_call!(tr, target, func, combined_args, result_type)
        result === missing && error("Unknown kwcall: $func with args $positional_args, kwargs $kwargs")
        return result
    end
end

"""
    extract_kwargs_from_ir(ref, target) -> Dict{Symbol, Any}

Extract keyword arguments from an IR reference to a NamedTuple.
"""
function extract_kwargs_from_ir(@nospecialize(ref), target::TileTarget)
    kwargs = Dict{Symbol, Any}()

    if ref isa SSAValue
        # Look up the SSA value in the code
        stmt = code(target)[ref.id]
        if stmt isa Expr && stmt.head === :new
            # %new(@NamedTuple{...}, val1, val2, ...)
            nt_type = stmt.args[1]
            values = stmt.args[2:end]

            # Extract field names from the NamedTuple type
            if nt_type isa Type && nt_type <: NamedTuple
                names = fieldnames(nt_type)
                for (name, val) in zip(names, values)
                    kwargs[name] = val
                end
            end
        end
    end

    return kwargs
end

"""
    emit_load_kwcall!(tr, target, positional_args, kwargs, result_type) -> Value

Emit load call with keyword arguments.
"""
function emit_load_kwcall!(tr::Translation, target::TileTarget,
                           positional_args::Vector{Any},
                           kwargs::Dict{Symbol, Any},
                           @nospecialize(result_type))
    # positional_args[1] = array
    # kwargs[:index] = index tuple
    # kwargs[:shape] = shape tuple

    array_arg = positional_args[1]
    index_arg = get(kwargs, :index, nothing)
    shape_arg = get(kwargs, :shape, nothing)

    if index_arg === nothing || shape_arg === nothing
        error("load() requires index and shape keyword arguments")
    end

    # Call emit_load! with the arguments in the expected order
    combined_args = Any[array_arg, index_arg, shape_arg]
    return emit_load!(tr, target, combined_args, result_type)
end

"""
    emit_store_kwcall!(tr, target, positional_args, kwargs, result_type) -> Value

Emit store call with keyword arguments.
"""
function emit_store_kwcall!(tr::Translation, target::TileTarget,
                            positional_args::Vector{Any},
                            kwargs::Dict{Symbol, Any},
                            @nospecialize(result_type))
    # positional_args[1] = array
    # kwargs[:index] = index tuple
    # kwargs[:tile] = tile value

    array_arg = positional_args[1]
    index_arg = get(kwargs, :index, nothing)
    tile_arg = get(kwargs, :tile, nothing)

    if index_arg === nothing || tile_arg === nothing
        error("store() requires index and tile keyword arguments")
    end

    # Call emit_store! with the arguments in the expected order
    combined_args = Any[array_arg, index_arg, tile_arg]
    return emit_store!(tr, target, combined_args, result_type)
end

"""
    resolve_function_in_ir(f, target) -> Any

Resolve a function reference, handling SSAValues that reference GlobalRefs in the IR.
"""
function resolve_function_in_ir(@nospecialize(f), target::TileTarget)
    if f isa SSAValue
        # Look up what this SSA value refers to in the code
        stmt = code(target)[f.id]
        if stmt isa GlobalRef
            return getfield(stmt.mod, stmt.name)
        elseif stmt isa QuoteNode
            return stmt.value
        end
        # Might be a computed function - fall through
    end
    return resolve_function(f)
end

"""
    emit_invoke!(tr, target, expr::Expr, result_type) -> Union{Value, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(tr::Translation, target::TileTarget,
                      expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    f = expr.args[2]
    call_args = expr.args[3:end]

    # Resolve the function from the GlobalRef
    func = resolve_function(f)

    result = emit_known_call!(tr, target, func, call_args, result_type)
    result === missing && error("Unknown function call: $func with args $call_args")
    return result
end

"""
    resolve_function(f) -> Any

Resolve a function reference to its actual value.
"""
function resolve_function(@nospecialize(f))
    if f isa GlobalRef
        return getfield(f.mod, f.name)
    elseif f isa QuoteNode
        return f.value
    else
        return f
    end
end

"""
    emit_known_call!(tr, func, args, result_type) -> Union{Value, Nothing, Missing}

Emit bytecode for a known function call.
Returns `missing` if the function is not recognized.
"""
function emit_known_call!(tr::Translation, target::TileTarget, @nospecialize(func),
                          args::AbstractVector, @nospecialize(result_type))
    # cuTile intrinsics
    func === bid && return emit_bid!(tr, args, result_type)
    func === load && return emit_load!(tr, target, args, result_type)
    func === store && return emit_store!(tr, target, args, result_type)
    func === num_blocks && return emit_num_blocks!(tr, args, result_type)
    func === tile_add && return emit_tile_add!(tr, args, result_type)
    func === tile_sub && return emit_tile_sub!(tr, args, result_type)
    func === tile_mul && return emit_tile_mul!(tr, args, result_type)
    func === transpose && return emit_transpose!(tr, args, result_type)

    # Matrix multiply-accumulate
    func === mma && return emit_mma!(tr, args, result_type)

    # Tile construction
    func === full && return emit_full!(tr, target, args, result_type)

    # Array dimension operations
    func === num_tiles && return emit_num_tiles!(tr, target, args, result_type)

    # Integer arithmetic
    func === cdiv && return emit_cdiv!(tr, target, args, result_type)
    func === floordiv && return emit_floordiv!(tr, target, args, result_type)

    # Core intrinsics
    func === Core.tuple && return nothing
    func === Base.getfield && return emit_getfield!(tr, target, args, result_type)

    # Arithmetic operators on Tiles (when not inlined to tile_add etc.)
    func === Base.:(+) && return emit_tile_add!(tr, args, result_type)
    func === Base.:(-) && return emit_tile_sub!(tr, args, result_type)
    func === Base.:(*) && return emit_tile_mul!(tr, args, result_type)

    return missing
end

"""
    emit_getfield!(tr, target, args, result_type) -> Union{Value, Vector{Value}, Nothing}

Handle getfield for destructured struct arguments.
Returns the flat value(s) for the accessed field, or nothing if not a destructured arg.
"""
function emit_getfield!(tr::Translation, target::TileTarget,
                        args::AbstractVector, @nospecialize(result_type))
    # getfield(obj, field) - args[1] is the object, args[2] is the field name
    if length(args) < 2
        return nothing
    end

    obj_arg = args[1]
    field_arg = args[2]

    # Extract the field name
    field_name = extract_field_symbol(field_arg, target)
    if field_name === nothing
        return nothing
    end

    # Check if obj is a SlotNumber or Argument referencing a destructured arg
    arg_idx = extract_argument_index(obj_arg)
    if arg_idx === nothing
        return nothing
    end

    # Look up the flat values for this field
    values = get_arg_flat_values(tr, arg_idx, field_name)
    if values === nothing
        return nothing
    end

    # Return single value or first value of tuple field
    # (tuple indexing would need additional handling)
    if length(values) == 1
        return values[1]
    else
        # For tuple fields, we return the first value for now
        # Full tuple support would require tracking the tuple as a whole
        return values[1]
    end
end

"""
    extract_field_symbol(arg, target) -> Union{Symbol, Nothing}

Extract a field name symbol from an argument (QuoteNode or SSA reference).
"""
function extract_field_symbol(@nospecialize(arg), target::TileTarget)
    if arg isa QuoteNode && arg.value isa Symbol
        return arg.value
    elseif arg isa Core.SSAValue
        stmt = code(target)[arg.id]
        if stmt isa QuoteNode && stmt.value isa Symbol
            return stmt.value
        end
    elseif arg isa Symbol
        return arg
    end
    return nothing
end

"""
    extract_argument_index(arg) -> Union{Int, Nothing}

Extract the argument index from a SlotNumber or Argument.
Returns the 1-based argument index (slot index - 1).
"""
function extract_argument_index(@nospecialize(arg))
    if arg isa SlotNumber
        # Slot 2 corresponds to arg 1, slot 3 to arg 2, etc.
        return arg.id - 1
    elseif arg isa Argument
        return arg.n - 1
    end
    return nothing
end

#=============================================================================
 cuTile Intrinsic Emitters
=============================================================================#

"""
    emit_bid!(tr, args, result_type) -> Value

Emit GetTileBlockIdOp for bid(axis).
"""
function emit_bid!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # bid(axis) - axis should be a constant 0, 1, or 2
    if length(args) != 1
        error("bid() requires exactly 1 argument (axis)")
    end

    axis_val = args[1]
    axis = extract_constant_int(axis_val)
    if axis === nothing
        error("bid() axis must be a compile-time constant")
    end
    if !(axis in (0, 1, 2))
        error("bid() axis must be 0, 1, or 2, got $axis")
    end

    # Result type is Int32 scalar tile
    res_type = tile_type!(tt, I32(tt), Int[])

    # GetTileBlockIdOp returns (x, y, z) - we select the one we want
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(cb, res_type, res_type, res_type)

    result = (bid_x, bid_y, bid_z)[axis + 1]

    # Track which grid axis this value came from
    set_grid_axis!(tr, result, axis)

    return result
end

"""
    emit_num_blocks!(tr, args, result_type) -> Value

Emit GetNumTileBlocksOp for num_blocks(axis).
"""
function emit_num_blocks!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) != 1
        error("num_blocks() requires exactly 1 argument (axis)")
    end

    axis_val = args[1]
    axis = extract_constant_int(axis_val)
    if axis === nothing
        error("num_blocks() axis must be a compile-time constant")
    end
    if !(axis in (0, 1, 2))
        error("num_blocks() axis must be 0, 1, or 2, got $axis")
    end

    res_type = tile_type!(tt, I32(tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, res_type, res_type, res_type)

    return (nb_x, nb_y, nb_z)[axis + 1]
end

"""
    extract_pointer_element_type(tr::Translation, val::Value) -> Type

Extract the element type from a pointer value's Julia type.
Returns Float32 as fallback if type cannot be determined.
"""
function extract_pointer_element_type(tr::Translation, val::Value)
    jltype = get_julia_type(tr, val)
    if jltype !== nothing && jltype <: Ptr
        return eltype(jltype)
    end
    # Fallback to Float32
    return Float32
end

"""
    get_array_spec(T::Type{<:TileArray}) -> Union{ArraySpec, Nothing}

Extract the ArraySpec from a TileArray type parameter.
Returns nothing if the type doesn't have an ArraySpec.
"""
function get_array_spec(@nospecialize(T))
    if T <: TileArray
        # Handle UnionAll types (like TileArray{Float32, 2})
        base_type = get_base_type(T)
        if base_type isa DataType && length(base_type.parameters) >= 3
            S = base_type.parameters[3]
            if S isa ArraySpec  # Works for ArraySpec{N} since it's still <: ArraySpec
                return S
            end
        end
    end
    return nothing
end


"""
    emit_assume_aligned!(cb, tt, ptr_val, ptr_type, alignment) -> Value

Emit an AssumeOp asserting the pointer is aligned to `alignment` bytes.
Returns the assumed pointer value.
"""
function emit_assume_aligned!(cb::CodeBuilder, tt::TypeTable, ptr_val::Value, ptr_type::TypeId, alignment::Int)
    encode_AssumeOp!(cb, ptr_type, ptr_val, DivBy(alignment))
end

"""
    emit_assume_bounded!(cb, tt, val, val_type; lb=nothing, ub=nothing) -> Value

Emit an AssumeOp asserting the value is within bounds.
Returns the assumed value.
"""
function emit_assume_bounded!(cb::CodeBuilder, tt::TypeTable, val::Value, val_type::TypeId; lb=nothing, ub=nothing)
    encode_AssumeOp!(cb, val_type, val, Bounded(lb, ub))
end

"""
    emit_assume_divisible!(cb, tt, val, val_type, divisor) -> Value

Emit an AssumeOp asserting the value is divisible by `divisor`.
Returns the assumed value.
"""
function emit_assume_divisible!(cb::CodeBuilder, tt::TypeTable, val::Value, val_type::TypeId, divisor::Int)
    encode_AssumeOp!(cb, val_type, val, DivBy(divisor))
end

"""
    emit_load!(tr, args, result_type) -> Value

Emit load operation for ct.load(array; index, shape).
This creates: TensorView -> PartitionView -> LoadViewTkoOp.

Handles both raw pointer arguments and TileArray arguments.
For TileArray, uses the sizes/strides from the destructured parameters,
and emits AssumeOp for alignment hints based on ArraySpec.
"""
function emit_load!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # load(array, index, shape)
    if length(args) < 1
        error("load() requires at least an array argument")
    end

    array_arg = args[1]

    # Check if this is a TileArray argument (destructured)
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(tr, arg_idx)

    # Get pointer value and ArraySpec for specialization
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(tr, arg_idx, :ptr)
        if ptr_vals === nothing || isempty(ptr_vals)
            error("Cannot get ptr from TileArray argument")
        end
        array_val = ptr_vals[1]
        # Get element type and ArraySpec from the TileArray type
        tilearray_type = get_arg_type(tr, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        array_val = resolve_value(tr, array_arg)
        if array_val === nothing
            error("Cannot resolve array argument for load()")
        end
        elem_type = extract_pointer_element_type(tr, array_val)
    end

    # Parse shape argument first to know dimensions
    tile_shape = Int[16]  # Default
    if length(args) >= 3
        shape_arg = args[3]
        shape = extract_constant_tuple_from_arg(shape_arg, tr, target)
        if shape !== nothing
            tile_shape = collect(Int, shape)
        end
    end

    ndim = length(tile_shape)

    # Parse index argument - handle tuple indices for multi-dimensional loads
    index_vals = Value[]
    if length(args) >= 2
        index_arg = args[2]
        if index_arg isa Core.SSAValue
            # Look up the tuple construction to get the element SSA values
            tuple_stmt = code(target)[index_arg.id]
            is_tuple_call = tuple_stmt isa Expr && tuple_stmt.head === :call &&
                (tuple_stmt.args[1] isa GlobalRef && tuple_stmt.args[1].mod === Core && tuple_stmt.args[1].name === :tuple)
            if is_tuple_call
                # Core.tuple(arg1, arg2, ...) - extract each arg
                for elem_arg in tuple_stmt.args[2:end]
                    if elem_arg isa Core.SSAValue && haskey(tr.results, elem_arg.id)
                        push!(index_vals, tr.results[elem_arg.id])
                    end
                end
            elseif haskey(tr.results, index_arg.id)
                # Single value (1D case)
                push!(index_vals, tr.results[index_arg.id])
            end
        else
            v = resolve_value(tr, index_arg)
            if v !== nothing
                push!(index_vals, v)
            end
        end
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create types
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # TensorView type (ndim with dynamic shape/stride)
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = fill(DYNAMIC_SHAPE, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals = Value[]
    stride_vals = Value[]

    if is_tilearray
        # Use sizes and strides from the TileArray parameter
        sizes_from_arg = get_arg_flat_values(tr, arg_idx, :sizes)
        strides_from_arg = get_arg_flat_values(tr, arg_idx, :strides)

        if sizes_from_arg !== nothing && length(sizes_from_arg) >= ndim
            size_vals = sizes_from_arg[1:ndim]
        end
        if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
            stride_vals = strides_from_arg[1:ndim]
        end
    end

    # Fall back to computing from grid if not available
    if isempty(size_vals)
        nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)
        grid_sizes = [nb_x, nb_y, nb_z]

        for dim in 1:ndim
            tile_size_bytes = reinterpret(UInt8, [Int32(tile_shape[dim])])
            tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(tile_size_bytes))
            size_val = encode_MulIOp!(cb, scalar_i32, grid_sizes[dim], tile_size_val)
            push!(size_vals, size_val)
        end
    end

    if isempty(stride_vals)
        # Compute strides for column-major order (Julia/Fortran)
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [Int32(1)])
                stride_val = encode_ConstantOp!(cb, scalar_i32, collect(stride_bytes))
            else
                stride_val = encode_MulIOp!(cb, scalar_i32, stride_vals[end], size_vals[dim-1])
            end
            push!(stride_vals, stride_val)
        end
    end

    # Emit AssumeOps based on ArraySpec for optimization hints
    if array_spec !== nothing
        # Get the pointer type for AssumeOp (0-D tile of ptr<dtype>)
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, Int[])

        # If aligned, assert pointer alignment
        if array_spec.alignment > 0
            array_val = emit_assume_aligned!(cb, tt, array_val, ptr_tile_type, array_spec.alignment)
        end

        # Add bounds assumes for sizes and strides (non-negative)
        size_vals = [emit_assume_bounded!(cb, tt, v, scalar_i32; lb=0) for v in size_vals]
        stride_vals = [emit_assume_bounded!(cb, tt, v, scalar_i32; lb=0) for v in stride_vals]

        # Add divisibility assumes for sizes (enables tile boundary optimization)
        if hasproperty(array_spec, :shape_div_by)
            for (i, div_by) in enumerate(array_spec.shape_div_by)
                if div_by > 0 && i <= length(size_vals)
                    size_vals[i] = emit_assume_divisible!(cb, tt, size_vals[i], scalar_i32, div_by)
                end
            end
        end

        # Add divisibility assumes for strides (enables vectorized access)
        if hasproperty(array_spec, :stride_div_by)
            for (i, div_by) in enumerate(array_spec.stride_div_by)
                if div_by > 0 && i <= length(stride_vals)
                    stride_vals[i] = emit_assume_divisible!(cb, tt, stride_vals[i], scalar_i32, div_by)
                end
            end
        end
    end

    # Create tensor view from pointer
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Use provided indices or default to zeros
    if length(index_vals) < ndim
        idx_type = tile_type!(tt, I32(tt), Int[])
        while length(index_vals) < ndim
            idx_bytes = reinterpret(UInt8, [Int32(0)])
            push!(index_vals, encode_ConstantOp!(cb, idx_type, collect(idx_bytes)))
        end
    end

    # Load tile with token for memory ordering
    tile_val, out_token = encode_LoadViewTkoOp!(cb, tile_type, token_type, partition, index_vals;
                                                 token=tr.token)

    # Track the type and shape of this value for later operations
    set_value_type!(tr, tile_val, tile_type)
    set_tile_shape!(tr, tile_val, tile_shape)

    return tile_val
end

"""
    extract_constant_tuple_from_arg(arg, tr, target) -> Union{Tuple, Nothing}

Extract a constant tuple value from an argument.
Handles direct tuples, QuoteNodes, and SSA references to tuple constructions.

Note: With ghost type filtering, constant values from Val{V} parameters
flow through Julia's IR as QuoteNodes, so this function extracts them naturally.
This also handles tuples of Constant{T, V} instances.
"""
function extract_constant_tuple_from_arg(@nospecialize(arg), tr::Translation, target::TileTarget)
    if arg isa Tuple
        # Check if elements are Constant instances - extract their values
        elements = Any[]
        for elem in arg
            val = extract_value_from_quote(elem)
            if val === nothing
                # Try as plain value
                if elem isa Number
                    push!(elements, elem)
                else
                    return nothing
                end
            else
                push!(elements, val)
            end
        end
        return Tuple(elements)
    elseif arg isa QuoteNode && arg.value isa Tuple
        # Recursively extract with element handling
        return extract_constant_tuple_from_arg(arg.value, tr, target)
    elseif arg isa Core.SSAValue
        # Check if this is a tuple construction with constant elements
        stmt = code(target)[arg.id]
        if stmt isa QuoteNode && stmt.value isa Tuple
            return stmt.value
        elseif stmt isa Expr && stmt.head === :call
            callee = stmt.args[1]
            if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
                # This is Core.tuple(...) - extract each element
                elements = Any[]
                for elem in stmt.args[2:end]
                    val = extract_constant_value_from_ir(elem, target)
                    if val === nothing
                        return nothing  # Not all elements are constants
                    end
                    push!(elements, val)
                end
                return Tuple(elements)
            end
        end
    end
    return nothing
end

"""
    extract_constant_value_from_ir(arg, target) -> Union{Number, Nothing}

Extract a constant value from IR. Handles:
- Literal integer values
- QuoteNode wrapped values
- SSA values that are QuoteNodes in the IR
- Constant{T, V} instances (ghost types with embedded values)
"""
function extract_constant_value_from_ir(@nospecialize(arg), target::TileTarget)
    if arg isa Integer
        return arg
    elseif arg isa AbstractFloat
        return arg
    elseif arg isa QuoteNode
        return extract_value_from_quote(arg.value)
    elseif arg isa Core.SSAValue
        # Look up what this SSA value is in the code
        stmt = code(target)[arg.id]
        if stmt isa QuoteNode
            return extract_value_from_quote(stmt.value)
        elseif stmt isa Integer
            return stmt
        elseif stmt isa AbstractFloat
            return stmt
        end
    end
    return nothing
end

"""
    extract_value_from_quote(val) -> Union{Number, Nothing}

Extract a numeric value from a value that might be wrapped in Constant{T, V}.
"""
function extract_value_from_quote(@nospecialize(val))
    if val isa Integer || val isa AbstractFloat
        return val
    end

    # Check if it's a Constant{T, V} instance - extract V from type parameter
    T = typeof(val)
    if T isa DataType && T.name.name === :Constant && length(T.parameters) >= 2
        # Constant{Type, Value} - the value is in the second type parameter
        return T.parameters[2]
    end

    return nothing
end

# Backward-compatible version for calls that don't have target
function extract_constant_tuple_from_arg(@nospecialize(arg), tr::Translation)
    if arg isa Tuple
        return arg
    elseif arg isa QuoteNode && arg.value isa Tuple
        return arg.value
    end
    return nothing
end

"""
    emit_store!(tr, args, result_type)

Emit store operation for ct.store(array, index, tile).
Creates: TensorView -> PartitionView -> StoreViewTkoOp.

Handles both raw pointer arguments and TileArray arguments.
For TileArray, uses the sizes/strides from the destructured parameters,
and emits AssumeOp for alignment hints based on ArraySpec.
"""
function emit_store!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    # store(array, index, tile)
    if length(args) < 3
        error("store() requires array, index, and tile arguments")
    end

    array_arg = args[1]

    # Check if this is a TileArray argument (destructured)
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(tr, arg_idx)

    # Get pointer value and ArraySpec for specialization
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(tr, arg_idx, :ptr)
        if ptr_vals === nothing || isempty(ptr_vals)
            error("Cannot get ptr from TileArray argument")
        end
        array_val = ptr_vals[1]
        # Get element type and ArraySpec from the TileArray type
        tilearray_type = get_arg_type(tr, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        array_val = resolve_value(tr, array_arg)
        if array_val === nothing
            error("Cannot resolve array argument for store()")
        end
        elem_type = extract_pointer_element_type(tr, array_val)
    end

    # Parse tile argument to get its shape
    tile_val = resolve_value(tr, args[3])
    if tile_val === nothing
        error("store() requires a tile argument")
    end

    # Get tile shape from the tracked tile shapes
    tile_shape = get_tile_shape(tr, tile_val)
    if tile_shape === nothing
        error("Cannot determine tile shape for store() - tile value has no tracked shape")
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    ndim = length(tile_shape)

    # Parse index argument - handle tuple indices for multi-dimensional stores
    index_vals = Value[]
    index_arg = args[2]

    if index_arg isa Core.SSAValue
        # Look up the tuple construction to get the element SSA values
        tuple_stmt = code(target)[index_arg.id]
        is_tuple_call = tuple_stmt isa Expr && tuple_stmt.head === :call &&
            (tuple_stmt.args[1] isa GlobalRef && tuple_stmt.args[1].mod === Core && tuple_stmt.args[1].name === :tuple)
        if is_tuple_call
            # Core.tuple(arg1, arg2, ...) - extract each arg
            for elem_arg in tuple_stmt.args[2:end]
                if elem_arg isa Core.SSAValue && haskey(tr.results, elem_arg.id)
                    push!(index_vals, tr.results[elem_arg.id])
                end
            end
        elseif haskey(tr.results, index_arg.id)
            # Single value (1D case)
            push!(index_vals, tr.results[index_arg.id])
        end
    else
        v = resolve_value(tr, index_arg)
        if v !== nothing
            push!(index_vals, v)
        end
    end

    # TensorView type (ndim with dynamic shape/stride)
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = fill(DYNAMIC_SHAPE, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals = Value[]
    stride_vals = Value[]

    if is_tilearray
        # Use sizes and strides from the TileArray parameter
        sizes_from_arg = get_arg_flat_values(tr, arg_idx, :sizes)
        strides_from_arg = get_arg_flat_values(tr, arg_idx, :strides)

        if sizes_from_arg !== nothing && length(sizes_from_arg) >= ndim
            size_vals = sizes_from_arg[1:ndim]
        end
        if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
            stride_vals = strides_from_arg[1:ndim]
        end
    end

    # Fall back to computing from grid if not available
    if isempty(size_vals)
        nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)
        all_grid_sizes = [nb_x, nb_y, nb_z]

        # Determine which grid dimension corresponds to each array dimension
        index_grid_axes = Int[]
        for (i, idx_val) in enumerate(index_vals)
            grid_axis = get_grid_axis(tr, idx_val)
            if grid_axis !== nothing
                push!(index_grid_axes, grid_axis)
            else
                push!(index_grid_axes, i - 1)
            end
        end

        for dim in 1:ndim
            grid_axis = dim <= length(index_grid_axes) ? index_grid_axes[dim] : dim - 1
            grid_size = all_grid_sizes[grid_axis + 1]

            tile_size_bytes = reinterpret(UInt8, [Int32(tile_shape[dim])])
            tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(tile_size_bytes))
            size_val = encode_MulIOp!(cb, scalar_i32, grid_size, tile_size_val)
            push!(size_vals, size_val)
        end
    end

    if isempty(stride_vals)
        # Compute strides for column-major order (Julia/Fortran)
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [Int32(1)])
                stride_val = encode_ConstantOp!(cb, scalar_i32, collect(stride_bytes))
            else
                stride_val = encode_MulIOp!(cb, scalar_i32, stride_vals[end], size_vals[dim-1])
            end
            push!(stride_vals, stride_val)
        end
    end

    # Emit AssumeOps based on ArraySpec for optimization hints
    if array_spec !== nothing
        # Get the pointer type for AssumeOp (0-D tile of ptr<dtype>)
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, Int[])

        # If aligned, assert pointer alignment
        if array_spec.alignment > 0
            array_val = emit_assume_aligned!(cb, tt, array_val, ptr_tile_type, array_spec.alignment)
        end

        # Add bounds assumes for sizes and strides (non-negative)
        size_vals = [emit_assume_bounded!(cb, tt, v, scalar_i32; lb=0) for v in size_vals]
        stride_vals = [emit_assume_bounded!(cb, tt, v, scalar_i32; lb=0) for v in stride_vals]

        # Add divisibility assumes for sizes (enables tile boundary optimization)
        if hasproperty(array_spec, :shape_div_by)
            for (i, div_by) in enumerate(array_spec.shape_div_by)
                if div_by > 0 && i <= length(size_vals)
                    size_vals[i] = emit_assume_divisible!(cb, tt, size_vals[i], scalar_i32, div_by)
                end
            end
        end

        # Add divisibility assumes for strides (enables vectorized access)
        if hasproperty(array_spec, :stride_div_by)
            for (i, div_by) in enumerate(array_spec.stride_div_by)
                if div_by > 0 && i <= length(stride_vals)
                    stride_vals[i] = emit_assume_divisible!(cb, tt, stride_vals[i], scalar_i32, div_by)
                end
            end
        end
    end

    # Create tensor view from pointer
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Use provided indices or default to zeros
    if length(index_vals) < ndim
        idx_type = tile_type!(tt, I32(tt), Int[])
        while length(index_vals) < ndim
            idx_bytes = reinterpret(UInt8, [Int32(0)])
            push!(index_vals, encode_ConstantOp!(cb, idx_type, collect(idx_bytes)))
        end
    end

    # Store tile with token for memory ordering
    token_type = Token(tt)
    encode_StoreViewTkoOp!(cb, token_type, tile_val, partition, index_vals;
                           token=tr.token)

    return nothing
end

#=============================================================================
 Tile Arithmetic Emitters
=============================================================================#

"""
    emit_tile_add!(tr, args, result_type) -> Value

Emit AddFOp/AddIOp for tile_add(a, b).
"""
function emit_tile_add!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder

    if length(args) != 2
        error("tile_add() requires exactly 2 arguments")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for tile_add()")
    end

    # Get the type from the first operand (they should match)
    tile_type = get_value_type(tr, lhs)
    if tile_type === nothing
        error("Cannot determine tile type for tile_add()")
    end

    # Determine if float or int based on result_type
    elem_type = extract_tile_element_type(result_type)
    if elem_type <: AbstractFloat
        result = encode_AddFOp!(cb, tile_type, lhs, rhs)
    else
        result = encode_AddIOp!(cb, tile_type, lhs, rhs)
    end

    set_value_type!(tr, result, tile_type)
    # Propagate tile shape
    shape = get_tile_shape(tr, lhs)
    if shape !== nothing
        set_tile_shape!(tr, result, shape)
    end

    return result
end

"""
    emit_tile_sub!(tr, args, result_type) -> Value

Emit SubFOp/SubIOp for tile_sub(a, b).
"""
function emit_tile_sub!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder

    if length(args) != 2
        error("tile_sub() requires exactly 2 arguments")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for tile_sub()")
    end

    tile_type = get_value_type(tr, lhs)
    if tile_type === nothing
        error("Cannot determine tile type for tile_sub()")
    end

    elem_type = extract_tile_element_type(result_type)
    if elem_type <: AbstractFloat
        result = encode_SubFOp!(cb, tile_type, lhs, rhs)
    else
        result = encode_SubIOp!(cb, tile_type, lhs, rhs)
    end

    set_value_type!(tr, result, tile_type)
    # Propagate tile shape
    shape = get_tile_shape(tr, lhs)
    if shape !== nothing
        set_tile_shape!(tr, result, shape)
    end

    return result
end

"""
    emit_tile_mul!(tr, args, result_type) -> Value

Emit MulFOp/MulIOp for tile_mul(a, b).
"""
function emit_tile_mul!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder

    if length(args) != 2
        error("tile_mul() requires exactly 2 arguments")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])

    if lhs === nothing || rhs === nothing
        error("Cannot resolve operands for tile_mul()")
    end

    tile_type = get_value_type(tr, lhs)
    if tile_type === nothing
        error("Cannot determine tile type for tile_mul()")
    end

    elem_type = extract_tile_element_type(result_type)
    if elem_type <: AbstractFloat
        result = encode_MulFOp!(cb, tile_type, lhs, rhs)
    else
        result = encode_MulIOp!(cb, tile_type, lhs, rhs)
    end

    set_value_type!(tr, result, tile_type)
    # Propagate tile shape
    shape = get_tile_shape(tr, lhs)
    if shape !== nothing
        set_tile_shape!(tr, result, shape)
    end

    return result
end

"""
    emit_transpose!(tr, args, result_type) -> Value

Emit PermuteOp for transpose(tile).
Transpose swaps the last two dimensions, so for a 2D tile it's permutation [1, 0].
"""
function emit_transpose!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) != 1
        error("transpose() requires exactly 1 argument")
    end

    source = resolve_value(tr, args[1])
    if source === nothing
        error("Cannot resolve operand for transpose()")
    end

    # Get the input tile type
    input_tile_type = get_value_type(tr, source)
    if input_tile_type === nothing
        error("Cannot determine tile type for transpose()")
    end

    # Get input shape from tracked tile shapes
    input_shape = get_tile_shape(tr, source)
    if input_shape === nothing
        # Fallback: try to extract from result_type
        output_shape_tuple = extract_tile_shape(result_type)
        if output_shape_tuple !== nothing
            input_shape = collect(Int, reverse(output_shape_tuple))
        else
            error("Cannot determine tile shape for transpose()")
        end
    end

    # Output shape is reversed input shape
    output_shape = reverse(input_shape)

    # Get element type
    elem_type = extract_tile_element_type(result_type)
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create output tile type with transposed shape
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # For 2D transpose, permutation is [1, 0] (swap dimensions)
    # Note: Tile IR uses 0-based indexing for permutation
    ndim = length(output_shape)
    permutation = collect(ndim-1:-1:0)  # Reverse: [1, 0] for 2D

    result = encode_PermuteOp!(cb, output_tile_type, source, permutation)
    set_value_type!(tr, result, output_tile_type)
    set_tile_shape!(tr, result, output_shape)
    return result
end

"""
    extract_tile_shape(result_type) -> Union{Tuple, Nothing}

Extract the shape from a Tile{T, Shape} result type.
"""
function extract_tile_shape(@nospecialize(result_type))
    # Handle Core.Const wrapper
    if result_type isa Core.Const
        result_type = typeof(result_type.val)
    end

    # Check if it's a fully specified Tile type
    if result_type isa DataType && result_type.name.name === :Tile
        if length(result_type.parameters) >= 2
            shape = result_type.parameters[2]
            if shape isa Tuple
                return shape
            end
        end
    end

    return nothing
end

"""
    extract_tile_element_type(result_type) -> Type

Extract the element type from a Tile{T, Shape} result type.
Handles both fully specified types (Tile{Float32, (16,)}) and
partial types (Tile{Float32} which is a UnionAll).
"""
function extract_tile_element_type(@nospecialize(result_type))
    # Handle Core.Const wrapper
    if result_type isa Core.Const
        result_type = typeof(result_type.val)
    end

    # Check if it's a fully specified Tile type
    if result_type isa DataType && result_type.name.name === :Tile
        return result_type.parameters[1]
    end

    # Handle partial Tile{T} (UnionAll where Shape is not specified)
    if result_type isa UnionAll
        body = result_type.body
        if body isa DataType && body.name.name === :Tile && length(body.parameters) >= 1
            elem = body.parameters[1]
            if elem isa Type || elem isa DataType
                return elem
            end
        end
    end

    # Default to Float32
    return Float32
end

#=============================================================================
 Helper functions for extracting constants
=============================================================================#

"""
    extract_constant_int(val) -> Union{Int, Nothing}

Try to extract a constant integer from an SSA value.
"""
function extract_constant_int(@nospecialize(val))
    if val isa Integer
        return Int(val)
    elseif val isa QuoteNode && val.value isa Integer
        return Int(val.value)
    end
    return nothing
end

"""
    extract_constant_tuple(val) -> Union{Tuple, Nothing}

Try to extract a constant tuple from an SSA value.
"""
function extract_constant_tuple(@nospecialize(val))
    if val isa Tuple
        return val
    elseif val isa QuoteNode && val.value isa Tuple
        return val.value
    end
    return nothing
end

"""
    emit_constant!(tr, value, result_type) -> Value

Emit a constant value.
"""
function emit_constant!(tr::Translation, @nospecialize(value), @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    res_type = tile_type_for_julia!(tr, result_type)

    # Convert value to bytes
    bytes = constant_to_bytes(value, result_type)
    return encode_ConstantOp!(cb, res_type, bytes)
end

"""
    constant_to_bytes(value, T) -> Vector{UInt8}

Convert a Julia value to bytes for a Tile IR constant.
"""
function constant_to_bytes(@nospecialize(value), @nospecialize(T::Type))
    if T === Bool
        return UInt8[value ? 0xff : 0x00]
    elseif T === Int32 || T === UInt32
        return reinterpret(UInt8, [Int32(value)])
    elseif T === Int64 || T === UInt64
        return reinterpret(UInt8, [Int64(value)])
    elseif T === Float32
        return reinterpret(UInt8, [Float32(value)])
    elseif T === Float64
        return reinterpret(UInt8, [Float64(value)])
    else
        error("Cannot convert $T to constant bytes")
    end
end

"""
    emit_tileir(f, argtypes; name=nothing) -> Vector{UInt8}

Compile a Julia function to Tile IR bytecode.
"""
function emit_tileir(@nospecialize(f), @nospecialize(argtypes);
                     name::Union{String, Nothing} = nothing)
    target = TileTarget(f, argtypes)

    kernel_name = name === nothing ? string(target.mi.def.name) : name

    buf = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, target; name=kernel_name)
    end

    return buf
end

#=============================================================================
 Matrix Multiply-Accumulate Emitters
=============================================================================#

"""
    emit_mma!(tr, args, result_type) -> Value

Emit MmaFOp for matrix-multiply-accumulate: result = a @ b + acc.
"""
function emit_mma!(tr::Translation, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder

    if length(args) != 3
        error("mma() requires exactly 3 arguments (a, b, acc)")
    end

    lhs = resolve_value(tr, args[1])
    rhs = resolve_value(tr, args[2])
    acc = resolve_value(tr, args[3])

    if lhs === nothing || rhs === nothing || acc === nothing
        error("Cannot resolve operands for mma()")
    end

    # Get the accumulator type - result has same type as accumulator
    acc_type = get_value_type(tr, acc)
    if acc_type === nothing
        error("Cannot determine accumulator type for mma()")
    end

    result = encode_MmaFOp!(cb, acc_type, lhs, rhs, acc)
    set_value_type!(tr, result, acc_type)

    # Propagate tile shape from accumulator
    shape = get_tile_shape(tr, acc)
    if shape !== nothing
        set_tile_shape!(tr, result, shape)
    end

    return result
end

#=============================================================================
 Tile Construction Emitters
=============================================================================#

"""
    emit_full!(tr, target, args, result_type) -> Value

Emit a constant tile filled with a value.
full(shape, value, dtype) creates a tile of zeros/ones/etc.
"""
function emit_full!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) < 2
        error("full() requires at least shape and value arguments")
    end

    # Extract shape (compile-time constant)
    shape_arg = args[1]
    shape = extract_constant_tuple_from_arg(shape_arg, tr, target)
    if shape === nothing
        error("full() shape must be a compile-time constant tuple")
    end
    tile_shape = collect(Int, shape)

    # Extract value (compile-time constant)
    value_arg = args[2]
    value = extract_constant_value_from_ir(value_arg, target)
    if value === nothing
        # Try as a float
        if value_arg isa QuoteNode
            value = value_arg.value
        elseif value_arg isa AbstractFloat || value_arg isa Integer
            value = value_arg
        else
            error("full() value must be a compile-time constant")
        end
    end

    # Extract dtype (from type argument or result type)
    elem_type = Float32  # default
    if length(args) >= 3
        dtype_arg = args[3]
        if dtype_arg isa Type
            elem_type = dtype_arg
        elseif dtype_arg isa QuoteNode && dtype_arg.value isa Type
            elem_type = dtype_arg.value
        end
    end

    # If we can extract element type from result_type, use that
    result_elem_type = extract_tile_element_type(result_type)
    if result_elem_type !== Float32  # not the default fallback
        elem_type = result_elem_type
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create constant bytes for the fill value
    # For a tile constant, we need to:
    # 1. Create a 0-D scalar constant
    # 2. Reshape it to match the target rank (with 1s)
    # 3. Broadcast to the final shape
    scalar_type = tile_type!(tt, dtype, Int[])  # 0-D scalar tile

    # Convert value to bytes
    if elem_type === Float32
        value_bytes = reinterpret(UInt8, [Float32(value)])
    elseif elem_type === Float64
        value_bytes = reinterpret(UInt8, [Float64(value)])
    elseif elem_type === Int32
        value_bytes = reinterpret(UInt8, [Int32(value)])
    elseif elem_type === Int64
        value_bytes = reinterpret(UInt8, [Int64(value)])
    else
        value_bytes = reinterpret(UInt8, [Float32(value)])
    end

    # Create scalar constant
    scalar_val = encode_ConstantOp!(cb, scalar_type, collect(value_bytes))

    # Reshape to match target rank (prepend 1s)
    # e.g., for target shape (64, 64), reshape scalar to (1, 1)
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_type = tile_type!(tt, dtype, ones_shape)
        reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_val)
    else
        reshaped_val = scalar_val
    end

    # Broadcast to tile shape
    result = encode_BroadcastOp!(cb, tile_type, reshaped_val)

    set_value_type!(tr, result, tile_type)
    set_tile_shape!(tr, result, tile_shape)

    return result
end

#=============================================================================
 Array Dimension Operation Emitters
=============================================================================#

"""
    emit_num_tiles!(tr, target, args, result_type) -> Value

Emit code for num_tiles(arr, axis, shape).
Returns cdiv(arr.sizes[axis+1], shape[axis+1]).
"""
function emit_num_tiles!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) < 3
        error("num_tiles() requires arr, axis, and shape arguments")
    end

    array_arg = args[1]
    axis_arg = args[2]
    shape_arg = args[3]

    # Get the axis (0-indexed)
    axis = extract_constant_int(axis_arg)
    if axis === nothing
        # Try from IR
        axis = extract_constant_value_from_ir(axis_arg, target)
    end
    if axis === nothing
        error("num_tiles() axis must be a compile-time constant")
    end

    # Get the tile shape
    shape = extract_constant_tuple_from_arg(shape_arg, tr, target)
    if shape === nothing
        error("num_tiles() shape must be a compile-time constant tuple")
    end

    # Get the tile size for this axis (Julia is 1-indexed internally)
    tile_size = shape[axis + 1]

    # Get the array size for this axis
    arg_idx = extract_argument_index(array_arg)
    if arg_idx === nothing || !is_destructured_arg(tr, arg_idx)
        error("num_tiles() requires a TileArray argument")
    end

    sizes_vals = get_arg_flat_values(tr, arg_idx, :sizes)
    if sizes_vals === nothing || length(sizes_vals) <= axis
        error("Cannot get size for axis $axis from TileArray")
    end

    array_size = sizes_vals[axis + 1]  # 1-indexed in Julia

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Create constant for tile size
    tile_size_bytes = reinterpret(UInt8, [Int32(tile_size)])
    tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(tile_size_bytes))

    # Compute cdiv: (array_size + tile_size - 1) / tile_size
    one_bytes = reinterpret(UInt8, [Int32(1)])
    one_val = encode_ConstantOp!(cb, scalar_i32, collect(one_bytes))

    # array_size + tile_size
    sum1 = encode_AddIOp!(cb, scalar_i32, array_size, tile_size_val)
    # sum1 - 1
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    # (sum1 - 1) / tile_size
    result = encode_DivIOp!(cb, scalar_i32, sum2, tile_size_val; signedness=SignednessSigned, rounding=RoundingZero)

    set_value_type!(tr, result, scalar_i32)
    return result
end

#=============================================================================
 Integer Arithmetic Emitters
=============================================================================#

"""
    emit_cdiv!(tr, target, args, result_type) -> Value

Emit code for ceiling division: cdiv(a, b) = (a + b - 1) / b
"""
function emit_cdiv!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) != 2
        error("cdiv() requires exactly 2 arguments")
    end

    # Try to resolve as values first
    a = resolve_value(tr, args[1])
    b = resolve_value(tr, args[2])

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # If a is not a runtime value, try as constant
    if a === nothing
        a_const = extract_constant_int(args[1])
        if a_const === nothing
            a_const = extract_constant_value_from_ir(args[1], target)
        end
        if a_const !== nothing
            a_bytes = reinterpret(UInt8, [Int32(a_const)])
            a = encode_ConstantOp!(cb, scalar_i32, collect(a_bytes))
        else
            error("Cannot resolve first argument for cdiv()")
        end
    end

    # If b is not a runtime value, try as constant
    if b === nothing
        b_const = extract_constant_int(args[2])
        if b_const === nothing
            b_const = extract_constant_value_from_ir(args[2], target)
        end
        if b_const !== nothing
            b_bytes = reinterpret(UInt8, [Int32(b_const)])
            b = encode_ConstantOp!(cb, scalar_i32, collect(b_bytes))
        else
            error("Cannot resolve second argument for cdiv()")
        end
    end

    # Compute cdiv: (a + b - 1) / b
    one_bytes = reinterpret(UInt8, [Int32(1)])
    one_val = encode_ConstantOp!(cb, scalar_i32, collect(one_bytes))

    # a + b
    sum1 = encode_AddIOp!(cb, scalar_i32, a, b)
    # sum1 - 1
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    # (sum1 - 1) / b
    result = encode_DivIOp!(cb, scalar_i32, sum2, b; signedness=SignednessSigned, rounding=RoundingZero)

    set_value_type!(tr, result, scalar_i32)
    return result
end

"""
    emit_floordiv!(tr, target, args, result_type) -> Value

Emit code for floor division: floordiv(a, b) = a / b (truncated toward zero)
"""
function emit_floordiv!(tr::Translation, target::TileTarget, args::AbstractVector, @nospecialize(result_type))
    cb = tr.code_builder
    tt = tr.type_table

    if length(args) != 2
        error("floordiv() requires exactly 2 arguments")
    end

    a = resolve_value(tr, args[1])
    b = resolve_value(tr, args[2])

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Handle constants
    if a === nothing
        a_const = extract_constant_int(args[1])
        if a_const === nothing
            a_const = extract_constant_value_from_ir(args[1], target)
        end
        if a_const !== nothing
            a_bytes = reinterpret(UInt8, [Int32(a_const)])
            a = encode_ConstantOp!(cb, scalar_i32, collect(a_bytes))
        else
            error("Cannot resolve first argument for floordiv()")
        end
    end

    if b === nothing
        b_const = extract_constant_int(args[2])
        if b_const === nothing
            b_const = extract_constant_value_from_ir(args[2], target)
        end
        if b_const !== nothing
            b_bytes = reinterpret(UInt8, [Int32(b_const)])
            b = encode_ConstantOp!(cb, scalar_i32, collect(b_bytes))
        else
            error("Cannot resolve second argument for floordiv()")
        end
    end

    result = encode_DivIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned, rounding=RoundingZero)
    set_value_type!(tr, result, scalar_i32)
    return result
end

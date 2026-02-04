module DLFP8TypesExt

import cuTile as ct

using DLFP8Types: Float8_E4M3FN, Float8_E5M2

function ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E4M3FN})
    return ct.F8E4M3FN(table)
end

function ct.julia_to_tile_dtype!(table::ct.TypeTable, ::Type{Float8_E5M2})
    return ct.F8E5M2(table)
end

end

@testset "Types" begin

@testset "Tile" begin
    @test eltype(ct.Tile{Float32, (16,)}) == Float32
    @test eltype(ct.Tile{Float64, (32, 32)}) == Float64
    @test ct.tile_shape(ct.Tile{Float32, (16,)}) == (16,)
    @test ct.tile_shape(ct.Tile{Float32, (32, 32)}) == (32, 32)
end

end

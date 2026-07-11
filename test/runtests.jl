using Test

const TEST_MODE = get(ENV, "TEST_MODE", "default")

@testset "SCEMonteCarlo.jl" begin
    if TEST_MODE in ("default", "all", "unit")
        include("unit/fixtures.jl")
        include("unit/test_units.jl")
    end
    if TEST_MODE in ("default", "all", "aqua")
        include("aqua.jl")
    end
    if TEST_MODE in ("all", "jet")
        include("jet.jl")
    end
end

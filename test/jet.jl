using SCEMonteCarlo
using JET

@testset "JET" begin
    JET.test_package(SCEMonteCarlo; target_modules = (SCEMonteCarlo,))
end

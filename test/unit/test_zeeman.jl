# Zeeman term: constant per-atom moments in a uniform field
# (docs/specs/260822-zeeman-field/, decision record docs/specs/zeeman-field.md).

@testset "zeeman" begin
    @testset "field-free fingerprint pin (change detector)" begin
        # REGRESSION PIN — a change detector, not a correctness oracle. The Zeeman
        # spec extends `_fingerprint` (explicit magmoms/field mixing when present)
        # and must leave every field-free fingerprint byte-identical, because
        # dependent packages' checkpoint files (SCESpinDynamics) store the value.
        # Captured 2026-08-22 at c7a354a (src unchanged through 8073999) with the
        # dimer fixture via `julia --project=docs`. Recapture ONLY on an intended
        # fingerprint change — which is checkpoint-schema-version territory.
        H1 = TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        H2 = TiledHamiltonian(_dimer_model(); dims = (2, 2, 2))
        @test MC.model_fingerprint(H1) === 0x84d69fe51471f311
        @test MC.model_fingerprint(H2) === 0x107b7c1f7b2cf03e
    end
end

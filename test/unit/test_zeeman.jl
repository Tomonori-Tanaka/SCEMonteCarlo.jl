# Zeeman term: constant per-atom moments in a uniform field
# (decision record docs/specs/zeeman-field.md). The term is represented as body-1
# `ScaledTerm`s appended by the constructor, so the kernels are untouched; the gates
# below check that representation against closed forms written out by hand.

# Hand-written Zeeman energy −μ_B Σ_s m_{a(s)} (e_s · B): the oracle of every
# identity gate in this file (no package function beyond `site_atom`).
_hand_zeeman(H, mm, B, cfg) =
    -MU_B_EV_T * sum(mm[MC.site_atom(H, s)] * dot(cfg[s], B) for s = 1:n_sites(H))

# Colour class of site `s` (1-based), read off the CSR colouring.
function _color_of(H, s)
    for c = 1:H.n_colors
        s in H.color_sites[H.color_ptr[c]:(H.color_ptr[c + 1] - 1)] && return c
    end
    return 0
end

@testset "zeeman" begin
    B = SVector(0.3, -1.2, 2.0)             # tesla
    mm_dimer = [2.2, 2.2, 1.0, 0.0]         # atom 3 (SCE-inactive) carries a moment
    z3 = SVector(0, 0, 0)

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

    @testset "structure: synthetic terms, activation, colouring, lmax, show" begin
        H0 = TiledHamiltonian(_dimer_model())
        H = TiledHamiltonian(_dimer_model(); magmoms = mm_dimer, field = B)
        @test H.n_fitted_terms == 1
        @test length(H.terms) == 1 + 3                  # atoms 1, 2, 3 carry moments
        @test MC.has_field(H) && !MC.has_field(H0)
        @test H.magmoms == mm_dimer && H.field == B
        @test H0.magmoms === nothing && iszero(H0.field) && H0.n_fitted_terms == 1
        # the synthetic templates: coef 1, one atom, zero shift, l = 1, folded = hz in
        # lm_index order (rows (1,-1), (1,0), (1,1) = y, z, x)
        n1 = SCEFitting.Harmonics.N1
        for (t, a) in zip(H.terms[2:end], (1, 2, 3))
            @test t.coef == 1.0
            @test t.atoms == [a] && t.shifts == [z3] && t.ls == [1]
            @test t.folded == -(MU_B_EV_T * mm_dimer[a] / n1) .* [B[2], B[3], B[1]]
        end
        # activation: atom 3 becomes active (owns an instance); atom 4 stays inactive
        @test H.site_active == [true, true, true, false]
        @test H.n_active == 3
        @test H.site_has_l1 == [true, true, true, false]
        @test H.lmax == 1 && H.nlm == 4
        # colouring: the fitted sites keep their classes; the isolated site is class 1
        @test _color_of(H, 1) == _color_of(H0, 1) && _color_of(H, 2) == _color_of(H0, 2)
        @test _color_of(H, 3) == 1
        @test _color_of(H, 4) == 0
        # one instance per template and cell
        @test length(H.inst_term) == 4
        H8 = TiledHamiltonian(_dimer_model(); dims = (2, 2, 2), magmoms = mm_dimer,
                              field = B)
        @test length(H8.inst_term) == 4 * 8 && H8.n_active == 3 * 8
        # printing
        s = sprint(show, H)
        @test occursin("1 fitted + 3 zeeman terms", s)
        @test occursin("B = (0.3, -1.2, 2.0) T", s)
        @test occursin("(1 inactive)", s)
        @test !occursin("zeeman", sprint(show, H0)) && !occursin("B =", sprint(show, H0))

        # all-l=0 fitted list + field: lmax bumped to 1 so rows 2:4 exist
        l0 = MultipoleTerm(0.1, 2, [1, 2], [z3, z3], [0, 0], ones(1, 1))
        Hl0 = TiledHamiltonian(2, [l0]; magmoms = [1.0, 0.5], field = B)
        Hl0_0 = TiledHamiltonian(2, [l0])
        @test Hl0_0.lmax == 0 && Hl0_0.nlm == 1
        @test Hl0.lmax == 1 && Hl0.nlm == 4
        rng = MersenneTwister(3)
        cfg = _rand_config(rng, Hl0)
        E0 = total_energy(Hl0_0, cfg)
        hand = _hand_zeeman(Hl0, [1.0, 0.5], B, cfg)
        @test abs((total_energy(Hl0, cfg) - E0) - hand) <=
              10 * sqrt(n_sites(Hl0)) * eps(max(abs(E0), abs(hand)))
        # a pure paramagnet: no fitted term at all, only Zeeman templates
        Hp = TiledHamiltonian(2, MultipoleTerm[]; magmoms = [1.0, 2.0], field = B)
        @test Hp.n_fitted_terms == 0 && length(Hp.terms) == 2
        @test Hp.site_active == [true, true]
        cfg = _rand_config(rng, Hp)
        hand = _hand_zeeman(Hp, [1.0, 2.0], B, cfg)
        @test abs(total_energy(Hp, cfg) - hand) <= 10 * sqrt(n_sites(Hp)) * eps(abs(hand))
    end

    @testset "doors" begin
        m = _dimer_model()
        @test_throws ArgumentError TiledHamiltonian(m; field = B)            # no magmoms
        @test_throws ArgumentError TiledHamiltonian(m; magmoms = [1.0, 1.0])  # length
        @test_throws ArgumentError TiledHamiltonian(m; magmoms = [-1.0, 1, 1, 1])
        @test_throws ArgumentError TiledHamiltonian(m; magmoms = [NaN, 1, 1, 1])
        @test_throws ArgumentError TiledHamiltonian(m; magmoms = mm_dimer,
                                                    field = (1.0, Inf, 0.0))
        @test_throws ArgumentError TiledHamiltonian(m; magmoms = mm_dimer,
                                                    field = [1.0, 2.0])
        # no spin-dependent term at all (all moments zero, no fitted term)
        @test_throws ArgumentError TiledHamiltonian(2, MultipoleTerm[];
                                                    magmoms = [0.0, 0.0], field = B)
        # tuple and integer inputs are accepted and stored as Float64
        Ht = TiledHamiltonian(m; magmoms = [2, 2, 1, 0], field = (0, 0, 1))
        @test Ht.magmoms == [2.0, 2.0, 1.0, 0.0] && Ht.field == SVector(0.0, 0.0, 1.0)
    end

    @testset "field identity: total energy, zeeman_energy, ΔE, own-spin, bitwise" begin
        rng = MersenneTwister(11)
        cases = ((_dimer_model(), mm_dimer), (_biquadratic_model(3), [1.5, 0.7]))
        for (model, mm) in cases, dims in ((1, 1, 1), (2, 2, 2))
            H0 = TiledHamiltonian(model; dims = dims)
            H = TiledHamiltonian(model; dims = dims, magmoms = mm, field = B)
            for _ = 1:3
                cfg = _rand_config(rng, H)
                E0 = total_energy(H0, cfg)
                E = total_energy(H, cfg)
                hand = _hand_zeeman(H, mm, B, cfg)
                # the Zeeman instances are interleaved into one accumulator with the
                # fitted ones, so the difference carries O(√N·ulp(E)) — absolute bound
                tol = 10 * sqrt(n_sites(H)) * eps(max(abs(E0), abs(hand)))
                @test abs((E - E0) - hand) <= tol
                @test MC.zeeman_energy(H, cfg) ≈ hand rtol = 1e-12
                @test MC.zeeman_energy(H0, cfg) == 0.0
                zrows = MC._zrows(H, cfg)
                # program kernels ≡ reference kernels, bitwise, with the field
                @test MC._total_energy(H, zrows) == MC._total_energy_ref(H, zrows)
                ok = true
                for s = 1:n_sites(H)
                    ok &= MC.site_coeffs!(zeros(H.nlm), H, s, zrows) ==
                          MC._site_coeffs_ref!(zeros(H.nlm), H, s, zrows)
                end
                @test ok
                # exact single-spin ΔE and own-spin independence, on a coupled site
                # and (dimer) on the Zeeman-only site 3
                for s in (1, min(3, n_sites(H)))
                    c = MC.site_coeffs!(zeros(H.nlm), H, s, zrows)
                    e2 = _rand_spin(rng)
                    znew = MC._zlm_row!(zeros(H.nlm), e2, H.lmax)
                    ΔE = MC.delta_energy(c, view(zrows, :, s), znew)
                    cfg2 = copy(cfg)
                    cfg2[s] = e2
                    @test ΔE ≈ total_energy(H, cfg2) - E atol = 1e-12
                    @test MC.site_coeffs!(zeros(H.nlm), H, s, MC._zrows(H, cfg2)) == c
                end
            end
        end
    end

    @testset "gradient: closed-form field contribution, tangency, finite differences" begin
        rng = MersenneTwister(5)
        H0 = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        H = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1), magmoms = mm_dimer,
                             field = B)
        cfg = _rand_config(rng, H)
        G0 = MC.energy_gradient(H0, cfg)
        G = MC.energy_gradient(H, cfg)
        for s = 1:n_sites(H)
            e = cfg[s]
            m = mm_dimer[MC.site_atom(H, s)]
            expected = -MU_B_EV_T * m * (B - dot(e, B) * e)   # tangent projection
            @test G[s] - G0[s] ≈ expected atol = 1e-15
            @test abs(dot(G[s], e)) <= 1e-15
        end
        @test G[4] == zero(SVector{3,Float64}) && G[8] == zero(SVector{3,Float64})
        # the three gradient entry points agree bitwise with the field present
        for s in (1, 3)
            @test MC.site_gradient(H, s, cfg) == G[s]
        end
        @test MC.energy_gradient(H, cfg; ntasks = 3) == G
        # central finite differences of the total energy along a tangent direction,
        # on a coupled site and on the Zeeman-only site
        for s in (1, 3)
            e = cfg[s]
            t = normalize(cross(e, _rand_spin(rng)))
            h = 1e-5
            cp = copy(cfg)
            cm = copy(cfg)
            cp[s] = normalize(e + h * t)
            cm[s] = normalize(e - h * t)
            fd = (total_energy(H, cp) - total_energy(H, cm)) / (2h)
            @test fd ≈ dot(G[s], t) rtol = 1e-6
        end
    end

    @testset "magmoms alone (or B = 0): bitwise the field-free Hamiltonian" begin
        rng = MersenneTwister(8)
        H0 = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        for Hm in (TiledHamiltonian(_dimer_model(); dims = (2, 1, 1),
                                    magmoms = mm_dimer),
                   TiledHamiltonian(_dimer_model(); dims = (2, 1, 1),
                                    magmoms = mm_dimer, field = (0.0, 0.0, 0.0)))
            @test !MC.has_field(Hm)
            @test Hm.magmoms == mm_dimer && iszero(Hm.field)
            @test length(Hm.terms) == 1 && Hm.n_fitted_terms == 1
            @test Hm.site_active == H0.site_active && Hm.n_active == H0.n_active
            @test Hm.site_has_l1 == H0.site_has_l1
            @test Hm.color_sites == H0.color_sites && Hm.color_ptr == H0.color_ptr
            @test Hm.lmax == H0.lmax
            @test sprint(show, Hm) == sprint(show, H0)
            for _ = 1:3
                cfg = _rand_config(rng, Hm)
                @test total_energy(Hm, cfg) == total_energy(H0, cfg)
                @test MC.zeeman_energy(Hm, cfg) == 0.0
                zrows = MC._zrows(Hm, cfg)
                ok = true
                for s = 1:n_sites(Hm)
                    ok &= MC.site_coeffs!(zeros(Hm.nlm), Hm, s, zrows) ==
                          MC.site_coeffs!(zeros(H0.nlm), H0, s, zrows)
                end
                @test ok
                @test MC.energy_gradient(Hm, cfg) == MC.energy_gradient(H0, cfg)
            end
        end
    end

    @testset "zero-moment sublattice stays frozen and excluded with a field" begin
        H = TiledHamiltonian(_dimer_model(); magmoms = [2.0, 2.0, 0.0, 0.0], field = B)
        @test H.site_active == [true, true, false, false] && H.n_active == 2
        @test length(H.terms) == 3
        rng = Xoshiro(2)
        st = MC.ChainState(H, _rand_config(rng, H), Xoshiro(11), 0.6)
        sc = MC.SweepScratch(H)
        frozen = (st.config[3], st.config[4])
        for _ = 1:20
            MC.metropolis_sweep!(st, H, 1 / 0.02, sc)
            MC.overrelaxation_sweep!(st, H, 1 / 0.02, sc)
        end
        @test st.config[3] === frozen[1] && st.config[4] === frozen[2]
        @test st.att_metro == 20 * H.n_active
        G = MC.energy_gradient(H, st.config)
        @test G[3] == zero(SVector{3,Float64}) && G[4] == zero(SVector{3,Float64})
    end

    @testset "ReducedCell constructor form" begin
        model, cr = _stacked_chain_model()
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test n_atoms(red) == 1
        H0 = TiledHamiltonian(red; dims = (1, 1, 4))
        H = TiledHamiltonian(red; dims = (1, 1, 4), magmoms = [2.1], field = B)
        @test H.n_fitted_terms == length(H0.terms)
        @test length(H.terms) == H.n_fitted_terms + 1
        cfg = _rand_config(MersenneTwister(9), H)
        hand = _hand_zeeman(H, [2.1], B, cfg)
        tol = 10 * sqrt(n_sites(H)) * eps(max(abs(total_energy(H0, cfg)), abs(hand)))
        @test abs((total_energy(H, cfg) - total_energy(H0, cfg)) - hand) <= tol
        @test_throws ArgumentError TiledHamiltonian(red; magmoms = [1.0, 1.0], field = B)
    end
end

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

# The schema-v3 `_fingerprint` (src/checkpoint.jl), copied verbatim and self-contained
# (its own mixer and finalizer): the machine-portable change detector of the pin below.
# It pins the algorithm — loop order, mixer, finalizer — not a platform's bits. If the
# fingerprint is ever changed on purpose, update this copy together with the
# checkpoint schema version (checkpoint-schema.md).
_fpm_frozen(h::UInt64, x::UInt64) = (g = (h ⊻ x) * 0x00000100000001b3; g ⊻ (g >> 32))
_fpm_frozen(h::UInt64, x::Integer) = _fpm_frozen(h, reinterpret(UInt64, Int64(x)))
_fpm_frozen(h::UInt64, x::Float64) = _fpm_frozen(h, reinterpret(UInt64, x))
function _fpf_frozen(h::UInt64)
    h ⊻= h >> 33
    h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    return h ⊻ (h >> 33)
end
function _fingerprint_v3_frozen(H::MC.TiledHamiltonian)::UInt64
    h = 0xcbf29ce484222325
    h = _fpm_frozen(h, H.n_cell_atoms)
    for d in H.dims
        h = _fpm_frozen(h, d)
    end
    for t in H.terms
        h = _fpm_frozen(h, t.coef)
        for a in t.atoms
            h = _fpm_frozen(h, a)
        end
        for s in t.shifts
            h = _fpm_frozen(h, s[1])
            h = _fpm_frozen(h, s[2])
            h = _fpm_frozen(h, s[3])
        end
        for l in t.ls
            h = _fpm_frozen(h, l)
        end
        for v in t.folded
            h = _fpm_frozen(h, v)
        end
    end
    if H.magmoms !== nothing
        h = _fpm_frozen(h, 1)
        for m in H.magmoms
            h = _fpm_frozen(h, m)
        end
        for b in H.field
            h = _fpm_frozen(h, b)
        end
    end
    return _fpf_frozen(h)
end

@testset "zeeman" begin
    B = SVector(0.3, -1.2, 2.0)             # tesla
    mm_dimer = [2.2, 2.2, 1.0, 0.0]         # atom 3 (SCE-inactive) carries a moment
    z3 = SVector(0, 0, 0)

    @testset "fingerprint pin (change detector, schema v3)" begin
        # REGRESSION PIN — a change detector, not a correctness oracle. Two gates,
        # both machine-portable:
        #   (1) `_fingerprint_v3_frozen` above reproduces the current value on
        #       field-free and field-carrying models — SALC-basis ones included,
        #       whose `folded` tensors come out of LAPACK;
        #   (2) numeric values on the hand-built chain fixture, whose every mixed
        #       word is exactly representable (π/30 coef, 0/1 tensors, pure-Julia
        #       `^`). Captured 2026-08-22 at the schema-v3 commit (macOS, Julia
        #       1.12.7).
        # A numeric pin on a SALC-basis model is NOT portable — the first version
        # pinned `_dimer_model()` and passed on macOS but failed on ubuntu CI: null-
        # space/SVD tensors differ in their last bits across LAPACK builds, so such
        # fingerprints are machine-specific by construction (checkpoint-schema.md).
        # Recapture ONLY on an intended fingerprint change — which is a checkpoint
        # schema bump (v2 → v3 was the `_fp_mix` fold, 2026-08-22).
        for H in (TiledHamiltonian(_dimer_model(); dims = (1, 1, 1)),
                  TiledHamiltonian(_dimer_model(); dims = (2, 2, 2)),
                  TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1)),
                  TiledHamiltonian(_dimer_model(); magmoms = mm_dimer),
                  TiledHamiltonian(_dimer_model(); magmoms = mm_dimer, field = B))
            @test MC.model_fingerprint(H) === _fingerprint_v3_frozen(H)
        end
        Hc1 = MC.TiledHamiltonian(1, _chain_terms(0.05); dims = (2, 1, 1))
        Hc2 = MC.TiledHamiltonian(1, _chain_terms(0.05); dims = (4, 2, 1))
        @test MC.model_fingerprint(Hc1) === 0xb931b0bd3835a7f1
        @test MC.model_fingerprint(Hc2) === 0xc717ef65f154e359
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

    # --- M2: observables, dynamics-level consequences, checkpoint identity --------

    @testset "observables: :M and :M_B against hand sums" begin
        H0 = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        H = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1), magmoms = mm_dimer,
                             field = B)
        obs = standard_observables(H)
        names = [o.name for o in obs]
        @test names[1:7] == [o.name for o in standard_observables(H0)]
        @test names[8:9] == [:M, :M_B]
        cfg = _rand_config(MersenneTwister(4), H)
        # hand sum over the active sites (atoms 1–3 in both cells) per training cell
        hand = sum(mm_dimer[MC.site_atom(H, s)] * cfg[s]
                   for s = 1:n_sites(H) if H.site_active[s]) / 2
        @test obs[8].f(cfg, 0.0, H) ≈ hand rtol = 1e-14
        @test obs[9].f(cfg, 0.0, H) ≈ dot(hand, B / norm(B)) rtol = 1e-13
        # magmoms alone: :M present, :M_B absent; the frozen moment-carrying atom 3
        # is excluded (B3 — a frozen direction is not a moment)
        Hm = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1), magmoms = mm_dimer)
        obsm = standard_observables(Hm)
        @test [o.name for o in obsm][8:end] == [:M]
        @test !Hm.site_active[3] && !Hm.site_active[7]
        handm = sum(mm_dimer[MC.site_atom(Hm, s)] * cfg[s]
                    for s = 1:n_sites(Hm) if Hm.site_active[s]) / 2
        @test obsm[8].f(cfg, 0.0, Hm) ≈ handm rtol = 1e-14
        @test handm != hand
    end

    @testset "magmoms alone: a seeded run_mc is bitwise the field-free run" begin
        H0 = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        Hm = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1), magmoms = mm_dimer)
        kw = (; kT = [0.05, 0.02], sweeps_therm = 100, sweeps_measure = 200, nbins = 8,
              seed = 21, or_per_metropolis = 1)
        a = run_mc(H0; kw...)
        b = run_mc(Hm; kw...)
        @test a.final_config == b.final_config
        for (pa, pb) in zip(a.points, b.points)
            for k in keys(pa.stats)            # :m, :sublattice_m, C, χ, U, … bitwise
                @test pa.stats[k].mean == pb.stats[k].mean
                @test pa.stats[k].err == pb.stats[k].err
            end
            @test pa.acceptance_metropolis == pb.acceptance_metropolis
            @test pa.acceptance_or == pb.acceptance_or
            @test haskey(pb.stats, :M) && !haskey(pa.stats, :M)
            @test !haskey(pb.stats, :M_B)
        end
    end

    @testset "Langevin law for free moments in a field" begin
        # atoms 3–4 of the dimer are SCE-free; atom 3 gets m = 2 μ_B, atom 4 none, so
        # atom 3 is an isolated classical moment in the field: ⟨e·B̂⟩ = L(β μ_B m B).
        m3 = 2.0
        Bz = 20.0
        h = MU_B_EV_T * m3 * Bz                # its Zeeman energy scale (eV)
        H = TiledHamiltonian(_dimer_model(); magmoms = [0.0, 0.0, m3, 0.0],
                             field = (0.0, 0.0, Bz))
        @test length(H.terms) == 2 && H.site_active == [true, true, true, false]
        e3z = Observable(:e3z, 1, (cfg, E, H) -> cfg[3][3])
        # Tolerance: measured 2026-08-22 with seeds 301/302 — binning error of
        # ⟨e3z⟩ ≈ 0.008 / 0.009 (τ_int 1.8 / 5.3), deviations −0.0037 / +0.0031;
        # atol = 0.04 is ≈ 4.5σ. The mutation the gate must resolve is a sign flip
        # of the Zeeman term (⟨e3z⟩ → −L, an excursion of 2L ≈ 0.63 / 1.23 ≫ atol).
        for (i, βh) in enumerate([1.0, 2.5])
            r = run_mc(H; kT = h / βh, sweeps_therm = 500, sweeps_measure = 40_000,
                       measure_interval = 2, seed = 300 + i,
                       observables = [Observable(:energy, 1, (c, E, H) -> E),
                                      Observable(:energy2, 1, (c, E, H) -> E^2), e3z],
                       evaluables = Evaluable[])
            st = r.points[1].stats[:e3z]
            @test st.mean[1] ≈ _langevin(βh) atol = 0.04
            @test st.err[1] < 0.02
        end
    end

    @testset "overrelaxation with a field: pure l=1 stays exact" begin
        # the dimer is l = 1 only and the Zeeman templates are l = 1, so the
        # reflection about the total local field is still microcanonical
        H = TiledHamiltonian(_dimer_model(); magmoms = mm_dimer, field = B)
        rng = Xoshiro(9)
        st = MC.ChainState(H, MC._initial_config(H, nothing, rng), rng, 0.6)
        sc = MC.SweepScratch(H)
        E0 = st.energy
        e3_init = st.config[3]
        # an odd sweep count: the Zeeman-only site reflects about the FIXED axis B̂,
        # and two such reflections are the identity (bitwise), so after an even
        # number of sweeps it is back where it started
        for _ = 1:21
            MC.overrelaxation_sweep!(st, H, 1 / 0.01, sc)
        end
        @test st.att_or == 21 * 3              # sites 1, 2 (coupled) and 3 (Zeeman-only)
        @test st.acc_or == st.att_or
        @test st.energy ≈ E0 atol = 1e-12
        @test total_energy(H, st.config) ≈ E0 atol = 1e-12
        # the Zeeman-only site moved (one net reflection) but keeps e·B̂ exactly
        @test norm(st.config[3] - e3_init) > 1e-6
        @test dot(st.config[3], B) ≈ dot(e3_init, B) atol = 1e-12
        # the or_per_metropolis door: a model without an l = 1 channel gains one
        # through the field
        l2 = MultipoleTerm(0.05, 2, [1, 2], [z3, z3], [2, 2], ones(5, 5))
        Hq = TiledHamiltonian(2, [l2])
        @test !any(Hq.site_has_l1)
        @test_throws ArgumentError MC._resolve_or_passes(Hq, 1)
        HqB = TiledHamiltonian(2, [l2]; magmoms = [1.0, 1.0], field = B)
        @test all(HqB.site_has_l1)
        @test MC._resolve_or_passes(HqB, 1) == 1
    end

    @testset "ground state in a field with the default gtol / ladder" begin
        Bhat = B / norm(B)
        # ferro pair + Zeeman-only atom 3: every active spin ends along B̂
        H = TiledHamiltonian(_dimer_model(); magmoms = mm_dimer, field = B)
        fgs = find_ground_state(H; nstarts = 2, seed = 3)
        for s = 1:3
            @test dot(fgs.config[s], Bhat) ≈ 1 atol = 1e-6
        end
        # pure paramagnet: scale and ladder come from the Zeeman templates alone
        Hp = TiledHamiltonian(2, MultipoleTerm[]; magmoms = [1.0, 2.0], field = B)
        fgp = find_ground_state(Hp; nstarts = 2, seed = 4)
        for s = 1:2
            @test dot(fgp.config[s], Bhat) ≈ 1 atol = 1e-6
        end
        @test fgp.energy ≈ -MU_B_EV_T * 3.0 * norm(B) rtol = 1e-8
    end

    @testset "serial ≡ parallel sweeps with a field (bitwise)" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1),
                             magmoms = [1.5, 0.7], field = B)
        β = 1 / 0.05
        function run_chain_field(ntasks)
            st = MC.ChainState(H, MC._initial_config(H, nothing, Xoshiro(3)),
                               Xoshiro(5), 0.6)
            scs = [MC.SweepScratch(H) for _ = 1:ntasks]
            for _ = 1:25
                MC.metropolis_sweep!(st, H, β, scs)
                MC.overrelaxation_sweep!(st, H, β, scs)
            end
            return st
        end
        ref = run_chain_field(1)
        for nt in (2, 3)
            st = run_chain_field(nt)
            @test st.config == ref.config
            @test st.energy === ref.energy
            @test (st.acc_metro, st.att_metro, st.acc_or, st.att_or) ==
                  (ref.acc_metro, ref.att_metro, ref.acc_or, ref.att_or)
        end
    end

    @testset "fingerprint: moments and field are part of the identity" begin
        fp(; kw...) = MC.model_fingerprint(TiledHamiltonian(_dimer_model(); kw...))
        f0 = fp()
        @test fp(; magmoms = mm_dimer) !== f0                  # moments alone count
        @test fp(; magmoms = mm_dimer) === fp(; magmoms = mm_dimer)
        @test fp(; magmoms = mm_dimer, field = B) !== fp(; magmoms = mm_dimer)
        @test fp(; magmoms = mm_dimer, field = B) === fp(; magmoms = mm_dimer, field = B)
        # Sign flips of the field must change the identity. The pre-v3 mixer carried
        # a Float64 sign bit only into bit 63 of the hash, so B → −B (3·n_moment_atoms
        # folded words + 3 field words) or a single-component flip cancelled whenever
        # the number of moment-carrying atoms was odd; the v3 fold removes that
        # linearity (mixer gate in test_checkpoint.jl). Odd (3, 1) and even (2)
        # moment-atom counts, full and single-component flips:
        for mm in (mm_dimer, [2.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0])
            @test fp(; magmoms = mm, field = B) !== fp(; magmoms = mm, field = -B)
            @test fp(; magmoms = mm, field = B) !==
                  fp(; magmoms = mm, field = (-B[1], B[2], B[3]))
            @test fp(; magmoms = mm, field = B) !==
                  fp(; magmoms = mm, field = (B[1], -B[2], B[3]))
        end
        # The same linearity let an even number of sign flips INSIDE a fitted
        # `folded` tensor collide under the pre-v3 mixer (recorded as `@test_broken`
        # until the v3 fold, 2026-08-22):
        f1 = zeros(3, 3)
        f1[1, 1] = 0.4
        f1[2, 2] = -0.3
        f1[3, 3] = 0.2
        f2 = copy(f1)
        f2[1, 1] = -0.4
        f2[2, 2] = 0.3
        fpt(f) = MC.model_fingerprint(MC.TiledHamiltonian(2,
            [MultipoleTerm(0.1, 2, [1, 2], [z3, z3], [1, 1], f)]))
        @test fpt(f1) !== fpt(f2)
        # same Zeeman templates (m·B unchanged), different moments ⇒ different :M
        @test fp(; magmoms = mm_dimer, field = B) !==
              fp(; magmoms = mm_dimer ./ 2, field = 2 .* B)
    end

    @testset "checkpoint: resume is bit-identical with a field; mismatch errors" begin
        dir = mktempdir()
        Hc = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1),
                              magmoms = [1.5, 0.7], field = B)
        # MC, interrupted mid-measure (the poison pattern of test_checkpoint.jl)
        cnt = Ref(0)
        poison = Observable(:poison, 1, (cfg, E, Hh) ->
                            (cnt[] += 1) >= 300 ? error("poison interrupt") : 0.0)
        benign = Observable(:poison, 1, (cfg, E, Hh) -> 0.0)
        obs = [standard_observables(Hc); benign]
        kw = (; kT = [0.5, 0.3], sweeps_therm = 200, sweeps_measure = 400,
              measure_interval = 2, nbins = 8, renorm_interval = 100, seed = 42,
              observables = obs)
        path = joinpath(dir, "mc_field.jld2")
        a = run_mc(Hc; kw...)
        err = try
            run_mc(Hc; kw..., observables = [standard_observables(Hc); poison],
                   checkpoint = path, checkpoint_interval = 150)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("poison", err.msg)
        MC.jldopen(path, "r") do f
            @test f["progress/temp_index"] == 2 && f["progress/phase"] == "measure"
            @test 0 < f["progress/sweep"] < 400                # genuinely mid-run
            @test f["zeeman/magmoms"] == [1.5, 0.7]            # informational group
            @test f["zeeman/field"] == Vector(B)
        end
        c = resume(path, Hc; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
        @test haskey(a.points[1].stats, :M_B)
        # a different field, rescaled (m, B), or no field ⇒ fingerprint mismatch
        for Hbad in (TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1),
                                      magmoms = [1.5, 0.7], field = 2 .* B),
                     TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1),
                                      magmoms = [0.75, 0.35], field = 2 .* B),
                     TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1)))
            @test_throws ErrorException resume(path, Hbad; observables = obs)
        end
        # PT (no end-of-run write: the file lands mid-run by construction)
        kwp = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
               exchange_interval = 7, nbins = 8, seed = 11)
        pp = joinpath(dir, "pt_field.jld2")
        pa = run_pt(Hc; kwp...)
        pb = run_pt(Hc; kwp..., checkpoint = pp, checkpoint_interval = 120)
        _assert_same_result(pa, pb)
        @test pa.final_configs == pb.final_configs
        MC.jldopen(pp, "r") do f
            @test f["progress/phase"] in ("therm", "measure")
            @test f["progress/done"] > 0
            @test f["zeeman/field"] == Vector(B)
        end
        pc = resume(pp, Hc)
        _assert_same_result(pa, pc)
        @test pa.final_configs == pc.final_configs
        @test pa.swap_acceptance == pc.swap_acceptance
    end
end

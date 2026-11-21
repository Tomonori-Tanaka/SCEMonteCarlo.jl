# Constrained MC: the torque observables (docs/specs/constrained-mc.md D5).
# G3  — `total_torque · w` against the central difference of `total_energy` under a
#       rigid global rotation about w (no gradient code on the oracle side);
#       isotropic models give exactly-zero total torque; the Zeeman torque against
#       a hand-written sum.
# G17 — partial (term-subset) torques plus the analytic Zeeman torque sum to the
#       total; each partial torque against its own rotated-energy difference; the
#       doors of `torque_observable`.

# −dE/dα at α = 0 for the rigid rotation of every spin about `w`, by central
# difference of an energy function `Ef(config)`. By the torque convention
# T·w = −dE/dα this is the oracle of T·w.
function _tq_rotation_fd(Ef, config, w; h = 1e-5)
    Ep = Ef([_cmcref_rot(e, w, h) for e in config])
    Em = Ef([_cmcref_rot(e, w, -h) for e in config])
    return -(Ep - Em) / (2h)
end

# The spherical frame at (θ, φ): n̂ and its two tangent unit vectors ê_θ, ê_φ. The
# gates use the generic direction θ = 0.6, φ = 0.35, where every Cartesian
# component of the frame is nonzero (at θ = 0 a lab-frame/body-frame mix-up is
# invisible).
_tq_spherical_frame(θ, φ) = (_cmcref_eθ(θ, φ), _cmcref_eφ(θ, φ), _cmcref_dir(θ, φ))

# Hand-written Zeeman torque of the full system. Derivation (independent of the
# package's helper): E_Z = −μ_B Σ_s m_s (e_s·B). Rotating every spin by α about w,
# de_s/dα = w × e_s, so dE_Z/dα = −μ_B Σ m_s (w × e_s)·B = −w · μ_B Σ m_s (e_s × B)
# (scalar triple product). With T·w = −dE/dα: T_Z = +μ_B Σ_s m_s e_s × B — a moment
# along +B feels no torque, one tilted off B is turned toward it.
_tq_hand_zeeman_torque(H, mm, B, config) =
    MU_B_EV_T * sum(mm[MC.site_atom(H, s)] * cross(config[s], B)
                    for s = 1:n_sites(H))

_tq_hand_zeeman_energy(H, mm, B, config) =
    -MU_B_EV_T * sum(mm[MC.site_atom(H, s)] * dot(config[s], B) for s = 1:n_sites(H))

@testset "CMC torque" begin
    rng = MersenneTwister(611)
    frame = _tq_spherical_frame(0.6, 0.35)

    @testset "G3: T·w ≡ −dE/dα under a global rotation" begin
        cases = [("F2g", MC.TiledHamiltonian(2, _f2g_terms())),
                 ("F3a ×2", MC.TiledHamiltonian(3, _f3a_terms(); dims = (2, 1, 1))),
                 ("F4", MC.TiledHamiltonian(4, _f4_terms())),
                 ("3-body chain (shifts)",
                  MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (5, 1, 1))),
                 ("FZ", MC.TiledHamiltonian(2, _f2g_terms(); magmoms = _fz_magmoms(),
                                            field = _fz_field()))]
        for (name, H) in cases
            worst = 0.0
            smallest = Inf
            for _ = 1:3
                config = _rand_config(rng, H)
                T = MC.total_torque(H, config)
                for w in frame
                    fd = _tq_rotation_fd(c -> total_energy(H, c), config, w)
                    @test isapprox(dot(T, w), fd; rtol = 1e-7, atol = 1e-10)
                    worst = max(worst, abs(dot(T, w) - fd))
                    smallest = min(smallest, abs(fd))
                end
            end
            # non-vacuity: every component probed is a genuinely nonzero torque
            @test smallest > 1e-4
        end
    end

    @testset "G3: the spherical frame is ∂n̂/∂θ and (1/sinθ)∂n̂/∂φ" begin
        h = 1e-6
        for (θ, φ) in ((0.6, 0.35), (2.1, -1.3))
            eθ, eφ, n = _tq_spherical_frame(θ, φ)
            @test norm(eθ) ≈ 1 && norm(eφ) ≈ 1 && abs(dot(eθ, eφ)) < 1e-15
            @test isapprox(eθ, (_cmcref_dir(θ + h, φ) - _cmcref_dir(θ - h, φ)) / (2h);
                           atol = 1e-9)
            @test isapprox(eφ, (_cmcref_dir(θ, φ + h) - _cmcref_dir(θ, φ - h)) /
                               (2h * sin(θ)); atol = 1e-9)
        end
    end

    @testset "G3: closed-form signs of single-ion torques" begin
        x̂, ŷ, ẑ = SVector(1.0, 0, 0), SVector(0.0, 1, 0), SVector(0.0, 0, 1)
        uniform(H, e) = MC.SpinConfig(fill(e, n_sites(H)))
        # (a) uniaxial E = −K Σ_s (e_s·ẑ)²: a body-1 l = 2, m = 0 term (Z₂₀ ∝ 3z² − 1).
        # K is read off an energy difference, never from the coefficient literal.
        H = MC.TiledHamiltonian(2, [MultipoleTerm(-0.01, 1, [a], [_CMC_Z], [2],
                                                  [0.0, 0.0, 1.0, 0.0, 0.0])
                                    for a = 1:2]; dims = (2, 1, 1))
        N = n_sites(H)
        E_z = total_energy(H, uniform(H, ẑ))
        K = (total_energy(H, uniform(H, x̂)) - E_z) / N
        @test K > 1e-3
        for _ = 1:5                     # the fixture really is −K Σ (e·ẑ)² + const
            c = _rand_config(rng, H)
            @test total_energy(H, c) - E_z ≈ K * sum(1 - e[3]^2 for e in c) atol = 1e-14
        end
        # e = (sinθ, 0, cosθ) on every site: T·ŷ = −N K sin 2θ
        for θ in (0.3, 0.9, 2.2)
            T = MC.total_torque(H, uniform(H, SVector(sin(θ), 0.0, cos(θ))))
            @test T[2] ≈ -N * K * sin(2θ) rtol = 1e-12
            @test abs(T[1]) < 1e-14 && abs(T[3]) < 1e-14
        end
        # (b) E = −K Σ_s (e_s·x̂)²: x² − 1/3 = (x² − y²)/2 − (3z² − 1)/6, built from
        # the tesseral normalizations N₂₂ = Z₂₂(x̂), N₂₀ = Z₂₀(ẑ)/2
        zx = MC._zlm_row!(zeros(9), x̂, 2)
        zz = MC._zlm_row!(zeros(9), ẑ, 2)
        n22, n20 = zx[9], zz[7] / 2
        folded = [0.0, 0.0, -1 / (6 * n20), 0.0, 1 / (2 * n22)]
        H = MC.TiledHamiltonian(2, [MultipoleTerm(-0.01, 1, [a], [_CMC_Z], [2],
                                                  copy(folded)) for a = 1:2];
                                dims = (2, 1, 1))
        E_x = total_energy(H, uniform(H, x̂))
        K = (total_energy(H, uniform(H, ŷ)) - E_x) / N
        @test K > 1e-3
        for _ = 1:5                     # the fixture really is −K Σ (e·x̂)² + const
            c = _rand_config(rng, H)
            @test total_energy(H, c) - E_x ≈ K * sum(1 - e[1]^2 for e in c) atol = 1e-14
        end
        # θ = π/2, e = (cosφ, sinφ, 0) on every site: T·ê_θ = N K sin 2φ
        for φ in (0.35, 1.1, 2.6)
            eθ, _, n = _tq_spherical_frame(π / 2, φ)
            T = MC.total_torque(H, uniform(H, n))
            @test dot(T, eθ) ≈ N * K * sin(2φ) rtol = 1e-12
        end
    end

    @testset "G3: isotropic model — ΣT = 0 per configuration" begin
        # isotropic l = 1 ring (`_chain_terms`, 3 sites) + isotropic l = 2 pair +
        # a body-1 l = 0 term: every coupling is rotation-invariant
        z = _CMC_Z
        x = SVector(1, 0, 0)
        terms = vcat(_chain_terms(0.05),
                     [MultipoleTerm(0.03, 2, [1, 1], [z, x], [2, 2], _iso_folded(2)),
                      MultipoleTerm(0.2, 1, [1], [z], [0], [1.0])])
        H = MC.TiledHamiltonian(1, terms; dims = (3, 1, 1))
        for _ = 1:5
            config = _rand_config(rng, H)
            T = MC.total_torque(H, config)
            @test norm(T) < 1e-12
            # non-vacuity: the individual site torques are far from zero
            G = MC.energy_gradient(H, config)
            @test maximum(norm(cross(G[s], config[s])) for s = 1:n_sites(H)) > 1e-3
        end
    end

    @testset "G3: Zeeman torque ≡ μ_B Σ m e × B" begin
        mm, B = _fz_magmoms(), _fz_field()
        for dims in ((1, 1, 1), (2, 1, 1))
            H0 = MC.TiledHamiltonian(2, _f2g_terms(); dims = dims)
            Hz = MC.TiledHamiltonian(2, _f2g_terms(); dims = dims, magmoms = mm,
                                     field = B)
            for _ = 1:3
                config = _rand_config(rng, Hz)
                Tz = _tq_hand_zeeman_torque(Hz, mm, B, config)
                @test norm(Tz) > 1e-4                       # non-vacuity
                # the sign derivation above, checked by finite differences of the
                # hand-written Zeeman energy (no package code on either side)
                for w in frame
                    fd = _tq_rotation_fd(c -> _tq_hand_zeeman_energy(Hz, mm, B, c),
                                      config, w)
                    @test isapprox(dot(Tz, w), fd; rtol = 1e-7, atol = 1e-12)
                end
                # the package: the field's share of the total torque, and the
                # analytic helper the CMC observables use
                T_field = MC.total_torque(Hz, config) - MC.total_torque(H0, config)
                @test T_field ≈ Tz atol = 1e-12
                @test MC._zeeman_torque(Hz, config) ≈ Tz atol = 1e-15 rtol = 1e-13
            end
            @test MC._zeeman_torque(H0, _rand_config(rng, H0)) == zero(SVector{3,Float64})
        end
    end

    @testset "G17: Σ partial torques + T_Zeeman ≡ total torque" begin
        for (name, n, terms, kw) in
            (("F3a", 3, _f3a_terms(), (;)),
             ("FZ", 2, _f2g_terms(), (; magmoms = _fz_magmoms(), field = _fz_field())))
            dims = (2, 1, 1)
            ncells = 2
            H = MC.TiledHamiltonian(n, terms; dims = dims, kw...)
            bodies = sort(unique(t.body for t in terms))
            parts = [MC.TiledHamiltonian(n, filter(t -> t.body == b, terms);
                                         dims = dims) for b in bodies]
            obs = [MC.torque_observable(Symbol(:torque_body, b), Hp, H)
                   for (b, Hp) in zip(bodies, parts)]
            @test all(o -> o.ncomp == 3, obs)
            for _ = 1:3
                config = _rand_config(rng, H)
                E = total_energy(H, config)
                Tparts = [SVector{3,Float64}(o.f(config, E, H)) for o in obs]
                Tz = MC._zeeman_torque(H, config)
                @test sum(Tparts) * ncells + Tz ≈ MC.total_torque(H, config) atol = 1e-12
                # each part is itself a rotated-energy derivative of ITS energy,
                # per training cell
                for (Hp, Tp) in zip(parts, Tparts)
                    for w in frame
                        fd = _tq_rotation_fd(c -> total_energy(Hp, c), config, w)
                        @test isapprox(dot(Tp, w) * ncells, fd; rtol = 1e-7,
                                       atol = 1e-10)
                    end
                end
                if name == "FZ"
                    @test norm(Tz) > 1e-4           # the Zeeman share is visible
                end
            end
        end
    end

    @testset "G17: a site-relabelled partial H is distinguishable" begin
        # Non-vacuity of the door below: on F3a, relabelling the atoms of the
        # partial term list (a → a+1 mod 3) changes every per-atom torque. If the
        # door let such an Hpart through, the partial torques would be wrong by
        # this much — far beyond the 1e-12 of the sum identity.
        terms = _f3a_terms()
        perm(t) = MultipoleTerm(t.coef, t.body, [mod1(a + 1, 3) for a in t.atoms],
                                t.shifts, t.ls, t.folded)
        H = MC.TiledHamiltonian(3, terms; dims = (2, 1, 1))
        for b in (1, 2)
            sub = filter(t -> t.body == b, terms)
            Hp = MC.TiledHamiltonian(3, sub; dims = (2, 1, 1))
            Hq = MC.TiledHamiltonian(3, perm.(sub); dims = (2, 1, 1))
            config = _rand_config(rng, H)
            @test norm(MC.total_torque(Hp, config) - MC.total_torque(Hq, config)) > 1e-4
            @test_throws ArgumentError MC.torque_observable(:bad, Hq, H)
        end
    end

    @testset "G17: torque_observable doors" begin
        terms = _f3a_terms()
        onsite = filter(t -> t.body == 1, terms)
        H = MC.TiledHamiltonian(3, terms; dims = (2, 1, 1))
        # accepted: any subset of the same list, the full list included
        @test MC.torque_observable(:t1, MC.TiledHamiltonian(3, onsite; dims = (2, 1, 1)),
                                   H) isa Observable
        @test MC.torque_observable(:all, MC.TiledHamiltonian(3, terms; dims = (2, 1, 1)),
                                   H) isa Observable
        # a different term list: a perturbed coefficient, a foreign single-ion tensor
        bumped = [MultipoleTerm(t.coef * (1 + 1e-12), t.body, t.atoms, t.shifts, t.ls,
                                t.folded) for t in onsite]
        @test_throws ArgumentError MC.torque_observable(
            :bad, MC.TiledHamiltonian(3, bumped; dims = (2, 1, 1)), H)
        @test_throws ArgumentError MC.torque_observable(
            :bad, MC.TiledHamiltonian(3, filter(t -> t.body == 1, _f3_terms());
                                      dims = (2, 1, 1)), H)
        # a field on Hpart — rejected by the field door itself (its appended Zeeman
        # templates would also fail the template match, so check the message)
        err = _cmcref_thrown(() -> MC.torque_observable(
            :bad, MC.TiledHamiltonian(3, onsite; dims = (2, 1, 1),
                                      magmoms = [1.0, 1.0, 1.0],
                                      field = (0.0, 0.0, 1.0)), H))
        @test err isa ArgumentError && occursin("carries an external field", err.msg)
        # a template that differs ONLY in its shifts (3-body chain, nonzero shifts)
        x = SVector(1, 0, 0)
        chain = _threebody_terms(0.05)
        moved = [MultipoleTerm(t.coef, t.body, t.atoms, [_CMC_Z, x, 3 * x], t.ls,
                               t.folded) for t in chain]
        Hc = MC.TiledHamiltonian(1, chain; dims = (5, 1, 1))
        err = _cmcref_thrown(() -> MC.torque_observable(
            :bad, MC.TiledHamiltonian(1, moved; dims = (5, 1, 1)), Hc))
        @test err isa ArgumentError && occursin("matches no fitted template", err.msg)
        @test MC.torque_observable(:ok, MC.TiledHamiltonian(1, chain; dims = (5, 1, 1)),
                                   Hc) isa Observable
        # bound to the H object it was built with: measuring in a run on another
        # (even an equal) Hamiltonian is refused
        o = MC.torque_observable(:t1, MC.TiledHamiltonian(3, onsite; dims = (2, 1, 1)),
                                 H)
        Hsame = MC.TiledHamiltonian(3, terms; dims = (2, 1, 1))
        config = _rand_config(rng, H)
        @test length(o.f(config, 0.0, H)) == 3
        @test_throws ArgumentError o.f(config, 0.0, Hsame)
        # different dims; different cell (same terms read with a 4-atom cell)
        @test_throws ArgumentError MC.torque_observable(
            :bad, MC.TiledHamiltonian(3, onsite; dims = (1, 2, 1)), H)
        @test_throws ArgumentError MC.torque_observable(
            :bad, MC.TiledHamiltonian(3, onsite; dims = (1, 1, 1)), H)
        @test_throws ArgumentError MC.torque_observable(
            :bad, MC.TiledHamiltonian(4, onsite; dims = (2, 1, 1)), H)
        # the Zeeman templates of a field-carrying H are not "fitted" templates
        Hz = MC.TiledHamiltonian(2, _f2g_terms(); dims = (2, 1, 1),
                                 magmoms = _fz_magmoms(), field = _fz_field())
        @test Hz.n_fitted_terms < length(Hz.terms)
        @test MC.torque_observable(
            :ok, MC.TiledHamiltonian(2, _f2g_terms(); dims = (2, 1, 1)), Hz) isa
              Observable
    end
end

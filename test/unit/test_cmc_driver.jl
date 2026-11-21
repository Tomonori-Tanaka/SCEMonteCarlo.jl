# Constrained MC: the driver `run_cmc`, its result, the observables, and the
# end-of-thermalization warnings (docs/specs/constrained-mc.md D3–D5).
#
# OBS — the default observables on a dims (2,1,1) supercell (n_cells = 2 ≠ n_sites),
#       against sums written here from their definitions; the field-only Zeeman
#       torque against the hand-written μ_B Σ m e × B.
# D4  — the `carryover = false` restart of the constrained mover: back to the
#       default pattern sign(w) n̂ written here from the weights, no RNG consumed
#       (every stream compared with a copy taken before); the annealing ladder and
#       the result.
# D4d — through `run_cmc`: the first measured configuration of the second
#       temperature lies within one sweep of the default pattern (the restart is
#       not a no-op).
# G6d — through the public driver: the mean torque of an N = 2 dimer against the
#       deterministic quadrature of the constrained partition function Z₂(n̂)
#       (fixtures_cmc.jl; the oracle of the G6 sweep gate).
# G0d — the two end-of-thermalization warnings fire through `run_cmc` exactly when
#       their conditions hold, once per temperature and per run (no rate limit),
#       naming kT and n̂.
# G18 — through `run_cmc`: every `:torque_frame` sample is the frame projection of
#       the `:torque` sample of the same measurement, and every `:torque` sample is
#       `total_torque / n_cells` of the measured configuration (through the public
#       call of the observables); the binned means agree to rounding.
# G20 — through `run_cmc`: the chain-state torque path of the measurement gives
#       statistics bitwise equal to those of the same observables behind plain
#       closures (every value from the public `total_torque` call) — also over a
#       `carryover = false` ladder with frequent renormalization and across a
#       mid-measurement resume that crosses the two paths; `_measure_chain!` takes
#       the chain-state path (the slot evaluation counts).
# G19 — the start-time door: a set from `cmc_observables` built for another
#       direction, other weights, or another `H`, a torque observable bound to
#       another `H`, and (near a pole) a `:torque_frame` whose frame axes differ
#       are refused before the run starts; a set for the same constraint (also
#       rebuilt from a re-normalized direction an ulp away, `:auto` vs the equal
#       `:magmoms`, the angle vs the vector form of one direction, poles included)
#       runs.
# Doors — the checkpoint-interval door, keywords of `run_mc` that CMC does not
#       have, invalid values, constraint-resolution and `init` errors.

using Logging: Logging, Warn

# --- helpers --------------------------------------------------------------------------

# Run `f` under a capturing logger; returns (value, records).
function _drv_logs(f)
    logger = Test.TestLogger(min_level = Warn)
    value = Logging.with_logger(f, logger)
    return value, logger.logs
end

# Warning records whose message matches `pattern`.
_drv_matching(logs, pattern) = filter(l -> occursin(pattern, string(l.message)), logs)

# G6 through the driver: the F2u dimer at (θ, φ) = (0.6, 0.35), β = 20, uniform
# weights (`:auto` on a moment-free H), DEFAULT step adaptation (target 0.5, window
# 50, thermalization only — the measurement phase runs on the frozen step).
function _drv_g6(seed)
    θ, φ = 0.6, 0.35
    β = 20.0
    H = MC.TiledHamiltonian(2, _f2u_terms())
    Ef, J, _, _, recon = _cmcref_dimer_model(H)
    ref, conv = _cmcref_torque_ref(Ef, β, 1.0, 1.0, θ, φ)
    u = _cmcref_dir(θ, φ)
    # Cartesian reference: ⟨T⟩ = (⟨T⟩·ê_φ) ê_φ + (⟨T⟩·ê_θ) ê_θ + (⟨T⟩·n̂) n̂
    Tref = ref[1] * _cmcref_eφ(θ, φ) + ref[2] * _cmcref_eθ(θ, φ) + ref[3] * u
    r = run_cmc(H, MagnetizationConstraint(θ, φ); kT = 1 / β, sweeps_therm = 2_000,
                sweeps_measure = 200_000, seed = seed)
    p = r.points[1]
    T = p.stats[:torque]
    # x and y are the statistical rows. The z component is not: F2u is invariant
    # under rotations about ẑ, so T·ẑ = 0 configuration by configuration (rounding
    # only) — it is returned separately and checked as an identity.
    rows = [(; name = "F2u θ=0.6 T_$(("x", "y")[c])", value = T.mean[c],
             σ = T.err[c], exact = Tref[c], εmax = _DRV_G6_EPS[c]) for c = 1:2]
    return rows, p, T.mean[3], Tref[3], conv, recon, β * J
end

# σ caps of the G6 rows: ≈ 2× the largest binning σ over 4 seeds (6101–6104,
# measured 2026-11-21): T_x 6.4e-5, T_y 1.24e-4.
const _DRV_G6_EPS = (1.3e-4, 2.5e-4)

@testset "CMC driver" begin
    n̂ = _cmcref_dir(0.6, 0.35)

    @testset "OBS: definitions and normalization (n_cells = 2)" begin
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        @test H.n_sites == 16                          # n_cells = 2 ≠ n_sites
        w = _fmix_weights()
        con = MagnetizationConstraint(n̂; weights = w)
        obs = cmc_observables(H, con)
        @test [o.name for o in obs] ==
              [:energy, :energy2, :torque, :M_par, :constraint_residual, :torque_frame]
        @test [o.ncomp for o in obs] == [1, 1, 3, 1, 1, 3]
        byname = Dict(o.name => o for o in obs)
        rng = MersenneTwister(7101)
        for _ = 1:5
            x = _rand_config(rng, H)
            E = total_energy(H, x)
            # M = Σ_{active, w ≠ 0} w e, written from the fixture's definition: site
            # s carries atom site_atom(s); atom 8 is inactive, atom 7 has w = 0
            M = zero(SVector{3,Float64})
            for s = 1:16
                a = MC.site_atom(H, s)
                a == _FMIX_INACTIVE && continue
                M += w[a] * x[s]
            end
            Mpar = dot(M, n̂)
            @test byname[:energy].f(x, E, H) == E
            @test byname[:energy2].f(x, E, H) == E^2
            @test byname[:torque].f(x, E, H) ≈ MC.total_torque(H, x) / 2 rtol = 1e-14
            @test byname[:M_par].f(x, E, H) ≈ Mpar / 2 rtol = 1e-12
            Mpar > 0 && @test byname[:constraint_residual].f(x, E, H) ≈
                              norm(M - Mpar * n̂) / Mpar rtol = 1e-12
        end
        # on the constrained start the residual is rounding-level and M∥ = Σ|w|/2
        x0 = MC._cmc_initial_config(H, MC._resolve_constraint(H, con), nothing)
        @test byname[:M_par].f(x0, 0.0, H) ≈ 2 * sum(abs, w[1:7]) / 2 rtol = 1e-14
        @test byname[:constraint_residual].f(x0, 0.0, H) < 1e-15
        # bound to its H: an equal but distinct Hamiltonian is refused
        H2 = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        for name in (:torque, :M_par, :constraint_residual, :torque_frame)
            @test_throws ArgumentError byname[name].f(x0, 0.0, H2)
        end
        # the field adds the analytic Zeeman torque, per cell:
        # T_Z = μ_B Σ_active m_a e × B / n_cells (decision record D5)
        Hz = MC.TiledHamiltonian(2, _f2g_terms(); dims = (2, 1, 1),
                                 magmoms = _fz_magmoms(), field = _fz_field())
        obz = cmc_observables(Hz, MagnetizationConstraint(n̂))
        @test [o.name for o in obz] ==
              [:energy, :energy2, :torque, :M_par, :constraint_residual, :torque_frame,
               :torque_zeeman]
        x = _rand_config(MersenneTwister(7102), Hz)
        Tz = sum(MU_B_EV_T * _fz_magmoms()[MC.site_atom(Hz, s)] * cross(x[s], _fz_field())
                 for s = 1:4)
        @test obz[end].f(x, 0.0, Hz) ≈ Tz / 2 rtol = 1e-13
        # :auto resolves to the moments here: M∥ = Σ m (e·n̂) / n_cells
        Mm = sum(_fz_magmoms()[MC.site_atom(Hz, s)] * x[s] for s = 1:4)
        @test obz[4].f(x, 0.0, Hz) ≈ dot(Mm, n̂) / 2 rtol = 1e-13
        # the Zeeman torque is built from the MOMENTS whatever the constraint
        # weights, while :M_par follows the weights: explicit (1, −3) and :uniform
        for (w, wvec) in (([1.0, -3.0], [1.0, -3.0]), (:uniform, [1.0, 1.0]))
            ob = cmc_observables(Hz, MagnetizationConstraint(n̂; weights = w))
            @test ob[end].name === :torque_zeeman
            @test ob[end].f(x, 0.0, Hz) ≈ Tz / 2 rtol = 1e-13
            Mw = sum(wvec[MC.site_atom(Hz, s)] * x[s] for s = 1:4)
            @test ob[4].f(x, 0.0, Hz) ≈ dot(Mw, n̂) / 2 rtol = 1e-13
        end
        # observables measure along the constraint they were BUILT for — not the
        # run's (a run refuses a mismatched set at its start, G19): a mismatch
        # shows as a value
        n̂2 = _cmcref_dir(1.1, -0.4)
        ob1 = cmc_observables(Hz, MagnetizationConstraint(n̂))
        ob2 = cmc_observables(Hz, MagnetizationConstraint(n̂2))
        @test ob2[4].f(x, 0.0, Hz) ≈ dot(Mm, n̂2) / 2 rtol = 1e-13
        @test abs(ob2[4].f(x, 0.0, Hz) - ob1[4].f(x, 0.0, Hz)) > 1e-3
        # the evaluables: specific heat only, the standard definition
        ev = cmc_evaluables()
        @test [e.name for e in ev] == [:specific_heat]
        @test ev[1].inputs == [:energy, :energy2]
        @test ev[1].f((; energy = 2.0, energy2 = 5.0), 0.5, 4) ≈ (5.0 - 4.0) / (4 * 0.25)
    end

    @testset "D4: carryover = false restart is deterministic and RNG-free" begin
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        w = _fmix_weights()
        rc = MC._resolve_constraint(H, MagnetizationConstraint(n̂; weights = w))
        start = MC._cmc_initial_config(H, rc, nothing)
        mover = MC._CMCMover(H, rc, MC._CMCScratch(H), start)
        plan = MC.UpdatePlan([0.05]; sweeps_therm = 0, sweeps_measure = 1,
                             measure_interval = 1, or_per_metropolis = 0, step = 0.7,
                             adapt_target = 0.5, adapt_interval = 50,
                             renorm_interval = 1_000, nbins = 2, carryover = false,
                             seed = 1)
        st = MC.ChainState(H, copy(start), Xoshiro(7201), 0.7)
        scs = [MC.SweepScratch(H)]
        for _ = 1:20
            MC._mover_sweep!(mover, st, H, 20.0, scs, plan)
        end
        st.step = 0.123
        @test st.config != start                       # the chain has moved
        rng0 = copy(st.rng)
        site0 = copy.(st.site_rngs)
        MC._restart_chain!(mover, st, H, plan)
        # the default pattern, written from the weights: sign(w) n̂ on 𝒞, n̂ elsewhere
        # (the zero-weight atom 7 and the inactive atom 8 included)
        expect = [w[MC.site_atom(H, s)] < 0 ? -n̂ : n̂ for s = 1:H.n_sites]
        @test st.config == expect
        @test st.energy ≈ total_energy(H, MC.SpinConfig(expect)) rtol = 1e-13
        @test st.zrows == MC.ChainState(H, MC.SpinConfig(expect), Xoshiro(1), 1.0).zrows
        @test st.step == 0.7
        @test st.rng == rng0                           # no RNG consumed
        @test st.site_rngs == site0
        @test mover.start == expect                    # the stored start untouched
        # the mover asserts that its buffers belong to H
        Hs = MC.TiledHamiltonian(2, _fconst_terms(2))       # l = 0 only: nlm = 1
        @test Hs.nlm != H.nlm
        @test_throws AssertionError MC._CMCMover(H, rc, MC._CMCScratch(Hs), start)
    end

    @testset "D4d: carryover = false restarts the second temperature" begin
        # A constant-energy chain with a tiny frozen step barely moves: one sweep
        # displaces a spin by O(step) = O(1e-4). The FIRST measured configuration of
        # each temperature (sweeps_therm = 0, so after one sweep) must therefore lie
        # within a few 1e-4 of the default pattern (every spin along n̂) — at the
        # second temperature too, if the restart really happened. A restart that is
        # a no-op (the chain aliasing the stored start, so the "restart" copies the
        # evolved state onto itself) leaves it after 2000 sweeps of random walk,
        # ~√2000 · 1e-4 away.
        H = MC.TiledHamiltonian(1, _fconst_terms(1); dims = (8, 1, 1))
        firsts = MC.SpinConfig[]
        count = Ref(0)
        spy = Observable(:spy, 1, (config, E, Hrun) -> begin
            count[] % 2000 == 0 && push!(firsts, copy(config))
            count[] += 1
            0.0
        end)
        obs = vcat(cmc_observables(H, MagnetizationConstraint(n̂)), spy)
        # (a step below the adaptation clamp also triggers the step-floor warning;
        # captured here, it is not what this testset is about)
        _drv_logs(() -> run_cmc(H, MagnetizationConstraint(n̂); kT = [1.0, 1.0],
                                step = 1e-4, sweeps_therm = 0, sweeps_measure = 2000,
                                carryover = false, nbins = 2, observables = obs,
                                seed = 7202))
        @test length(firsts) == 2
        dev = [maximum(s -> norm(f[s] - n̂), 1:8) for f in firsts]
        _CMC_VERBOSE &&
            println("D4d  first-config deviation from n̂ per temperature ", dev)
        @test all(d -> 0 < d < 1e-3, dev)
    end

    @testset "D4: annealing ladder, carryover, and the result" begin
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        con = MagnetizationConstraint(n̂; weights = _fmix_weights())
        kw = (; sweeps_therm = 100, sweeps_measure = 200, nbins = 4)
        a = run_cmc(H, con; kT = [0.1, 0.05], kw..., seed = 7301)
        @test a isa CMCResult
        @test [p.kT for p in a.points] == [0.1, 0.05]
        @test [p.temperature for p in a.points] == [0.1, 0.05] ./ KB_EV
        @test all(p -> isnan(p.acceptance_or), a.points)
        @test all(p -> 0 < p.acceptance_metropolis < 1, a.points)
        @test sort(collect(keys(a.points[1].stats))) ==
              [:M_par, :constraint_residual, :energy, :energy2, :specific_heat, :torque,
               :torque_frame]
        # every measured configuration kept the constraint
        @test all(p -> p.stats[:constraint_residual].mean[1] < 1e-12, a.points)
        @test all(p -> p.stats[:M_par].mean[1] > 0, a.points)
        # the result's constraint: the run's direction bitwise, resolved weights
        @test a.constraint.direction == con.direction
        @test a.constraint.weights == _fmix_weights()
        @test a.seed == 7301
        x = a.final_config
        @test dot(_cmcref_M(x, [_fmix_weights()[MC.site_atom(H, s)] for s = 1:16],
                        [MC.site_atom(H, s) != _FMIX_INACTIVE for s = 1:16]), n̂) > 0
        # the direction is carried over bitwise, not normalized again: normalizing a
        # normalized vector moves about a third of them by an ulp (non-vacuity
        # asserted on the directions below)
        Hs = MC.TiledHamiltonian(2, _f2g_terms())
        rng = MersenneTwister(7303)
        moved = 0
        for _ = 1:12
            c = MagnetizationConstraint(randn(rng, 3))
            moved += MagnetizationConstraint(c.direction).direction != c.direction
            r = run_cmc(Hs, c; kT = 0.1, sweeps_therm = 0, sweeps_measure = 2, nbins = 2,
                        seed = 1)
            @test r.constraint.direction == c.direction
        end
        @test moved >= 1
        # :auto / :uniform are reported resolved
        b = run_cmc(H, MagnetizationConstraint(n̂); kT = 0.1, kw..., seed = 7302)
        @test b.constraint.weights == ones(8)
        # a −0.0 moment (admitted by the field door) is reported as +0.0, so the
        # result's constraint agrees in isequal / hash with one built from the
        # same numbers
        H3 = MC.TiledHamiltonian(3, _f3_terms(); magmoms = [2.0, -0.0, 1.0],
                                 field = (0.0, 0.0, 1.0))
        r3 = run_cmc(H3, MagnetizationConstraint(n̂); kT = 0.1, sweeps_therm = 0,
                     sweeps_measure = 2, nbins = 2, seed = 7304)
        ref3 = MagnetizationConstraint(n̂; weights = [2.0, 0.0, 1.0])
        @test isequal(r3.constraint, ref3)
        @test hash(r3.constraint) == hash(ref3)
        @test !signbit(r3.constraint.weights[2])
        # determinism, and the ladder's first point does not depend on what follows
        @test run_cmc(H, con; kT = [0.1, 0.05], kw..., seed = 7301).final_config == x
        single = run_cmc(H, con; kT = 0.1, kw..., seed = 7301)
        _assert_same_result(single, CMCResult(a.points[1:1], a.constraint, x, a.seed))
        # carryover = false restarts the second temperature (a different chain)
        c = run_cmc(H, con; kT = [0.1, 0.05], kw..., carryover = false, seed = 7301)
        _assert_same_result(single, CMCResult(c.points[1:1], c.constraint, x, c.seed))
        @test c.final_config != x
        # `temperature` in kelvin is the same ladder
        d = run_cmc(H, con; temperature = [0.1, 0.05] ./ KB_EV, kw..., seed = 7301)
        @test [p.kT for p in d.points] ≈ [0.1, 0.05] rtol = 1e-15
        # show: the compact form and the table (torque columns, no χ / U columns)
        @test sprint(show, a) == "CMCResult(2 temperatures, 16 sites)"
        txt = sprint(show, MIME"text/plain"(), a)
        @test occursin("CMCResult: 2 temperature(s), 16 sites, seed 7301", txt)
        @test occursin("torque_x", txt) && occursin("torque_z", txt) &&
              occursin("M_par", txt) && occursin(":torque_frame", txt)
        @test !occursin("chi", txt)
        # long weight vectors are abbreviated; a user `:torque` that is not a
        # 3-vector prints as "-" instead of failing
        @test MC._weights_str(collect(1.0:10.0)) ==
              "[1, 2, 3, 4, 5, 6, … (10 cell atoms)]"
        odd = TempResult(0.1, 0.1 / KB_EV,
                         Dict(:torque => ObservableStat(:torque, [1.0], [0.1], [NaN], 1)),
                         0.5, NaN, 0.6, 0.0)
        txt2 = sprint(show, MIME"text/plain"(),
                      CMCResult([odd], a.constraint, x, UInt64(1)))
        @test occursin(r"0\.1\s+\S+\s+-\s+-\s+-\s+-", txt2)
        # header, constraint, column names, two rows, the closing note
        @test length(split(chomp(txt), '\n')) == 6
    end

    @testset "G6d: N = 2 dimer torque vs quadrature, through run_cmc" begin
        # The oracle is the quadrature of the G6 gate (fixtures_cmc.jl): Z₂(n̂) on a
        # Gauss–Legendre × trapezoid grid, F = −kT ln Z₂, ⟨T⟩ from the five-point
        # derivative; Cartesian here because `:torque` is Cartesian per cell (one
        # cell: n_cells = 1). Default step adaptation (target 0.5, window 50, 2000
        # thermalization sweeps; the adapted step, 1.38–1.64 rad over the seeds
        # below, is frozen for the 2·10⁵ measurement sweeps).
        # Measured (seed 6101): T_x −9.887e-3 vs exact −9.922e-3, σ 6.0e-5, +0.58σ;
        # T_y 2.7247e-2 vs 2.7182e-2, σ 1.21e-4, +0.54σ. Over seeds 6101–6104:
        # |dev| ≤ 0.82σ, σ(T_x) ≤ 6.4e-5, σ(T_y) ≤ 1.24e-4; atol = 5σ. The
        # constraint residual averages ~1e-13 (rounding of the renormalizations,
        # a random walk the decision record estimates).
        # Mutant: run_cmc dispatching to the unconstrained `_SweepMover` samples
        # the free dimer, whose mean torque vanishes — T_x +9.83e-3, T_y −2.70e-2,
        # which the gate reports as 83σ / 235σ in the mutant run's OWN σ (the σ the
        # gate divides by; 164σ / 222σ in the σ of the correct run); see the
        # mutation table of docs/specs/constrained-mc.md.
        rows, p, Tz, Tz_ref, conv, recon, βJ = _drv_g6(6101)
        @test recon < 1e-14
        @test -2 <= βJ <= -1
        @test conv < 1e-10
        @test abs(Tz) < 1e-15 && abs(Tz_ref) < 1e-12        # rotation about ẑ
        @test p.stats[:constraint_residual].mean[1] < 1e-12
        _cmcref_check_rows("G6d", rows)
    end

    @testset "G0d: end-of-thermalization warnings, through run_cmc" begin
        H2 = MC.TiledHamiltonian(2, _f2u_terms())
        J = _dimer_J(H2)
        @test J < 0                                     # ferro
        kw = (; sweeps_therm = 2_000, sweeps_measure = 100)

        # weak constraint FIRES: weights (1, −1) on a ferro dimer far below |J| —
        # the spins align, so M = e₁ − e₂ nearly vanishes: |M| is the angle between
        # them, of order √(kT/|J|) = 0.01 at kT = 1e-4 |J|, so M∥/Σ|w| ~ 0.005
        # against the threshold 0.05 (a 10-σ-like margin; at 1e-3 |J| the tail
        # above 0.05 is a few per cent). Once per temperature and per run, with
        # kT and n̂ in the record.
        conw = MagnetizationConstraint(n̂; weights = [1.0, -1.0])
        kts = abs(J) .* [1e-4, 2e-5]
        _, logs = _drv_logs(() -> run_cmc(H2, conw; kT = kts, kw..., seed = 7401))
        weak = _drv_matching(logs, "weak constraint")
        _CMC_VERBOSE && println("G0d  weak ratios ", [l.kwargs[:ratio] for l in weak])
        @test length(weak) == 2
        @test [l.kwargs[:kT] for l in weak] == kts
        @test all(l -> l.kwargs[:direction] == Tuple(conw.direction), weak)
        @test all(l -> l.kwargs[:ratio] < 0.05, weak)
        @test occursin("kT = $(kts[1])", string(weak[1].message))
        _, logs2 = _drv_logs(() -> run_cmc(H2, conw; kT = kts[1], kw..., seed = 7402))
        @test length(_drv_matching(logs2, "weak constraint")) == 1   # a second run too

        # the hook sees the END of thermalization: with sweeps_therm = 0 that is the
        # constrained start (e₁ = n̂, e₂ = −n̂, M∥/Σ|w| = 1), so the weak warning must
        # NOT fire — although the measurement then drives the same chain to the
        # nearly cancelling state that warns above (a hook placed after the
        # measurement would warn here)
        @test_logs min_level = Warn run_cmc(H2, conw; kT = kts[1], sweeps_therm = 0,
                                            sweeps_measure = 2_000, seed = 7401)

        # neither fires on an ordered ferro under uniform weights
        @test_logs min_level = Warn run_cmc(H2, MagnetizationConstraint(n̂);
                                            kT = abs(J) / 3, kw..., seed = 7403)

        # step floor FIRES: constant energy, weights (1, 1000). A move with the
        # w = 1000 source pushes the w = 1 partner by 1000× its own displacement, so
        # it is null unless the rotation is ≲ 1e-3 rad; with adapt_target = 0.95 the
        # pooled acceptance (measured 0.72 at the floor) never reaches the target and
        # the step is driven onto the clamp
        Hc = MC.TiledHamiltonian(2, _fconst_terms(2))
        conr = MagnetizationConstraint(n̂; weights = [1.0, 1000.0])
        r, logs = _drv_logs(() -> run_cmc(Hc, conr; kT = [1.0, 0.5], kw...,
                                          adapt_target = 0.95, seed = 7404))
        floor = _drv_matching(logs, "lower clamp")
        @test length(floor) == 2
        @test [l.kwargs[:kT] for l in floor] == [1.0, 0.5]
        @test all(l -> l.kwargs[:direction] == Tuple(conr.direction), floor)
        @test all(p -> p.final_step == MC._STEP_MIN, r.points)
        @test isempty(_drv_matching(logs, "weak constraint"))
        # ... and does not fire at the default target (measured: the step stays at
        # 0.6, the pooled acceptance is 0.5)
        r = @test_logs min_level = Warn run_cmc(Hc, conr; kT = 1.0, kw..., seed = 7404)
        @test r.points[1].final_step > 10 * MC._STEP_MIN
    end

    @testset "G18: :torque_frame through run_cmc, sample by sample" begin
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))   # n_cells = 2
        θ, φ = 0.6, 0.35
        con = MagnetizationConstraint(θ, φ; weights = _fmix_weights())
        # recorders around the default observables: each call is forwarded
        # unchanged, so the run measures exactly the default set
        configs, torques, frames = MC.SpinConfig[], SVector{3,Float64}[],
                                   SVector{3,Float64}[]
        function rec(o, sink)
            return Observable(o.name, o.ncomp, (c, E, Hr) -> begin
                                  v = o.f(c, E, Hr)
                                  push!(sink, SVector{3,Float64}(v))
                                  o.name === :torque && push!(configs, copy(c))
                                  return v
                              end)
        end
        obs = map(cmc_observables(H, con)) do o
            o.name === :torque ? rec(o, torques) :
            o.name === :torque_frame ? rec(o, frames) : o
        end
        r = run_cmc(H, con; kT = 0.05, sweeps_therm = 50, sweeps_measure = 120,
                    measure_interval = 3, nbins = 4, observables = obs, seed = 7501)
        @test length(torques) == length(frames) == length(configs) == 40
        eφ, eθ, n = _cmcref_eφ(θ, φ), _cmcref_eθ(θ, φ), _cmcref_dir(θ, φ)
        for k in eachindex(torques)
            @test torques[k] == MC.total_torque(H, configs[k]) / 2
            T = torques[k]
            @test frames[k] ≈ [dot(eφ, T), dot(eθ, T), dot(n, T)] atol = 1e-15
        end
        @test length(unique(torques)) == 40                       # non-vacuity
        st, sf = r.points[1].stats[:torque], r.points[1].stats[:torque_frame]
        @test sf.mean ≈ [dot(eφ, st.mean), dot(eθ, st.mean), dot(n, st.mean)] atol = 1e-15
        @test sf.count == st.count == 40
    end

    @testset "G20: run_cmc's chain-state torque ≡ the public call, bitwise" begin
        # Fmix: signed weights, a weight-0 (hybrid) site, an inactive site; partial
        # torques of every body order next to the default set
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        con = MagnetizationConstraint(0.6, 0.35; weights = _fmix_weights())
        parts = [torque_observable(Symbol(:torque_body, b),
                                   MC.TiledHamiltonian(8, filter(t -> t.body == b,
                                                                 _fmix_terms());
                                                       dims = (2, 1, 1)), H)
                 for b = 1:3]
        obs = [cmc_observables(H, con); parts]
        # the same observables behind plain closures: the measurement sees a generic
        # function, so every value comes from the public call (`total_torque`)
        plain = [Observable(o.name, o.ncomp, (c, E, Hr) -> o.f(c, E, Hr)) for o in obs]
        kw = (; kT = 0.05, sweeps_therm = 30, sweeps_measure = 90, measure_interval = 2,
              nbins = 5, seed = 7701)
        r = run_cmc(H, con; kw..., observables = obs)
        rp = run_cmc(H, con; kw..., observables = plain)
        @test r.final_config == rp.final_config
        for o in obs
            a, b = r.points[1].stats[o.name], rp.points[1].stats[o.name]
            @test a.mean == b.mean && a.err == b.err && a.tau_int == b.tau_int &&
                  a.count == b.count == 45
        end
        # non-vacuity: every torque statistic is nonzero and fluctuates
        for name in (:torque, :torque_frame, :torque_body1, :torque_body2,
                     :torque_body3)
            s = r.points[1].stats[name]
            @test norm(s.mean) > 0 && all(>(0), s.err)
        end
        # the rows across the other state changes of a run: a two-temperature ladder
        # with `carryover = false` (the restart resets the configuration and its
        # rows), a renormalization every 7 sweeps, and a resume from a
        # mid-measurement interruption of the second temperature — crossing the two
        # paths (interrupted on one, resumed on the other)
        _, benign = _poison_pair(1)
        kw2 = (; kT = [0.08, 0.05], carryover = false, sweeps_therm = 20,
               sweeps_measure = 60, measure_interval = 2, renorm_interval = 7,
               nbins = 5, seed = 7702)
        ref = run_cmc(H, con; kw2..., observables = [obs; benign])
        _assert_same_result(ref, run_cmc(H, con; kw2..., observables = [plain; benign]))
        for (first_set, resumed_set) in ((obs, plain), (plain, obs))
            poison, _ = _poison_pair(30 + 12)       # measurement 12 of temperature 2
            ck = tempname() * ".jld2"
            err = _cmcref_thrown(() -> run_cmc(H, con; kw2...,
                                               observables = [first_set; poison],
                                               checkpoint = ck, checkpoint_interval = 5))
            @test err isa ErrorException && occursin("poison", err.msg)
            t, phase, sweep = MC.jldopen(ck, "r") do f
                (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
            end
            @test t == 2 && phase == "measure" && 0 < sweep < 60       # mid-measure
            res = resume(ck, H; observables = [resumed_set; benign], checkpoint = nothing)
            _assert_same_result(ref, res)
            @test res.final_config == ref.final_config
        end
        @test ref.points[1].stats[:torque].mean != ref.points[2].stats[:torque].mean
    end

    @testset "G20: the run's measurement takes the chain-state path" begin
        # `_measure_chain!` (the measurement of `_run_temperature!`) must route every
        # torque observable through the chain state: one gradient pass per
        # Hamiltonian and measurement, counted on the chain's scratch. A measurement
        # through the public call leaves the scratch untouched (same values — the
        # bitwise gates above cannot see it — but a row rebuild and a gradient
        # vector per observable).
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        con = MagnetizationConstraint(0.6, 0.35; weights = _fmix_weights())
        Hpair = MC.TiledHamiltonian(8, filter(t -> t.body == 2, _fmix_terms());
                                    dims = (2, 1, 1))
        obs = [cmc_observables(H, con); torque_observable(:torque_pair, Hpair, H)]
        accs = [MC.ObsAccumulator(o, 16, 2) for o in obs]
        rc = MC._resolve_constraint(H, con)
        st = MC.ChainState(H, MC._cmc_initial_config(H, rc, nothing), Xoshiro(2101),
                           0.8)
        sc, scratch = MC.SweepScratch(H), MC._CMCScratch(H)
        ms = MC._MeasureScratch()
        for _ = 1:6
            MC._cmc_sweep!(st, H, 1 / 0.05, sc, rc, scratch)
            MC._measure_chain!(accs, st, H, ms)
        end
        @test ms.stamp == 6
        # two slots — H (`:torque` and `:torque_frame` share it) and Hpair — each
        # evaluated once per measurement
        @test length(ms.slots) == 2
        @test count(s -> s.Hgrad === H, ms.slots) == 1 &&
              count(s -> s.Hgrad === Hpair, ms.slots) == 1
        @test all(s -> s.evaluations == 6, ms.slots)
        @test all(acc -> acc.binner.n == 6, accs)            # every observable recorded
    end

    @testset "G19: observables built for another constraint are refused at the start" begin
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        w = _fmix_weights()
        con = MagnetizationConstraint(n̂; weights = w)
        kw = (; kT = 0.1, sweeps_therm = 5, sweeps_measure = 5, nbins = 2, seed = 7601)
        refused(c2) = _cmcref_thrown(() -> run_cmc(H, con; kw...,
                                                   observables = cmc_observables(H, c2)))
        # another direction (0.01 rad away), another weight on one active atom, the
        # weights scaled (M∥ in other units), another H
        for c2 in (MagnetizationConstraint(_cmcref_dir(0.6, 0.36); weights = w),
                   MagnetizationConstraint(n̂; weights = [w[1:5]; 1.0; w[7:8]]),
                   MagnetizationConstraint(n̂; weights = 2 .* w))
            err = refused(c2)
            @test err isa ArgumentError && occursin("another constraint", err.msg)
        end
        H2 = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        err = _cmcref_thrown(() -> run_cmc(H, con; kw...,
                                           observables = cmc_observables(H2, con)))
        @test err isa ArgumentError && occursin("different TiledHamiltonian", err.msg)
        # a torque observable without a constraint tag bound to another H object:
        # `:torque` of a set for H2, and a term-subset torque built against H2 (its
        # Hpart is a legitimate subset; the full H it is bound to is not the run's).
        # Refused by the door, before the run starts — measured, it would throw only
        # at the first measurement, after the thermalization and a checkpoint write
        Hpair = MC.TiledHamiltonian(8, filter(t -> t.body == 2, _fmix_terms());
                                    dims = (2, 1, 1))
        tq2 = only(filter(o -> o.name === :torque, cmc_observables(H2, con)))
        for foreign in (tq2, torque_observable(:torque_pair, Hpair, H2))
            set = [filter(o -> o.name !== foreign.name, cmc_observables(H, con));
                   foreign]
            ck = tempname() * ".jld2"
            err = _cmcref_thrown(() -> run_cmc(H, con; kw..., checkpoint = ck,
                                               checkpoint_interval = 1,
                                               observables = set))
            @test err isa ArgumentError && startswith(err.msg, "run_cmc: observable") &&
                  occursin("different TiledHamiltonian", err.msg)
            @test !isfile(ck)
        end
        # ... admitted when bound to the run's H (the subset Hpart is not the run's H)
        @test run_cmc(H, con; kw...,
                      observables = [cmc_observables(H, con);
                                     torque_observable(:torque_pair, Hpair, H)]) isa
              CMCResult
        # near a pole the frame of :torque_frame is compared on its own: a run at
        # n̂ = ẑ (pole frame (ŷ, x̂)) and observables built for (θ, φ) = (5e-7, π/2) —
        # inside the direction tolerance (chord 5e-7), but with the frame (−x̂, ŷ):
        # the tangent components would be exchanged. Refused both ways round
        cz = MagnetizationConstraint(0.0, 0.0; weights = w)
        cnear = MagnetizationConstraint(5e-7, π / 2; weights = w)
        @test norm(cnear.direction - cz.direction) < 1e-6        # within the chord
        for (crun, cobs) in ((cz, cnear), (cnear, cz))
            err = _cmcref_thrown(() -> run_cmc(H, crun; kw...,
                                               observables = cmc_observables(H, cobs)))
            @test err isa ArgumentError && occursin(":torque_frame", err.msg) &&
                  occursin("torque frame", err.msg)
        end
        # ... while the angle and the vector forms of one direction pass, poles
        # included (the Float64 θ = π leaves sinθ ≈ 1.2e-16 in x/y; the vector form
        # is normalized again)
        πf = Float64(π)
        for (ca, cv) in (((0.0, 1.3), [0.0, 0.0, 1.0]), ((πf, 1.3), [0.0, 0.0, -1.0]),
                         ((π, 0.0), [0.0, 0.0, -1.0]), ((π / 2, 0.0), [1.0, 0.0, 0.0]),
                         ((0.6, 0.35), collect(_cmcref_dir(0.6, 0.35))),
                         ((2.0, -1.0), 3 .* collect(_cmcref_dir(2.0, -1.0))))
            a = MagnetizationConstraint(ca...; weights = w)
            v = MagnetizationConstraint(cv; weights = w)
            @test run_cmc(H, a; kw..., observables = cmc_observables(H, v)) isa CMCResult
            @test run_cmc(H, v; kw..., observables = cmc_observables(H, a)) isa CMCResult
        end
        # refused BEFORE anything runs: no checkpoint file is written
        ck = tempname() * ".jld2"
        @test_throws ArgumentError run_cmc(
            H, con; kw..., checkpoint = ck,
            observables = cmc_observables(H, MagnetizationConstraint(-n̂; weights = w)))
        @test !isfile(ck)
        # admitted: the same constraint; one rebuilt from a direction that moves by
        # an ulp when normalized again; observables of your own around the set
        @test run_cmc(H, con; kw..., observables = cmc_observables(H, con)) isa CMCResult
        rng = MersenneTwister(7602)
        conr = con
        while true
            c = MagnetizationConstraint(randn(rng, 3); weights = w)
            MagnetizationConstraint(c.direction; weights = w).direction !=
            c.direction || continue
            conr = c
            break
        end
        again = MagnetizationConstraint(conr.direction; weights = w)
        @test again.direction != conr.direction                     # an ulp away
        @test run_cmc(H, conr; kw..., observables = cmc_observables(H, again)) isa
              CMCResult
        # a weight on the inactive atom 8 enters no measurement
        @test run_cmc(H, con; kw...,
                      observables = cmc_observables(
                          H, MagnetizationConstraint(n̂; weights = [w[1:7]; 5.0]))) isa
              CMCResult
        extra = Observable(:e3, 1, (c, E, Hr) -> E^3)
        @test run_cmc(H, con; kw..., observables = [cmc_observables(H, con); extra]) isa
              CMCResult
        # `:auto` and `:magmoms` resolve to the same weights on a moment-carrying H
        Hz = MC.TiledHamiltonian(2, _f2g_terms(); dims = (2, 1, 1),
                                 magmoms = _fz_magmoms(), field = _fz_field())
        @test run_cmc(Hz, MagnetizationConstraint(n̂); kw...,
                      observables = cmc_observables(
                          Hz, MagnetizationConstraint(n̂; weights = :magmoms))) isa
              CMCResult
        @test_throws ArgumentError run_cmc(
            Hz, MagnetizationConstraint(n̂); kw...,
            observables = cmc_observables(Hz, MagnetizationConstraint(n̂;
                                                                      weights = :uniform)))
    end

    @testset "Doors: run_cmc arguments" begin
        H = MC.TiledHamiltonian(2, _f2g_terms())
        con = MagnetizationConstraint(n̂)
        kw = (; kT = 0.1, sweeps_therm = 10, sweeps_measure = 10, nbins = 2)
        # checkpointing (test_cmc_checkpoint.jl): the shared interval door
        @test_throws r"checkpoint_interval must be" run_cmc(
            H, con; kw..., checkpoint = tempname() * ".jld2", checkpoint_interval = -1)
        # keywords of run_mc that a constrained run does not have: explained refusals
        @test_throws r"no overrelaxation" run_cmc(H, con; kw..., or_per_metropolis = 1)
        @test_throws r"or_per_metropolis" run_cmc(H, con; kw..., or_per_metropolis = 0)
        @test_throws r"serial" run_cmc(H, con; kw..., sweep_tasks = 2)
        @test_throws MethodError run_cmc(H, con; kw..., no_such_keyword = 1)
        # invalid values (the shared UpdatePlan / resolve_kt doors)
        @test_throws ArgumentError run_cmc(H, con; kw..., sweeps_measure = 0)
        @test_throws ArgumentError run_cmc(H, con; kw..., adapt_target = 1.0)
        @test_throws ArgumentError run_cmc(H, con; sweeps_measure = 10)   # no T
        @test_throws ArgumentError run_cmc(H, con; kw..., temperature = 300.0)
        @test_throws ArgumentError run_cmc(H, con; kw..., observables = Observable[])
        # χ and U are not CMC quantities: the standard evaluables need :m2 / :absm
        @test_throws r"needs observable" run_cmc(H, con; kw...,
                                                 evaluables = standard_evaluables())
        # constraint resolution
        @test_throws r"carries no magmoms" run_cmc(
            H, MagnetizationConstraint(n̂; weights = :magmoms); kw...)
        @test_throws r"2 cell atoms" run_cmc(
            H, MagnetizationConstraint(n̂; weights = [1.0, 1.0, 1.0]); kw...)
        @test_throws r"needs at least 2" run_cmc(
            H, MagnetizationConstraint(n̂; weights = [1.0, 0.0]); kw...)
        # init: a vanishing weighted magnetization, a wrong size
        ẑ = SVector(0.0, 0.0, 1.0)
        @test_throws r"direction is undefined" run_cmc(H, con; kw..., init = [ẑ, -ẑ])
        @test_throws DimensionMismatch run_cmc(H, con; kw..., init = [ẑ])
        # observables built for another Hamiltonian are refused at measurement
        H2 = MC.TiledHamiltonian(2, _f2g_terms())
        @test_throws r"different TiledHamiltonian" run_cmc(
            H, con; kw..., observables = cmc_observables(H2, con))
    end
end

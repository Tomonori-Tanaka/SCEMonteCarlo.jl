# Constrained MC, M6: checkpoint / resume of `run_cmc` (kind "cmc",
# docs/specs/checkpoint-schema.md; docs/specs/constrained-mc.md D6).
#
# G13 — resume ≡ uninterrupted, bitwise (`==` / `isequal` on every result-shaped
#       quantity, the final configuration, the result's constraint, and the chain
#       state of the final checkpoint file — energy, step, counters, the chain
#       and per-site RNG words, and the file's `start/config` and `constraint/`
#       as rewritten by the resumed writer). Every gate interrupts the WRITER mid-run with the
#       `_poison_pair` pattern (fixtures.jl) — `_mc_loop!` writes an unconditional
#       end-of-temperature checkpoint, so a completed file would only compare its
#       stored results with themselves — and asserts the file's position before
#       resuming. Covered: a three-temperature ladder with `carryover = true`; the
#       same with `carryover = false` and a user `init` (the resumed tail crosses a
#       restart, so the stored start is load-bearing); a mid-thermalization file
#       (inside an adaptation window) and a double resume (the second file written
#       by a resumed writer); the boundary-only cadence with `carryover = false`;
#       a field model with signed weights; non-trivial `measure_interval` /
#       `renorm_interval` throughout; and the default arguments of `resume`
#       (interrupted by a poisoned evaluable, so the file carries the default
#       observable names), on an idempotent and a non-idempotent direction.
# RT  — the stored constraint round-trips (`==`, `isequal`, `hash`; the direction
#       bitwise, a −0.0 weight folded).
# Doors — wrong H (fingerprint), a "cmc" file through the GPU method, a partner
#       rule other than "uniform", explicit standard observables / evaluables on a
#       "cmc" file, corrupted constraint / plan / start, a relabelled kind (refused
#       by name before the defaults are resolved), and the consistency door
#       (rewritten direction / start / chain that every other door admits).
# Kinds — an "mc" file resumed with default arguments still resolves the standard
#       observables and evaluables.

# A constraint whose stored direction is NOT idempotent under normalization
# (normalizing it again moves it by an ulp), from a fixed stream. The constraint
# itself is returned: building a new one from its direction would normalize again.
function _cmcck_nonidempotent_con(seed, weights)
    rng = MersenneTwister(seed)
    while true
        c = MagnetizationConstraint(randn(rng, 3); weights = weights)
        MagnetizationConstraint(c.direction).direction != c.direction && return c
    end
end

# Run `f`, which must die by the poison (an observable or an evaluable).
function _cmcck_interrupted(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("poison", err.msg)
    return nothing
end

_cmcck_progress(path) = MC.jldopen(path, "r") do f
    (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
end

# The chain block of a checkpoint file, every stored field, with the constrained
# run's own groups (written anew by every writer, a resumed one included).
_cmcck_chain(path) = MC.jldopen(path, "r") do f
    (; config = f["chain/config"], energy = f["chain/energy"], rng = f["chain/rng"],
     site_rngs = f["chain/site_rngs"], step = f["chain/step"],
     frozen = f["chain/frozen"], counters = f["chain/counters"],
     max_drift = f["chain/max_drift"], progress = _cmcck_progress(path),
     start = f["start/config"], direction = f["constraint/direction"],
     weights = f["constraint/weights"])
end

# Bitwise equality of two constrained results.
function _cmcck_same(a, c)
    @test c isa CMCResult
    _assert_same_result(a, c)
    @test a.final_config == c.final_config
    @test isequal(a.constraint, c.constraint)
    @test a.constraint.direction === c.constraint.direction
    @test a.seed == c.seed
    return nothing
end

# A poisoned evaluable: dies at its first finalization (the end of the first
# temperature) — interrupts a run whose observables are the defaults.
_cmcck_poison_ev() = Evaluable(:poison_ev, [:energy], (m, kT, n) -> error("poison"))

@testset "CMC checkpoint / resume (M6)" begin
    dir = mktempdir()
    n̂ = _sw_dir(0.6, 0.35)
    H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))   # 16 sites
    con = MagnetizationConstraint(n̂; weights = _fmix_weights())
    # three temperatures; measure_interval 3 and renorm_interval 70 put the resumed
    # tail off the measurement grid and across renormalizations
    base = (; kT = [0.1, 0.07, 0.05], sweeps_therm = 100, sweeps_measure = 300,
            measure_interval = 3, renorm_interval = 70, nbins = 4)

    @testset "G13: carryover = true, mid-measure" begin
        # 100 measurements per temperature; the poison fires at temperature 2's
        # 60th (measure sweep 180). Writes every 130 sweeps, the clock reset at
        # every write: temperature 2 writes at measure sweeps 30 and 160, so the
        # file holds measure sweep 160 (not a multiple of 3).
        poison, benign = _poison_pair(160)
        obs = [cmc_observables(H, con); benign]
        kw = (; base..., seed = 8101, observables = obs)
        a = run_cmc(H, con; kw...)                                  # no checkpoint
        pb = joinpath(dir, "cmc_b.jld2")
        b = run_cmc(H, con; kw..., checkpoint = pb, checkpoint_interval = 130)
        _cmcck_same(a, b)                                  # writing consumes no RNG
        @test _cmcck_progress(pb) == (4, "therm", 0)       # the end-of-run boundary
        path = joinpath(dir, "cmc_a.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kw...,
                                         observables = [cmc_observables(H, con); poison],
                                         checkpoint = path, checkpoint_interval = 130))
        ti, phase, sweep = _cmcck_progress(path)
        @test ti == 2 && phase == "measure" && sweep == 160    # genuinely mid-run
        c = resume(path, H; observables = obs)
        _cmcck_same(a, c)
        # the resumed run kept writing to `path` with the stored cadence; its final
        # chain state equals the uninterrupted checkpointed run's, RNG words included
        @test _cmcck_chain(path) == _cmcck_chain(pb)
    end

    @testset "G13: carryover = false with a user init, mid-measure" begin
        # The resumed tail crosses the restart of temperature 3, which returns to the
        # rotated `init` — derivable from nothing else in the file, so it must be
        # stored. Non-vacuity: the stored start is that rotated `init`, not the
        # default pattern.
        init = _rand_config(MersenneTwister(8102), H)
        rc = MC._resolve_constraint(H, con)
        start = MC._cmc_initial_config(H, rc, init)
        @test start != MC._cmc_initial_config(H, rc, nothing)
        poison, benign = _poison_pair(160)
        obs = [cmc_observables(H, con); benign]
        kw = (; base..., carryover = false, init = init, seed = 8103,
              observables = obs)
        a = run_cmc(H, con; kw...)
        pb = joinpath(dir, "cmc_init_b.jld2")
        b = run_cmc(H, con; kw..., checkpoint = pb, checkpoint_interval = 130)
        _cmcck_same(a, b)
        path = joinpath(dir, "cmc_init.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kw...,
                                         observables = [cmc_observables(H, con); poison],
                                         checkpoint = path, checkpoint_interval = 130))
        ti, phase, sweep = _cmcck_progress(path)
        @test ti == 2 && phase == "measure" && sweep == 160
        stored = MC.jldopen(f -> f["start/config"], path, "r")
        @test stored == _config_matrix(start)
        c = resume(path, H; observables = obs)
        _cmcck_same(a, c)
        @test _cmcck_chain(path) == _cmcck_chain(pb)
        # the restart really changes the chain: temperature 3 of the carried-over
        # ladder differs
        a_co = run_cmc(H, con; kw..., carryover = true)
        @test a_co.points[3].stats[:energy].mean != a.points[3].stats[:energy].mean
    end

    @testset "G13: mid-thermalization, then a double resume" begin
        # carryover = false + user init, writes every 70 sweeps. The poison fires at
        # temperature 2's first measurement (global measurement 101), so the file
        # holds therm sweep 70 of temperature 2 — inside an adaptation window
        # (adapt_interval 50: the window's counters are 20 sweeps in). The resumed
        # run is interrupted AGAIN (its own 50th measurement, measure sweep 150;
        # its writes land at measure sweeps 40 and 110), so the second resume
        # starts from a file written by a resumed writer and crosses the restart
        # of temperature 3.
        init = _rand_config(MersenneTwister(8112), H)
        poison, benign = _poison_pair(101)
        obs = [cmc_observables(H, con); benign]
        kw = (; base..., carryover = false, init = init, seed = 8113,
              observables = obs)
        a = run_cmc(H, con; kw...)
        pb = joinpath(dir, "cmc_therm_b.jld2")
        b = run_cmc(H, con; kw..., checkpoint = pb, checkpoint_interval = 70)
        _cmcck_same(a, b)
        path = joinpath(dir, "cmc_therm.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kw...,
                                         observables = [cmc_observables(H, con); poison],
                                         checkpoint = path, checkpoint_interval = 70))
        @test _cmcck_progress(path) == (2, "therm", 70)
        # the first resume from mid-thermalization, run to completion elsewhere
        pc = joinpath(dir, "cmc_therm_c.jld2")
        cp(path, pc; force = true)
        c = resume(pc, H; observables = obs)
        _cmcck_same(a, c)
        @test _cmcck_chain(pc) == _cmcck_chain(pb)
        # the double resume
        poison2, _ = _poison_pair(50)
        _cmcck_interrupted(() -> resume(path, H;
                                        observables = [cmc_observables(H, con);
                                                       poison2]))
        @test _cmcck_progress(path) == (2, "measure", 110)
        d = resume(path, H; observables = obs)
        _cmcck_same(a, d)
        @test _cmcck_chain(path) == _cmcck_chain(pb)
    end

    @testset "G13: boundary-only cadence, carryover = false" begin
        # interval 0: the file's last write is the end-of-temperature-2 boundary
        # (the poison fires at temperature 3's 50th measurement), so resume replays
        # temperature 3 in full, starting with the restart to the stored start
        init = _rand_config(MersenneTwister(8104), H)
        poison, benign = _poison_pair(250)
        obs = [cmc_observables(H, con); benign]
        kw = (; base..., carryover = false, init = init, seed = 8105,
              observables = obs)
        a = run_cmc(H, con; kw...)
        path = joinpath(dir, "cmc_boundary.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kw...,
                                         observables = [cmc_observables(H, con); poison],
                                         checkpoint = path))
        @test _cmcck_progress(path) == (3, "therm", 0)
        c = resume(path, H; observables = obs)
        _cmcck_same(a, c)
    end

    @testset "G13: field model with signed weights, mid-measure" begin
        # FZ on dims (2,1,1): the Zeeman terms and the :torque_zeeman observable,
        # weights (1, −2) (the second sublattice starts antiparallel to n̂).
        # 100 measurements per temperature, the poison at temperature 2's 70th
        # (measure sweep 140); writes every 70 sweeps: temperature 2 writes at
        # therm sweep 70 and measure sweeps 60 and 130.
        Hz = MC.TiledHamiltonian(2, _f2g_terms(); dims = (2, 1, 1),
                                 magmoms = _fz_magmoms(), field = _fz_field())
        conz = MagnetizationConstraint(n̂; weights = [1.0, -2.0])
        poison, benign = _poison_pair(170)
        obs = [cmc_observables(Hz, conz); benign]
        @test :torque_zeeman in [o.name for o in obs]
        kw = (; kT = [0.08, 0.04], sweeps_therm = 80, sweeps_measure = 200,
              measure_interval = 2, renorm_interval = 30, nbins = 4, seed = 8106,
              observables = obs)
        a = run_cmc(Hz, conz; kw...)
        path = joinpath(dir, "cmc_field.jld2")
        _cmcck_interrupted(() -> run_cmc(Hz, conz; kw...,
                                         observables = [cmc_observables(Hz, conz);
                                                        poison],
                                         checkpoint = path, checkpoint_interval = 70))
        @test _cmcck_progress(path) == (2, "measure", 130)
        c = resume(path, Hz; observables = obs)
        _cmcck_same(a, c)
        @test c.constraint.weights == [1.0, -2.0]
    end

    @testset "G13: resume with default arguments; completed files" begin
        # Defaults throughout, so the file stores the DEFAULT observable names and
        # `resume(path, H)` must resolve cmc_observables(H, stored constraint) and
        # cmc_evaluables(). The poisoned evaluable dies at temperature 1's
        # finalization; the file holds measure sweep 290 of temperature 1 (writes
        # at measure sweeps 30, 160, 290).
        kw = (; base..., seed = 8107)
        a = run_cmc(H, con; kw...)
        path = joinpath(dir, "cmc_default.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kw...,
                                         evaluables = [cmc_evaluables();
                                                       _cmcck_poison_ev()],
                                         checkpoint = path, checkpoint_interval = 130))
        @test _cmcck_progress(path) == (1, "measure", 290)
        c = resume(path, H)
        _cmcck_same(a, c)
        @test sort(collect(keys(c.points[1].stats))) ==
              [:M_par, :constraint_residual, :energy, :energy2, :specific_heat, :torque]
        # again on a direction whose normalization is not idempotent: the default
        # observables must measure along the stored n̂ bitwise (n̂(0.6, 0.35) above
        # is idempotent, so it cannot tell)
        conr = _cmcck_nonidempotent_con(8114, _fmix_weights())
        @test MagnetizationConstraint(conr.direction).direction != conr.direction
        kwr = (; kT = [0.1, 0.07], sweeps_therm = 50, sweeps_measure = 150,
               measure_interval = 3, nbins = 4, seed = 8115)
        ar = run_cmc(H, conr; kwr...)
        pr = joinpath(dir, "cmc_default_r.jld2")
        _cmcck_interrupted(() -> run_cmc(H, conr; kwr...,
                                         evaluables = [cmc_evaluables();
                                                       _cmcck_poison_ev()],
                                         checkpoint = pr, checkpoint_interval = 45))
        # writes at therm sweep 45 and measure sweeps 40, 85, 130
        @test _cmcck_progress(pr) == (1, "measure", 130)
        _cmcck_same(ar, resume(pr, H))
        # a completed file: the early return rebuilds the constrained result
        done = joinpath(dir, "cmc_done.jld2")
        b = run_cmc(H, con; kw..., checkpoint = done, checkpoint_interval = 0)
        @test _cmcck_progress(done) == (4, "therm", 0)
        d = resume(done, H)
        _cmcck_same(a, d)
        _cmcck_same(b, resume(done, H; checkpoint = nothing))
    end

    @testset "RT: the stored constraint round-trips" begin
        # a −0.0 moment (admitted by the field door) resolves to a −0.0 weight under
        # :auto; the file stores it folded, so the stored record agrees in ==,
        # isequal, and hash with a constraint built from the same numbers
        H3 = MC.TiledHamiltonian(3, _f3_terms(); magmoms = [2.0, -0.0, 1.0],
                                 field = (0.0, 0.0, 1.0))
        p3 = joinpath(dir, "cmc_rt.jld2")
        r3 = run_cmc(H3, MagnetizationConstraint(n̂); kT = 0.1, sweeps_therm = 0,
                     sweeps_measure = 2, nbins = 2, seed = 8108, checkpoint = p3)
        ref3 = MagnetizationConstraint(n̂; weights = [2.0, 0.0, 1.0])
        got = MC.jldopen(f -> MC._read_constraint(f, H3), p3, "r")
        @test got == ref3
        @test isequal(got, ref3)
        @test hash(got) == hash(ref3)
        @test isequal(got, r3.constraint)
        @test !signbit(MC.jldopen(f -> f["constraint/weights"], p3, "r")[2])
        @test MC.jldopen(f -> f["constraint/partner_rule"], p3, "r") == "uniform"
        @test MC.jldopen(f -> (f["plan/or_per_metropolis"], f["plan/sweep_tasks"]),
                         p3, "r") == (0, 1)
        @test isequal(resume(p3, H3).constraint, ref3)
        # the direction round-trips bitwise — not normalized again, which moves
        # about a third of normalized vectors by an ulp (non-vacuity asserted)
        Hs = MC.TiledHamiltonian(2, _f2g_terms())
        rng = MersenneTwister(8109)
        moved = 0
        for k = 1:12
            c = MagnetizationConstraint(randn(rng, 3); weights = [1.0, 0.5])
            moved += MagnetizationConstraint(c.direction).direction != c.direction
            p = joinpath(dir, "cmc_dir_$k.jld2")
            run_cmc(Hs, c; kT = 0.1, sweeps_therm = 0, sweeps_measure = 2, nbins = 2,
                    seed = 1, checkpoint = p)
            got = MC.jldopen(f -> MC._read_constraint(f, Hs), p, "r")
            @test got.direction === c.direction
            @test isequal(got, c)
            @test hash(got) == hash(c)
            @test resume(p, Hs).constraint.direction === c.direction
        end
        @test moved >= 1
    end

    @testset "doors" begin
        path = joinpath(dir, "cmc_guard.jld2")
        run_cmc(H, con; kT = 0.1, sweeps_therm = 20, sweeps_measure = 40, nbins = 2,
                seed = 8110, checkpoint = path, checkpoint_interval = 25)
        @test _cmcck_progress(path) == (2, "therm", 0)
        # wrong H: the fingerprint door, before anything else is derived
        H2 = MC.TiledHamiltonian(8, _fmix_terms(); dims = (1, 1, 1))
        @test_throws r"fingerprint does not match" resume(path, H2)
        # a "cmc" file through the GPU method: refused by kind, by name
        gH = MC.GPUTiledHamiltonian(MC.KernelAbstractions.CPU(), H)
        @test_throws r"CPU driver" resume(path, gH)
        # explicitly passed standard lists keep their meaning: a name mismatch with
        # the stored observables (the failure an OLD package version produces with
        # its default arguments), and the CMC observables with the standard
        # evaluables (no :m2 / :absm measured)
        @test_throws r"do not match the checkpoint" resume(
            path, H; observables = standard_observables(H))
        @test_throws r"needs observable" resume(
            path, H; observables = cmc_observables(H, con),
            evaluables = standard_evaluables())
        # corrupted files
        function corrupt(name, key, value)
            p = joinpath(dir, name)
            cp(path, p; force = true)
            MC.jldopen(p, "r+") do f
                delete!(f, key)
                f[key] = value
            end
            return p
        end
        @test_throws r"partner_rule \"nearest\" is not supported" resume(
            corrupt("pr.jld2", "constraint/partner_rule", "nearest"), H)
        @test_throws r"not a unit 3-vector" resume(
            corrupt("dir.jld2", "constraint/direction", [0.0, 0.0, 2.0]), H)
        @test_throws r"per-cell-atom" resume(
            corrupt("w.jld2", "constraint/weights", ones(3)), H)
        @test_throws r"or_per_metropolis = 1" resume(
            corrupt("or.jld2", "plan/or_per_metropolis", 1), H)
        @test_throws r"start/config has 3 sites" resume(
            corrupt("st.jld2", "start/config", repeat([0.0, 0.0, 1.0], 1, 3)), H)
        # the kind is refused by name right after it is read — before the defaults
        # are resolved (which would turn it into an observable-name mismatch)
        @test_throws r"unknown checkpoint kind foo" resume(
            corrupt("kind.jld2", "kind", "foo"), H)
        @test_throws r"gpu_run_pt" resume(corrupt("kind2.jld2", "kind", "gpu_pt"), H)
        # the consistency door: plausible rewrites (unit vectors everywhere) that
        # every door above admits, refused because the stored configurations do
        # not satisfy the stored constraint. `path` is a completed file, so the
        # chain-side refusal is also the early-return path.
        @test_throws r"start/config does not satisfy the stored constraint" resume(
            corrupt("dz.jld2", "constraint/direction", [0.0, 0.0, 1.0]), H)
        @test_throws r"start/config does not satisfy the stored constraint" resume(
            corrupt("sx.jld2", "start/config", repeat([1.0, 0.0, 0.0], 1, 16)), H)
        @test_throws r"chain/config does not satisfy the stored constraint" resume(
            corrupt("cx.jld2", "chain/config", repeat([1.0, 0.0, 0.0], 1, 16)), H)
        # … and a mid-run file (the resumed run would otherwise sample on)
        poison, benign = _poison_pair(30)
        mid = joinpath(dir, "cmc_guard_mid.jld2")
        _cmcck_interrupted(() -> run_cmc(H, con; kT = [0.1, 0.07], sweeps_therm = 20,
                                         sweeps_measure = 40, nbins = 2, seed = 8116,
                                         observables = [cmc_observables(H, con);
                                                        poison],
                                         checkpoint = mid, checkpoint_interval = 25))
        @test _cmcck_progress(mid)[1] == 1 && _cmcck_progress(mid)[2] == "measure"
        midobs = [cmc_observables(H, con); benign]
        q = joinpath(dir, "mid_dz.jld2")
        cp(mid, q; force = true)
        MC.jldopen(q, "r+") do f
            delete!(f, "constraint/direction")
            f["constraint/direction"] = [0.0, 1.0, 0.0]
        end
        @test_throws r"does not satisfy the stored constraint" resume(
            q, H; observables = midobs)
        @test resume(mid, H; observables = midobs, checkpoint = nothing) isa CMCResult
    end

    @testset "kinds: an \"mc\" file resumes with the standard defaults" begin
        # run_mc with the default observables, interrupted at temperature 1's
        # finalization by the poisoned evaluable; `resume(path, H)` must resolve
        # standard_observables / standard_evaluables for this kind
        Hb = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        kw = (; kT = [0.5, 0.3], sweeps_therm = 100, sweeps_measure = 300,
              measure_interval = 3, renorm_interval = 70, nbins = 4, seed = 8111)
        a = run_mc(Hb; kw...)
        path = joinpath(dir, "mc_default.jld2")
        _cmcck_interrupted(() -> run_mc(Hb; kw...,
                                        evaluables = [standard_evaluables();
                                                      _cmcck_poison_ev()],
                                        checkpoint = path, checkpoint_interval = 130))
        @test _cmcck_progress(path) == (1, "measure", 290)
        c = resume(path, Hb)
        @test c isa MCResult
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
        @test haskey(c.points[1].stats, :susceptibility) && haskey(c.points[1].stats, :m)
    end
end

# Checkpoint / resume: bit-identity gates (==, never ≈) for MC and PT, Xoshiro
# round-trips, and the schema guards.

# Everything result-shaped must be bit-equal between two runs.
function _assert_same_result(a, b)
    @test length(a.points) == length(b.points)
    for (pa, pb) in zip(a.points, b.points)
        @test pa.kT == pb.kT
        @test sort(collect(keys(pa.stats))) == sort(collect(keys(pb.stats)))
        for k in keys(pa.stats)
            @test pa.stats[k].mean == pb.stats[k].mean
            @test pa.stats[k].err == pb.stats[k].err
            @test isequal(pa.stats[k].tau_int, pb.stats[k].tau_int)
            @test pa.stats[k].count == pb.stats[k].count
        end
        @test pa.acceptance_metropolis == pb.acceptance_metropolis
        @test isequal(pa.acceptance_or, pb.acceptance_or)
        @test pa.final_step == pb.final_step
        @test pa.max_drift == pb.max_drift
    end
end

# The interrupted-writer pattern: a completed mc file ALWAYS ends at the completed
# marker — `_mc_loop!` writes an unconditional end-of-temperature boundary
# checkpoint — so a resume built on a finished run returns the stored result
# without re-running a sweep, and a resume-equals-uninterrupted assertion on it
# compares the file with itself. Mid-run continuation teeth therefore REQUIRE a
# writer that actually stops: the poison observable throws at the n-th
# measurement, leaving the file at the last periodic (or boundary) write, and
# `resume` completes the run from there. The benign twin re-supplies the name at
# resume (the checkpoint validates observable names/ncomps, never functions) and
# returns the same 0.0 the poison did before it fired, so the continued
# statistics stay bit-comparable. The PT gates below need none of this:
# `_pt_run!` has no end-of-run write, so their files land mid-measure by interval
# arithmetic (asserted where it matters).
# [Backported from SLCEMonteCarlo.jl 9f722e9.]
function _poison_pair(n::Int)
    cnt = Ref(0)
    return Observable(:poison, 1,
                      (cfg, E, H) -> (cnt[] += 1) >= n ? error("poison interrupt") :
                                     0.0),
           Observable(:poison, 1, (cfg, E, H) -> 0.0)
end

function _interrupted(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("poison", err.msg)
    return nothing
end

_mc_progress(path) = MC.jldopen(path, "r") do f
    (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
end

@testset "checkpoint / resume" begin
    dir = mktempdir()
    H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))

    @testset "model_fingerprint facade" begin
        # the public facade IS the internal fingerprint (dependent-package tier)
        @test MC.model_fingerprint(H) === MC._fingerprint(H)
        @test MC.model_fingerprint(H) ===
              MC.model_fingerprint(TiledHamiltonian(_biquadratic_model(0);
                                                    dims = (2, 1, 1)))
        @test MC.model_fingerprint(H) !==
              MC.model_fingerprint(TiledHamiltonian(_biquadratic_model(0);
                                                    dims = (2, 2, 1)))
    end

    @testset "Xoshiro word round-trip" begin
        rng = Xoshiro(1234)
        rand(rng, 17)
        words = MC._rng_words(rng)
        @test length(words) == fieldcount(Xoshiro)
        rng2 = MC._rng_from_words(words)
        @test all(rand(rng, UInt64) == rand(rng2, UInt64) for _ = 1:100)
        @test_throws ErrorException MC._rng_from_words(UInt64[1, 2])
    end

    @testset "MC: checkpointing does not perturb, resume is bit-identical" begin
        poison, benign = _poison_pair(300)   # temp-2 measurement 100 of 200
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3], sweeps_therm = 200, sweeps_measure = 400,
              measure_interval = 2, nbins = 8, renorm_interval = 100, seed = 42,
              observables = obs)
        path = joinpath(dir, "mc.jld2")
        a = run_mc(H; kw...)                                # no checkpointing
        b = run_mc(H; kw..., checkpoint = path, checkpoint_interval = 150)
        _assert_same_result(a, b)                           # writing consumes no RNG
        @test a.final_config == b.final_config
        # mid-run continuation, via the interrupted writer (see `_poison_pair`):
        # the poison run dies at temp-2 measure sweep 200, its file holds the
        # periodic write at measure sweep 100, and resume must complete THAT run
        # into `a`, bit for bit
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path, checkpoint_interval = 150))
        ti, phase, sweep = _mc_progress(path)
        @test ti == 2 && phase == "measure" && 0 < sweep < 400   # genuinely mid-run
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
        @test c isa MCResult
    end

    @testset "MC: resume from a thermalization-phase checkpoint" begin
        # the poison fires at temp-2's FIRST measurement, so the file's last
        # periodic write (global sweep 520 = temp-2 therm sweep 120) sits inside
        # thermalization — the phase this gate exists to resume from
        poison, benign = _poison_pair(101)
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3], sweeps_therm = 300, sweeps_measure = 100,
              measure_interval = 1, nbins = 8, seed = 7, observables = obs)
        path = joinpath(dir, "mc_therm.jld2")
        a = run_mc(H; kw...)
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path, checkpoint_interval = 260))
        ti, phase, sweep = _mc_progress(path)
        @test ti == 2 && phase == "therm" && 0 < sweep < 300
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "MC: boundary-only checkpoints (interval 0) and carryover=false" begin
        # poison at temp-3 measurement 50: with interval 0 the file's last write
        # is the end-of-temp-2 boundary, so resume replays temp 3 IN FULL —
        # thermalization, measure phase, and the carryover=false restart (whose
        # random redraw must come out of the restored RNG bit-identically)
        poison, benign = _poison_pair(250)
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 100, sweeps_measure = 100,
              nbins = 4, carryover = false, seed = 3, observables = obs)
        path = joinpath(dir, "mc_boundary.jld2")
        a = run_mc(H; kw...)
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path))       # interval 0
        ti, phase, sweep = _mc_progress(path)
        @test ti == 3 && phase == "therm" && sweep == 0   # end-of-temp-2 boundary
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "PT: resume is bit-identical" begin
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
              exchange_interval = 7, nbins = 8, seed = 11)
        path = joinpath(dir, "pt.jld2")
        a = run_pt(H; kw...)
        b = run_pt(H; kw..., checkpoint = path, checkpoint_interval = 120)
        _assert_same_result(a, b)
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        c = resume(path, H)
        @test c isa PTResult
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
        @test a.swap_acceptance == c.swap_acceptance
    end

    @testset "PT: resume from the phase-boundary checkpoint (interval 0)" begin
        kw = (; kT = [0.5, 0.2], sweeps_therm = 100, sweeps_measure = 200,
              exchange_interval = 9, nbins = 4, seed = 13)
        path = joinpath(dir, "pt_boundary.jld2")
        a = run_pt(H; kw...)
        run_pt(H; kw..., checkpoint = path)
        c = resume(path, H)
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
    end

    # GPU PT ("gpu_pt" kind) — device sibling of the PT gates above, on the KA
    # CPU backend. As `_pt_run!`, `_gpu_pt_run!` has NO end-of-run write, so a
    # completed run's file genuinely sits mid-measure; the non-vacuity of the
    # resume gate is the interval arithmetic, asserted below.
    @testset "GPU PT: checkpointing does not perturb, resume is bit-identical" begin
        gH = MC.GPUTiledHamiltonian(MC.KernelAbstractions.CPU(), H)
        # Deliberately NON-default knobs, so the resumed tail exercises every
        # restored quantity: workgroupsize 32 (a resume must use the STORED ws
        # — it enters the ΔE fold, so silently falling back to 128 differs in
        # bits), measure_interval 3 (the tail crosses measurement offsets ⇒
        # `phase_sweeps := done` is load-bearing), renorm_interval 40 (the tail
        # renormalizes ⇒ gates the rebuilt-rows re-anchor and the stored
        # max_drift).
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
              exchange_interval = 7, measure_interval = 3, renorm_interval = 40,
              nbins = 8, seed = 17, workgroupsize = 32)
        path = joinpath(dir, "gpu_pt.jld2")
        a = gpu_run_pt(gH; kw...)                          # no checkpointing
        b = gpu_run_pt(gH; kw..., checkpoint = path, checkpoint_interval = 120)
        _assert_same_result(a, b)                          # writing perturbs nothing
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        # the file's last write is mid-measure (interval arithmetic: writes at
        # done 126 and 252 of 300) — the resumed tail is genuinely re-run
        phase, done = MC.jldopen(path, "r") do f
            (f["progress/phase"], f["progress/done"])
        end
        @test phase == "measure" && 0 < done < 300
        c = resume(path, gH)
        @test c isa PTResult
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
        @test a.swap_acceptance == c.swap_acceptance

        # kind-direction guards: each driver refuses the other's file, by name
        err = try
            resume(path, H)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("gpu_run_pt", err.msg)
        ppath = joinpath(dir, "pt_for_gpu_guard.jld2")
        run_pt(H; kT = [0.5, 0.2], sweeps_therm = 30, sweeps_measure = 60,
               nbins = 4, seed = 1, checkpoint = ppath, checkpoint_interval = 20)
        err = try
            resume(ppath, gH)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("CPU driver", err.msg)
        # fingerprint mismatch through the GPU method
        gH2 = MC.GPUTiledHamiltonian(MC.KernelAbstractions.CPU(),
                                     TiledHamiltonian(_biquadratic_model(0);
                                                      dims = (2, 2, 1)))
        @test_throws ErrorException resume(path, gH2)
    end

    @testset "GPU PT: resume from a thermalization-phase checkpoint" begin
        # The walltime-kill-during-thermalization path. A therm-phase file is
        # unreachable through the public API (the unconditional therm→measure
        # boundary write always overwrites it), so this gate is white-box: it
        # rebuilds the run's front half exactly as `gpu_run_pt` does (the G8
        # seed-derivation contract, already pinned by test_gpu_pt.jl's
        # composition gate) and drives ONLY the therm phase with a
        # checkpointer, leaving the file mid-therm; `resume` must then complete
        # the run into the uninterrupted result, bit for bit — restoring the
        # mid-adapt-window counters, the adapted step, and the therm parity.
        gH = MC.GPUTiledHamiltonian(MC.KernelAbstractions.CPU(), H)
        kts = [0.5, 0.3]
        seed = 23
        kw = (; kT = kts, sweeps_therm = 150, sweeps_measure = 100,
              exchange_interval = 7, nbins = 4, seed = seed)
        a = gpu_run_pt(gH; kw...)
        # (ck_interval, expected last therm write): 40 ⇒ mid-therm (done 126,
        # counters 26 sweeps into the adapt window); 10 ⇒ the phase-end edge
        # (done == 150: the therm loop must skip and the boundary must replay)
        for (ck_interval, expect) in ((40, 126), (10, 150))
            plan = MC.UpdatePlan(MC.resolve_kt(nothing, kts);
                                 sweeps_therm = 150, sweeps_measure = 100,
                                 measure_interval = 1, or_per_metropolis = 0,
                                 step = 0.6, adapt_target = 0.5,
                                 adapt_interval = 50, renorm_interval = 1_000,
                                 nbins = 4, carryover = false, sweep_tasks = 1,
                                 seed = seed)
            path = joinpath(dir, "gpu_pt_therm_$(ck_interval).jld2")
            ck = MC._make_checkpointer(path, ck_interval, H, plan,
                                       standard_observables(H), "gpu_pt", 7)
            master = Xoshiro(plan.seed)
            lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                                 rand(master, UInt64), rand(master, UInt64))
                         for _ in kts]
            exchange_rng = Xoshiro(rand(master, UInt64), rand(master, UInt64),
                                   rand(master, UInt64), rand(master, UInt64))
            dev_seeds = [rand(master, UInt64) for _ in kts]
            lanes = [begin
                         st = MC.ChainState(H,
                                            MC._initial_config(H, nothing,
                                                               lane_rngs[r]),
                                            lane_rngs[r], plan.step0)
                         MC._GPUPTLane(MC.GPUChainState(gH, st;
                                                        seed = dev_seeds[r]),
                                       st, kts[r], 1.0 / kts[r],
                                       MC.ObsAccumulator[], 0)
                     end
                     for r in eachindex(kts)]
            MC._gpu_pt_phase!(lanes, gH, plan, plan.sweeps_therm, 7, false,
                              exchange_rng, zeros(Int, 1), zeros(Int, 1), 0,
                              128; ck = ck)
            phase, done = MC.jldopen(path, "r") do f
                (f["progress/phase"], f["progress/done"])
            end
            @test phase == "therm" && done == expect
            c = resume(path, gH)
            _assert_same_result(a, c)
            @test a.final_configs == c.final_configs
            @test a.swap_acceptance == c.swap_acceptance
        end
    end

    @testset "GPU PT: resume from the phase-boundary checkpoint (interval 0)" begin
        # interval 0 ⇒ the only write is the unconditional thermalization →
        # measurement boundary, so resume replays the WHOLE measurement phase
        # (fresh accumulators, restored exchange RNG and parity)
        gH = MC.GPUTiledHamiltonian(MC.KernelAbstractions.CPU(), H)
        kw = (; kT = [0.5, 0.2], sweeps_therm = 100, sweeps_measure = 200,
              exchange_interval = 9, nbins = 4, seed = 19)
        path = joinpath(dir, "gpu_pt_boundary.jld2")
        a = gpu_run_pt(gH; kw...)
        gpu_run_pt(gH; kw..., checkpoint = path)
        phase, done = MC.jldopen(path, "r") do f
            (f["progress/phase"], f["progress/done"])
        end
        @test phase == "measure" && done == 0      # the boundary write itself
        c = resume(path, gH)
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
        @test a.swap_acceptance == c.swap_acceptance
    end

    @testset "resume of a completed run is idempotent" begin
        # a job-array retry loop may call resume on a checkpoint whose run already
        # finished — it must return the finished result unchanged (MC and PT)
        path = joinpath(dir, "done_mc.jld2")
        a = run_mc(H; kT = [0.5, 0.3], sweeps_therm = 100, sweeps_measure = 200,
                   nbins = 8, seed = 9, checkpoint = path,
                   checkpoint_interval = 50)
        b = resume(path, H)
        @test b.final_config == a.final_config
        @test all(b.points[i].stats[:energy].mean == a.points[i].stats[:energy].mean
                  for i in eachindex(a.points))
        pp = joinpath(dir, "done_pt.jld2")
        c = run_pt(H; kT = [0.5, 0.3, 0.2], sweeps_therm = 100,
                   sweeps_measure = 200, nbins = 8, seed = 9, checkpoint = pp,
                   checkpoint_interval = 50)
        d = resume(pp, H)
        @test d.final_configs == c.final_configs
        @test d.swap_acceptance == c.swap_acceptance
    end

    @testset "schema and mismatch guards" begin
        path = joinpath(dir, "guard.jld2")
        run_mc(H; kT = 0.5, sweeps_therm = 50, sweeps_measure = 60, nbins = 4,
               seed = 1, checkpoint = path, checkpoint_interval = 40)
        # fingerprint mismatch: different dims
        H2 = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        @test_throws ErrorException resume(path, H2)
        # observable mismatch
        @test_throws ErrorException resume(path, H; observables = [
            Observable(:energy, 1, (c, E, h) -> E)])
        # missing file
        @test_throws ArgumentError resume(joinpath(dir, "nope.jld2"), H)
        # negative interval guard
        @test_throws ArgumentError run_mc(H; kT = 0.5, checkpoint = path,
                                          checkpoint_interval = -1)
        # a corrupted stored configuration is refused by the non-projecting
        # door (validated without projecting — restore must stay bit-exact)
        # [Backported from SLCEMonteCarlo.jl 3f71644.]
        cpath = joinpath(dir, "corrupt.jld2")
        cp(path, cpath)
        MC.jldopen(cpath, "r+") do f
            m = f["chain/config"]
            delete!(f, "chain/config")
            m[:, 1] .*= 2.5
            f["chain/config"] = m
        end
        @test_throws ArgumentError resume(cpath, H)
    end
end

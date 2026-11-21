# Constrained MC: direction lanes — `run_cmc(H, constraints; ntasks, …)`
# (docs/specs/constrained-mc.md D4).
#
# G14 — the lane vector is bitwise independent of `ntasks` (1; 2, 4 and 20 — three
#       different partitions of the six lanes into contiguous blocks: 3+3, 2+2+2,
#       and one lane per task; results, final configurations, constraints, seeds);
#       lane k is bitwise the single-chain run with lane seed k, the seeds derived
#       HERE from the documented rule (`rand(Xoshiro(UInt64(seed)), UInt64)` drawn in
#       lane order), not by the package's helper; every lane measures along its own
#       constraint (`:constraint_residual` at rounding level and `M∥ > 0` in every
#       lane, the final configuration's weighted magnetization — summed here —
#       parallel to that lane's n̂ with that lane's weights); duplicate constraints
#       are independent replicas. Non-vacuity of the ntasks gate: the directions
#       differ pairwise, so a permutation of results or seeds is visible.
# CK  — per-lane checkpoint files `<stem>.lane<k>.jld2` (names written here from the
#       stem rule: with and without the ".jld2" suffix, an upper-case ".JLD2", a
#       doubled suffix; no stray names), kind "cmc", the lane's seed and
#       constraint; `resume(file_k, H)` of a completed lane returns the lane's
#       result; a lane interrupted mid-measure (the `_poison_pair` pattern, position
#       asserted) resumes ≡ its uninterrupted lane, while the other lanes of the
#       same call complete — also the lane sharing its task.
# AGG — the failure contract: failing lanes stop no other lane; the failures arrive
#       as ONE CompositeException whose entries are all CapturedExceptions, in lane
#       order, for ntasks 1, 4, 20 (at 4 and 20 lane 1 is held until lane 6's
#       failure has been recorded, so the failures complete in reverse lane order
#       and a completion-order list is visible; at 1 a reversed list is); each
#       failure is logged at once (`@error`, naming the lane).
# INT — interrupts: an InterruptException raised in a lane (ntasks 1 and 4) or
#       delivered to the waiting caller (ntasks 4) makes `run_cmc` throw a BARE
#       InterruptException; no lane starts after it (the running lanes are held
#       until the stop flag is up, so a block that went on would visibly start its
#       next lane); the running lanes finish and every task has finished when
#       `run_cmc` throws (their files are complete at that moment, though they could
#       not finish before the interrupt). All orderings are explicit handshakes
#       (atomic flags set by the lanes or by the package's own log records), never
#       sleeps.
# Doors — every argument error is thrown before any lane runs (no checkpoint file
#       appears): an empty constraint vector, `ntasks < 1`, a negative seed, both
#       or neither of temperature / kT, a plan keyword out of range, an unknown
#       keyword (MethodError), observables given as a vector or a tuple, a function
#       that does not return a Vector{Observable}, `or_per_metropolis` /
#       `sweep_tasks`, a bad evaluable, a constraint that does not resolve against
#       H and an `init` refused for one lane's weights (both whatever the observables
#       function), a negative `checkpoint_interval`, a checkpoint path naming a
#       directory or leaving an empty stem. A one-shot temperature iterator is
#       consumed once (every lane gets the full ladder).

using Logging: Logging

# A thread-safe logger for the lane runs (the lane failures are logged from the lane
# tasks). It keeps the Error-level messages and, as a handshake for the tests, sets
# `flags[pattern]` when a Warn-or-above message containing `pattern` arrives — the
# package logs a lane failure after recording it, and an interrupt after raising
# its stop flag, so a flag set here means that step has happened.
struct _LnLogs <: Logging.AbstractLogger
    lock::ReentrantLock
    errors::Vector{String}
    flags::Vector{Pair{String,Threads.Atomic{Bool}}}
end
_LnLogs(flags) = _LnLogs(ReentrantLock(), String[], flags)
Logging.min_enabled_level(::_LnLogs) = Logging.Warn
Logging.shouldlog(::_LnLogs, args...) = true
Logging.catch_exceptions(::_LnLogs) = false
function Logging.handle_message(l::_LnLogs, level, message, args...; kwargs...)
    msg = string(message)
    level >= Logging.Error && lock(() -> push!(l.errors, msg), l.lock)
    for (pattern, flag) in l.flags
        occursin(pattern, msg) && (flag[] = true)
    end
    return nothing
end

# Run `f` under `_LnLogs(flags)`; returns (the exception it threw or `nothing`, the
# Error-level messages).
function _ln_run(f, flags = Pair{String,Threads.Atomic{Bool}}[])
    logs = _LnLogs(flags)
    err = Logging.with_logger(logs) do
        try
            f()
            nothing
        catch e
            e
        end
    end
    return err, logs.errors
end

# Block the calling task until `cond()` holds — a handshake, not a timing
# assumption: the polling interval only sets how soon it notices. The bound turns a
# broken handshake (a mutant that never signals) into a failure instead of a hang.
function _ln_await(cond; bound = 30.0)
    t0 = time()
    while !cond()
        time() - t0 > bound && error("lane handshake timed out")
        sleep(0.001)
    end
    return nothing
end

# The lane seeds of the documented rule, written independently of the package.
function _ln_seeds(seed, n)
    master = Xoshiro(UInt64(seed))
    return [rand(master, UInt64) for _ = 1:n]
end

# Bitwise equality of two constrained results.
function _ln_same(a, b)
    @test a isa CMCResult && b isa CMCResult
    _assert_same_result(a, b)
    @test a.final_config == b.final_config
    @test isequal(a.constraint, b.constraint)
    @test a.constraint.direction === b.constraint.direction
    @test a.seed == b.seed
    return nothing
end

# The lane failures of a lane run: the exceptions inside the CompositeException of
# CapturedExceptions it throws (anything else is a failed assertion).
function _ln_failures(e)
    @test e isa CompositeException
    e isa CompositeException || return Any[e]
    @test all(x -> x isa CapturedException, e.exceptions)
    return Any[x isa CapturedException ? x.ex : x for x in e.exceptions]
end

_ln_progress(path) = MC.jldopen(path, "r") do f
    (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
end

@testset "CMC direction lanes" begin
    dir = mktempdir()
    H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))   # 16 sites
    w = _fmix_weights()
    # six lanes: five directions with the Fmix weights (lane 5 duplicates lane 2)
    # and one with uniform weights (the inactive site 8 stays out of 𝒞 regardless)
    dirs = [(0.6, 0.35), (1.2, -0.8), (0.2, 2.5), (2.0, 1.0)]
    cons = [MagnetizationConstraint(_cmcref_dir(θ, φ); weights = w) for (θ, φ) in dirs]
    cons = [cons; MagnetizationConstraint(_cmcref_dir(1.2, -0.8); weights = w);
            MagnetizationConstraint(_cmcref_dir(1.5, 0.4); weights = :uniform)]
    lane_w = [fill(w, 5); [ones(8)]]
    @test cons[5] == cons[2]
    K = length(cons)
    base = (; kT = [0.1, 0.05], sweeps_therm = 60, sweeps_measure = 120,
            measure_interval = 2, renorm_interval = 70, nbins = 4)
    seed = 0x7a11_0007
    seeds = _ln_seeds(seed, K)
    @test allunique(seeds)

    r1 = run_cmc(H, cons; base..., seed = seed, ntasks = 1)
    @test r1 isa Vector{CMCResult} && length(r1) == K

    @testset "G14: bitwise independent of ntasks" begin
        for nt in (2, 4, 20)
            r = run_cmc(H, cons; base..., seed = seed, ntasks = nt)
            @test length(r) == K
            for k = 1:K
                _ln_same(r1[k], r[k])
            end
        end
    end

    @testset "G14: lane k ≡ the single chain with lane seed k" begin
        r4 = run_cmc(H, cons; base..., seed = seed, ntasks = 4)
        for k = 1:K
            @test r4[k].seed == seeds[k]
            single = run_cmc(H, cons[k]; base..., seed = seeds[k])
            _ln_same(single, r4[k])
        end
        # the duplicate lane is an independent replica
        @test r1[5].final_config != r1[2].final_config
        @test r1[5].points[1].stats[:energy].mean != r1[2].points[1].stats[:energy].mean
    end

    @testset "G14: every lane measures along its own constraint" begin
        act = H.site_active
        for k = 1:K
            n̂ = cons[k].direction
            @test r1[k].constraint.direction === n̂
            for p in r1[k].points
                @test p.stats[:constraint_residual].mean[1] < 1e-13
                @test p.stats[:M_par].mean[1] > 0
            end
            # the final configuration's weighted magnetization, summed here
            wsite = [lane_w[k][MC.site_atom(H, s)] for s = 1:H.n_sites]
            M = _cmcref_M(r1[k].final_config, wsite, act)
            Mpar = dot(M, n̂)
            @test Mpar > 0
            @test norm(M - Mpar * n̂) <= 1e-12 * sum(abs, wsite[act])
            # and not along any other lane's direction (non-vacuity of the residual
            # gate against a shared observable vector)
            for j = 1:K
                n̂j = cons[j].direction
                n̂j == n̂ && continue
                @test norm(M - dot(M, n̂j) * n̂j) / abs(dot(M, n̂j)) > 1e-2
            end
        end
    end

    @testset "CK: per-lane files, completed-lane resume" begin
        path = joinpath(dir, "scan.jld2")
        rc = run_cmc(H, cons; base..., seed = seed, ntasks = 4, checkpoint = path)
        for k = 1:K
            _ln_same(r1[k], rc[k])                    # writing consumes no RNG
        end
        files = sort(filter(f -> startswith(f, "scan"), readdir(dir)))
        @test files == sort(["scan.lane$k.jld2" for k = 1:K])
        for k = 1:K
            file = joinpath(dir, "scan.lane$k.jld2")
            MC.jldopen(file, "r") do f
                @test f["kind"] == "cmc"
                @test f["constraint/direction"] == Vector(cons[k].direction)
                @test f["constraint/weights"] == lane_w[k]
            end
            @test _ln_progress(file) == (3, "therm", 0)    # the end-of-run boundary
            _ln_same(r1[k], resume(file, H))
        end
        # a path without the ".jld2" suffix is the stem itself
        sub = mkpath(joinpath(dir, "bare"))
        run_cmc(H, cons[1:2]; base..., seed = seed, ntasks = 2,
                checkpoint = joinpath(sub, "lanes"))
        @test sort(readdir(sub)) == ["lanes.lane1.jld2", "lanes.lane2.jld2"]
        _ln_same(r1[2], resume(joinpath(sub, "lanes.lane2.jld2"), H))
        # the suffix is stripped in any letter case, once
        sub = mkpath(joinpath(dir, "cases"))
        one = (; base..., sweeps_measure = 20, measure_interval = 2, seed = seed)
        run_cmc(H, cons[1:1]; one..., checkpoint = joinpath(sub, "up.JLD2"))
        run_cmc(H, cons[1:1]; one..., checkpoint = joinpath(sub, "two.jld2.jld2"))
        @test sort(readdir(sub)) == ["two.jld2.lane1.jld2", "up.lane1.jld2"]
    end

    @testset "CK: an interrupted lane resumes ≡ uninterrupted" begin
        # Lane 3 alone carries the poison. 60 measurements per temperature; the
        # poison fires at temperature 2's 30th (measure sweep 60). Writes every 50
        # sweeps, the clock reset at every write (and at the temperature boundary):
        # temperature 2 writes at therm sweep 50 and measure sweeps 40 and 90, so
        # the file holds measure sweep 40. The other lanes run to completion.
        _, benign = _poison_pair(90)
        ref = run_cmc(H, cons; base..., seed = seed, ntasks = 4,
                      observables = c -> [cmc_observables(H, c); benign])
        for k = 1:K                       # an extra observable leaves the chains alone
            @test ref[k].final_config == r1[k].final_config
        end
        # the same with every lane serial on the calling task and with blocks of
        # two lanes over three tasks (lane 4 shares lane 3's task): a failing lane
        # stops no other lane, and the failure arrives in the same shape
        for nt in (1, 4)
            poison, benign = _poison_pair(90)
            sub = mkpath(joinpath(dir, "poison$nt"))
            path = joinpath(sub, "run.jld2")
            err, logs = _ln_run(() -> run_cmc(
                H, cons; base..., seed = seed, ntasks = nt, checkpoint = path,
                checkpoint_interval = 50,
                observables = c -> [cmc_observables(H, c);
                                    c == cons[3] ? poison : benign]))
            @test length(logs) == 1 && occursin("lane 3", logs[1])
            fails = _ln_failures(err)
            @test length(fails) == 1
            @test fails[1] isa ErrorException && occursin("poison", fails[1].msg)
            file3 = joinpath(sub, "run.lane3.jld2")
            @test _ln_progress(file3) == (2, "measure", 40)     # genuinely mid-run
            c3 = resume(file3, H; observables = [cmc_observables(H, cons[3]); benign])
            _ln_same(ref[3], c3)
            @test c3.seed == seeds[3]
            for k in (1, 2, 4, 5, 6)                 # the other lanes completed
                file = joinpath(sub, "run.lane$k.jld2")
                @test _ln_progress(file) == (3, "therm", 0)
                _ln_same(ref[k], resume(file, H;
                                        observables = [cmc_observables(H, cons[k]);
                                                       benign]))
            end
        end
    end

    @testset "AGG: two failing lanes, in lane order" begin
        # Lane 1 fails at its LAST measurement (2 × 60), lane 6 at its FIRST. With
        # ntasks ≥ 2 lane 1 is held at its last measurement until lane 6's failure
        # has been RECORDED (the logger sees its @error, written after the record;
        # lane 6 is in another block, and the hold frees the thread), so the
        # failures complete in the order 6, 1 — a completion-order list differs
        # from lane order at ntasks 4 and 20, a reversed one at every ntasks.
        for nt in (1, 4, 20)
            failed6 = Threads.Atomic{Bool}(false)
            obsf = function (c)
                cnt = Ref(0)
                k = c == cons[1] ? 1 : c == cons[6] ? 6 : 0
                at = k == 1 ? 120 : 1
                return [cmc_observables(H, c);
                        Observable(:fail, 1, (cfg, E, Hr) -> begin
                            (k > 0 && (cnt[] += 1) >= at) || return 0.0
                            k == 1 && nt > 1 && _ln_await(() -> failed6[])
                            error("fail lane $k")
                        end)]
            end
            err, logs = _ln_run(["run_cmc: lane 6 " => failed6]) do
                run_cmc(H, cons; base..., seed = seed, ntasks = nt, observables = obsf)
            end
            fails = _ln_failures(err)
            @test [f.msg for f in fails] == ["fail lane 1", "fail lane 6"]
            @test sort(logs) == sort(["run_cmc: lane $k (n̂ = " *
                                      "$(Tuple(cons[k].direction))) failed; the " *
                                      "other lanes go on" for k in (1, 6)])
        end
    end

    @testset "INT: interrupts" begin
        # Handshakes only (no timing assumption): lanes report "started" through an
        # atomic counter at their first measurement, and the running lanes are held
        # there until the logger has seen the package's interrupt warning — logged
        # right AFTER the stop flag is raised. A block that ignored the flag would
        # therefore start its next lane after the interrupt, which the file lists
        # show; the held lanes finish only after the interrupt, so their complete
        # files at the moment run_cmc throws show that the caller waited for them.
        # Lanes 2 and 5 share their constraint, so the lanes are told apart by the
        # observables' build order (the function is called in lane order).
        stopped_msg = "run_cmc: interrupted"
        # (a) raised in lane 2, serial: lane 1 complete, no later lane started
        sub = mkpath(joinpath(dir, "int_serial"))
        stopped = Threads.Atomic{Bool}(false)
        built = Ref(0)
        obsf = function (c)
            k = (built[] += 1)
            return [cmc_observables(H, c);
                    Observable(:intr, 1, (cfg, E, Hr) -> k == 2 ?
                                                         throw(InterruptException()) :
                                                         0.0)]
        end
        err, logs = _ln_run([stopped_msg => stopped]) do
            run_cmc(H, cons; base..., seed = seed, ntasks = 1, observables = obsf,
                    checkpoint = joinpath(sub, "run.jld2"))
        end
        @test typeof(err) === InterruptException
        @test isempty(logs) && stopped[]
        @test readdir(sub) == ["run.lane1.jld2"]
        @test _ln_progress(joinpath(sub, "run.lane1.jld2")) == (3, "therm", 0)

        # (b) raised in lane 1, ntasks = 4 (blocks 1-2, 3-4, 5-6): lane 1 waits
        # until lanes 3 and 5 have started, then raises; lanes 3 and 5 are held
        # until the stop flag is up. Lanes 2, 4, 6 never start.
        sub = mkpath(joinpath(dir, "int_lane"))
        stopped = Threads.Atomic{Bool}(false)
        started = Threads.Atomic{Int}(0)
        built = Ref(0)
        obsf = function (c)
            k = (built[] += 1)
            first_call = Ref(true)
            return [cmc_observables(H, c);
                    Observable(:intr, 1, (cfg, E, Hr) -> begin
                        first_call[] || return 0.0
                        first_call[] = false
                        if k == 1
                            _ln_await(() -> started[] == 2)
                            throw(InterruptException())
                        end
                        Threads.atomic_add!(started, 1)
                        _ln_await(() -> stopped[])
                        0.0
                    end)]
        end
        err, logs = _ln_run([stopped_msg => stopped]) do
            run_cmc(H, cons; base..., seed = seed, ntasks = 4, observables = obsf,
                    checkpoint = joinpath(sub, "run.jld2"))
        end
        @test typeof(err) === InterruptException
        @test isempty(logs) && started[] == 2
        @test sort(readdir(sub)) == ["run.lane3.jld2", "run.lane5.jld2"]
        for k in (3, 5)                    # complete at the moment run_cmc threw
            @test _ln_progress(joinpath(sub, "run.lane$k.jld2")) == (3, "therm", 0)
        end

        # (c) delivered to the waiting caller, ntasks = 4: lanes 1, 3, 5 report
        # started and are held until the stop flag is up; a deliverer task waits
        # until all three have started (so the caller has spawned every task) AND
        # the caller sits in a wait queue (`Task.queue`, an internal field: the
        # caller cannot leave that wait while every lane is held), then schedules
        # the InterruptException onto it. Lanes 2, 4, 6 never start.
        sub = mkpath(joinpath(dir, "int_caller"))
        stopped = Threads.Atomic{Bool}(false)
        started = Threads.Atomic{Int}(0)
        obsf = function (c)
            first_call = Ref(true)
            return [cmc_observables(H, c);
                    Observable(:intr, 1, (cfg, E, Hr) -> begin
                        first_call[] || return 0.0
                        first_call[] = false
                        Threads.atomic_add!(started, 1)
                        _ln_await(() -> stopped[])
                        0.0
                    end)]
        end
        caller = current_task()
        deliverer = Threads.@spawn begin
            _ln_await(() -> started[] == 3)
            _ln_await(() -> caller.queue !== nothing)
            schedule(caller, InterruptException(); error = true)
        end
        err, logs = _ln_run([stopped_msg => stopped]) do
            run_cmc(H, cons; base..., seed = seed, ntasks = 4, observables = obsf,
                    checkpoint = joinpath(sub, "run.jld2"))
        end
        wait(deliverer)
        @test typeof(err) === InterruptException
        @test isempty(logs) && started[] == 3
        @test sort(readdir(sub)) == ["run.lane$k.jld2" for k in (1, 3, 5)]
        for k in (1, 3, 5)
            @test _ln_progress(joinpath(sub, "run.lane$k.jld2")) == (3, "therm", 0)
        end
        # nothing is left running: a later lane run is unaffected
        r = run_cmc(H, cons[1:2]; base..., seed = seed, ntasks = 2)
        @test r[2].seed == seeds[2]
    end

    @testset "Doors: every argument error before any lane runs" begin
        sub = mkpath(joinpath(dir, "doors"))
        ck = joinpath(sub, "d.jld2")
        kw = (; base..., seed = seed, checkpoint = ck)
        @test_throws ArgumentError run_cmc(H, MagnetizationConstraint[]; kw...)
        @test_throws ArgumentError run_cmc(H, cons; kw..., ntasks = 0)
        @test_throws ArgumentError run_cmc(H, cons; base..., seed = -1, checkpoint = ck)
        nokt = (; sweeps_therm = 60, sweeps_measure = 120, seed = seed, checkpoint = ck)
        @test_throws ArgumentError run_cmc(H, cons; nokt...)                  # neither
        @test_throws ArgumentError run_cmc(H, cons; nokt..., kT = 0.1,
                                           temperature = 300.0)               # both
        @test_throws ArgumentError run_cmc(H, cons; kw..., sweeps_measure = 0)
        @test_throws ArgumentError run_cmc(H, cons; kw..., nbins = 1)
        @test_throws MethodError run_cmc(H, cons; kw..., sweeps_therms = 10)  # typo
        err = try
            run_cmc(H, cons; kw..., observables = cmc_observables(H, cons[1]))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("function", err.msg) &&
              occursin(":M_par", err.msg)
        err = try
            run_cmc(H, cons; kw..., observables = (cmc_observables(H, cons[1])...,))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("must be a function", err.msg)
        @test_throws ArgumentError run_cmc(H, cons; kw...,
                                           observables = c -> Any[cmc_observables(H, c)...])
        @test_throws ArgumentError run_cmc(H, cons; kw..., or_per_metropolis = 1)
        @test_throws ArgumentError run_cmc(H, cons; kw..., or_per_metropolis = 0)
        @test_throws ArgumentError run_cmc(H, cons; kw..., sweep_tasks = 2)
        # an evaluable whose input no lane measures: refused before any lane runs
        @test_throws ArgumentError run_cmc(H, cons; kw...,
                                           evaluables = standard_evaluables())
        # a constraint that cannot be resolved against H (8 cell atoms, 3 weights),
        # also when the observables function never resolves it
        bad = [cons[1], MagnetizationConstraint(_cmcref_dir(0.3, 0.1);
                                                weights = [1.0, 2.0, 3.0])]
        @test_throws ArgumentError run_cmc(H, bad; kw...)
        plain = c -> [Observable(:energy, 1, (cfg, E, Hr) -> E)]
        @test_throws ArgumentError run_cmc(H, bad; kw..., observables = plain,
                                           evaluables = Evaluable[])
        # an `init` refused for ONE lane's weights: all spins +x̂ cancel under weights
        # (1, −1) on atoms 1, 2 (two cells each), not under (1, 1)
        initx = [SVector(1.0, 0.0, 0.0) for _ = 1:H.n_sites]
        wok = [1.0, 1.0, 0, 0, 0, 0, 0, 0]
        wcancel = [1.0, -1.0, 0, 0, 0, 0, 0, 0]
        pair = [MagnetizationConstraint(_cmcref_dir(0.6, 0.35); weights = wok),
                MagnetizationConstraint(_cmcref_dir(0.6, 0.35); weights = wcancel)]
        run_cmc(H, pair[1:1]; base..., seed = seed, init = initx)   # lane 1 alone runs
        @test_throws ArgumentError run_cmc(H, pair; kw..., init = initx,
                                           observables = plain,
                                           evaluables = Evaluable[])
        @test_throws ArgumentError run_cmc(H, cons; kw..., checkpoint_interval = -1)
        @test isempty(readdir(sub))                 # no lane ran
        # checkpoint paths that name a directory or leave no stem
        for p in (sub, sub * "/", joinpath(sub, ".jld2"), joinpath(sub, ".JLD2"), "")
            err = try
                run_cmc(H, cons; base..., seed = seed, checkpoint = p)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("checkpoint", err.msg)
        end
        @test isempty(readdir(sub))
        # a one-shot temperature iterator is consumed once: every lane gets the
        # full ladder
        rs = run_cmc(H, cons[1:3]; kT = Iterators.Stateful([0.1, 0.05]),
                     sweeps_therm = 10, sweeps_measure = 10, seed = seed, ntasks = 1)
        @test [length(r.points) for r in rs] == [2, 2, 2]
        @test [p.kT for p in rs[3].points] == [0.1, 0.05]
    end
end

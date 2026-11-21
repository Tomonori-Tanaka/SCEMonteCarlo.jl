# Constrained Monte Carlo (CMC), part 4 of 4: direction lanes — one independent
# single-chain `run_cmc` per constraint, over threads (`docs/specs/constrained-mc.md`
# D4, "Direction lanes").

# The stem of the lane checkpoint files of `path`: `path` without a trailing ".jld2"
# (any letter case), else `path` itself. A path that names a directory, or whose
# stem would be empty or a directory, is refused.
function _cmc_lane_checkpoint_stem(path::AbstractString)::String
    (isempty(path) || endswith(path, '/') || isdir(path)) && throw(ArgumentError(
        "run_cmc: checkpoint = $(repr(path)) names a directory; give a file path " *
        "(lane k writes \"<stem>.lane<k>.jld2\")"))
    stem = endswith(lowercase(path), ".jld2") ? String(chop(path; tail = 5)) :
           String(path)
    (isempty(stem) || endswith(stem, '/')) && throw(ArgumentError(
        "run_cmc: checkpoint = $(repr(path)) leaves an empty file stem; give a file " *
        "path such as \"scan.jld2\" (lane k writes \"<stem>.lane<k>.jld2\")"))
    return stem
end

# The checkpoint file of lane `k`: "<stem>.lane<k>.jld2".
_cmc_lane_checkpoint_path(path::AbstractString, k::Integer)::String =
    string(_cmc_lane_checkpoint_stem(path), ".lane", k, ".jld2")

# The lane seeds of a master `seed`: `rand(Xoshiro(UInt64(seed)), UInt64)` drawn in
# lane order k = 1, 2, …, all of them before any lane runs.
function _cmc_lane_seeds(seed::Integer, nlanes::Integer)::Vector{UInt64}
    master = Xoshiro(UInt64(seed))
    return UInt64[rand(master, UInt64) for _ = 1:nlanes]
end

# An interrupt of a lane run: raise the stop flag (no block starts another lane),
# then — once per call — tell the user that the running lanes finish first.
function _cmc_lane_interrupt!(stop::Threads.Atomic{Bool},
                              interrupted::Threads.Atomic{Bool})
    stop[] = true
    Threads.atomic_xchg!(interrupted, true) || @warn(
        "run_cmc: interrupted — no further lane starts; the lanes already running " *
        "finish (and write their checkpoint files) before the InterruptException " *
        "is thrown")
    return nothing
end

# Wait for every lane task. An `InterruptException` delivered to the caller while it
# waits sets `stop` (no block starts another lane) and `interrupted`, and the wait
# goes on: no task outlives the call, so no lane keeps writing its checkpoint file
# after `run_cmc` has thrown.
function _cmc_lane_wait(tasks::Vector{Task}, stop::Threads.Atomic{Bool},
                        interrupted::Threads.Atomic{Bool})
    for t in tasks
        while !istaskdone(t)
            try
                wait(t)
            catch err
                err isa InterruptException || break   # a failed task: reported below
                _cmc_lane_interrupt!(stop, interrupted)
            end
        end
    end
    for t in tasks
        istaskfailed(t) && throw(TaskFailedException(t))
    end
    return nothing
end

# The lane scheduler, independent of what a lane computes: run `run_lane!(k)` for
# k = 1:nlanes over at most `nt` tasks — contiguous blocks of `cld(nlanes, nt)` lanes,
# one task per block, each running its lanes in order (`nt == 1`: every lane on the
# calling task). A lane that throws stops no other lane: the failure is recorded in
# slot k, then logged at once (`@error`, naming `lane_name(k)`). An
# `InterruptException` (raised in a lane, or delivered to the waiting caller) raises
# the stop flag that every block checks before it starts a lane; the running lanes
# finish, and once every task has finished a bare `InterruptException` is thrown.
# Otherwise returns the failures in lane order (empty when every lane succeeded).
function _run_lane_blocks(run_lane!, lane_name, nlanes::Int,
                          nt::Int)::Vector{CapturedException}
    # slot k of `failures` is written by lane k only
    failures = Vector{Union{Nothing,CapturedException}}(nothing, nlanes)
    stop = Threads.Atomic{Bool}(false)          # no block starts another lane
    interrupted = Threads.Atomic{Bool}(false)
    function run_block(block)
        for k in block
            stop[] && break
            try
                run_lane!(k)
            catch err
                if err isa InterruptException
                    _cmc_lane_interrupt!(stop, interrupted)
                else
                    bt = catch_backtrace()
                    failures[k] = CapturedException(err, bt)
                    @error("run_cmc: $(lane_name(k)) failed; the other lanes go on",
                           exception = (err, bt))
                end
            end
        end
        return nothing
    end
    if nt == 1
        run_block(1:nlanes)
    else
        tasks = Task[Threads.@spawn(run_block(block))
                     for block in Iterators.partition(1:nlanes, cld(nlanes, nt))]
        _cmc_lane_wait(tasks, stop, interrupted)
    end
    interrupted[] && throw(InterruptException())
    return CapturedException[f for f in failures if f !== nothing]
end

"""
    run_cmc(H::TiledHamiltonian, constraints::AbstractVector{MagnetizationConstraint};
            temperature = nothing, kT = nothing,
            observables = c -> cmc_observables(H, c), evaluables = cmc_evaluables(),
            ntasks = nothing, seed = rand(UInt64), checkpoint = nothing,
            kwargs...) -> Vector{CMCResult}

Run one independent constrained chain per constraint — a *lane* — over threads, and
return their results **in the order of `constraints`**
(`docs/specs/constrained-mc.md` D4). The keywords are those of the single-chain
[`run_cmc`](@ref) and mean the same for every lane: `temperature` / `kT`,
`sweeps_therm`, `sweeps_measure`, `measure_interval`, `step`, `adapt_target`,
`adapt_interval`, `renorm_interval`, `nbins`, `evaluables`, `init` (rotated onto each
lane's direction), `carryover`, `checkpoint_interval` — plus `ntasks` and the
per-lane `observables` and `checkpoint` below.

```julia
cons = [MagnetizationConstraint(θ, 0.0) for θ in range(0, π / 2; length = 7)]
rs = run_cmc(H, cons; temperature = 300.0, measure_interval = 10, seed = 1)
[r.points[1].stats[:torque].mean for r in rs]    # ⟨T⟩ per cell, one per θ
```

**Seeds and reproducibility.** The lane seeds are drawn from a master
`Xoshiro(UInt64(seed))` as `rand(master, UInt64)` in lane order `k = 1, 2, …`, all
before any lane runs; lane `k` is the single-chain run with its lane seed, which its
result records (`CMCResult.seed`, also stored in the lane's checkpoint). So `rs[k]`
is bit-identical to `run_cmc(H, constraints[k]; seed = rs[k].seed, observables =
observables(constraints[k]), …)` with the same other keywords, and the whole vector
is bitwise independent of `ntasks` and of the thread count (the lanes share no
mutable state; `H` is only read). Duplicate constraints are allowed: they are
independent replicas with distinct lane seeds.

**Observables** are a FUNCTION of the constraint, `constraint -> Vector{Observable}`,
called once per lane in lane order on the calling task before any lane starts —
`:M_par` and `:constraint_residual` measure along the constraint they were built
for, so a single vector shared by every lane would measure every lane along one
direction. Giving a vector (or anything that is not callable) throws an
`ArgumentError`. Add observables of your own with `c -> [cmc_observables(H, c);
extra]`. Every lane's set is also checked against the lane's constraint before any
lane runs, so `c -> obs` returning one fixed vector built by `cmc_observables` for
one direction is refused (`ArgumentError`, naming the lane) as soon as a lane has
another direction or other weights. `evaluables` is one vector shared by every
lane (evaluables read only the binned statistics, which do not depend on the
direction). With `ntasks ≥ 2` lanes measure and finalize concurrently, so
observables and evaluables of your own must not share mutable state across lanes
(the defaults hold only the lock-guarded torque memo of [`cmc_observables`](@ref),
one per call, which changes no value).

**Threads.** `ntasks` (default `min(length(constraints), Threads.nthreads())`)
bounds the number of tasks: the lanes are split into contiguous blocks of
`cld(length(constraints), ntasks)` lanes, one task per block, each running its
lanes in order; `ntasks = 1` runs every lane serially on the calling task.

**Checkpoints.** With `checkpoint = path`, lane `k` writes its own file
`"<stem>.lane<k>.jld2"` — `stem` is `path` without a trailing `".jld2"` (any letter
case), else `path` itself; a path naming a directory is refused — an ordinary
kind-`"cmc"` file holding the lane's constraint and seed. Continue an interrupted
lane with [`resume`](@ref)`(file, H)` (with the observables that lane was given); a
completed lane's file returns its stored result. There is no resume of the whole
set: resume the unfinished lanes one by one. Files of an earlier call with the
same stem are not removed: a lane that fails before its first write, or a re-run
with fewer lanes, leaves the earlier files in place — check their constraint and
seed (or use a fresh stem) before reusing them.

**Failures and interrupts.** Every argument is checked before any lane runs: an
empty constraint vector, `ntasks < 1`, the temperatures and the other plan
keywords, a vector or non-callable `observables`, a function that does not return
a `Vector{Observable}`, the observable / evaluable checks, a set from
`cmc_observables` built for another constraint or another `H`, every constraint's
resolution against `H` and its constrained start from `init`, the checkpoint path,
`or_per_metropolis` and `sweep_tasks` (unsupported, as for one chain); an unknown
keyword is a `MethodError` at the call. A lane that throws does not stop the
others: the failure is logged at once (`@error`, naming the lane and `n̂`), every
other lane runs, and then all failures are thrown together as a
`CompositeException` of `CapturedException`s in lane order, whatever `ntasks`. For
long scans pass `checkpoint`: the completed lanes' files then return their results
through `resume`. An `InterruptException` (Ctrl-C on the caller, or raised in a
lane) stops every block from starting another lane and logs a warning saying so;
lanes already running are NOT stopped mid-run — they finish (and write their
files) — and `run_cmc` throws a bare `InterruptException` once every task has
finished.
"""
function run_cmc(H::TiledHamiltonian,
                 constraints::AbstractVector{MagnetizationConstraint};
                 temperature = nothing, kT = nothing,
                 sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                 measure_interval::Integer = 1, step::Real = 0.6,
                 adapt_target::Real = 0.5, adapt_interval::Integer = 50,
                 renorm_interval::Integer = 1_000, nbins::Integer = 32,
                 observables = c -> cmc_observables(H, c),
                 evaluables::Vector{Evaluable} = cmc_evaluables(),
                 init = nothing, carryover::Bool = true,
                 ntasks::Union{Nothing,Integer} = nothing,
                 seed::Integer = rand(UInt64),
                 checkpoint::Union{Nothing,AbstractString} = nothing,
                 checkpoint_interval::Integer = 0,
                 or_per_metropolis = nothing,
                 sweep_tasks = nothing)::Vector{CMCResult}
    # --- validate: every door, before any lane runs (the errors arrive unwrapped) ---
    _cmc_refuse_unsupported(or_per_metropolis, sweep_tasks)
    nlanes = length(constraints)
    nlanes >= 1 || throw(ArgumentError("run_cmc: the constraint vector is empty"))
    nt = ntasks === nothing ? min(nlanes, Threads.nthreads()) : Int(ntasks)
    nt >= 1 || throw(ArgumentError("run_cmc: ntasks must be ≥ 1; got $nt"))
    nt = min(nt, nlanes)
    # resolved once: every lane gets the same Float64 ladder (and a one-shot
    # iterator is consumed once)
    kts = resolve_kt(temperature, kT)
    plan_kw = (; sweeps_therm = sweeps_therm, sweeps_measure = sweeps_measure,
               measure_interval = measure_interval, step = step,
               adapt_target = adapt_target, adapt_interval = adapt_interval,
               renorm_interval = renorm_interval, nbins = nbins)
    UpdatePlan(kts; plan_kw..., or_per_metropolis = 0, carryover = carryover,
               sweep_tasks = 1, seed = seed)              # the plan-level doors
    if checkpoint !== nothing
        _check_checkpoint_interval(checkpoint_interval)
        _cmc_lane_checkpoint_stem(checkpoint)
    end
    observables isa AbstractVector && throw(ArgumentError(
        "run_cmc over a vector of constraints takes observables as a function " *
        "constraint -> Vector{Observable}, not a vector: :M_par and " *
        ":constraint_residual measure along the constraint they were built for, so " *
        "one vector shared by every lane would measure every lane along one " *
        "direction. Pass e.g. c -> [cmc_observables(H, c); extra]"))
    lane_constraints = collect(MagnetizationConstraint, constraints)
    lane_observables = Vector{Vector{Observable}}(undef, nlanes)
    for k = 1:nlanes
        con = lane_constraints[k]
        # the constraint and its start, independently of the observables function
        rc, _ = _cmc_setup(H, con, init)
        obs = try
            observables(con)
        catch err
            (err isa MethodError && err.f === observables) || rethrow()
            throw(ArgumentError(
                "run_cmc: observables must be a function constraint -> " *
                "Vector{Observable}; got a $(typeof(observables))"))
        end
        obs isa Vector{Observable} || throw(ArgumentError(
            "run_cmc: observables(constraint) must return a Vector{Observable}; " *
            "lane $k got a $(typeof(obs))"))
        _check_observables(obs)
        _check_evaluables(obs, evaluables)
        # a set built for another constraint (e.g. `c -> obs` returning one fixed
        # vector) is refused here, before any lane runs
        _check_cmc_observables(obs, H, rc, "run_cmc (lane $k)")
        lane_observables[k] = obs
    end
    lane_seeds = _cmc_lane_seeds(seed, nlanes)    # all drawn before any lane runs
    lane_path(k) = checkpoint === nothing ? nothing :
                   _cmc_lane_checkpoint_path(checkpoint, k)

    # --- run: slot k of `results` is written by lane k only ---
    results = Vector{CMCResult}(undef, nlanes)
    function run_lane!(k)
        results[k] = run_cmc(H, lane_constraints[k]; kT = kts, plan_kw...,
                             observables = lane_observables[k],
                             evaluables = evaluables, init = init,
                             carryover = carryover, seed = lane_seeds[k],
                             checkpoint = lane_path(k),
                             checkpoint_interval = checkpoint_interval)
        return nothing
    end
    lane_name(k) = "lane $k (n̂ = $(Tuple(lane_constraints[k].direction)))"
    failed = _run_lane_blocks(run_lane!, lane_name, nlanes, nt)

    # --- aggregate ---
    isempty(failed) || throw(CompositeException(failed))
    return results
end

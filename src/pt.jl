# Replica exchange (parallel tempering) over threads
# (design + determinism guarantees: `docs/specs/pt-threads-determinism.md`).
#
# Lane r owns rung r of the temperature ladder: its ChainState, scratch, RNG, and
# accumulators. Between segments (every `exchange_interval` compound sweeps), the
# coordinator attempts adjacent-pair swaps of the chain *payload* (config / rows /
# energy) — RNG, step, and accumulators stay with the lane, so every lane's
# measurement stream is a fixed-temperature marginal and the adapted step remains
# per-temperature.
#
# Determinism: every random decision is attributed to a specific RNG whose
# consumption order is fixed by the (serial) segment schedule, never by thread
# timing — lane RNGs are consumed only inside that lane's sweeps, the dedicated
# exchange RNG only on the coordinator, with one uniform drawn *unconditionally*
# per attempted pair in ascending pair order. Results are bit-identical for a fixed
# seed regardless of `ntasks` / `JULIA_NUM_THREADS` (gated).

"""
    PTResult

Result of [`run_pt`](@ref): `points` (one [`TempResult`](@ref) per ladder rung, in
ladder order), the adjacent-pair `swap_acceptance` fractions (length
`n_rungs − 1`; over the whole run), each lane's `final_config`, and the run `seed`.
Prints as a summary table.
"""
struct PTResult
    points::Vector{TempResult}
    swap_acceptance::Vector{Float64}
    final_configs::Vector{SpinConfig}
    seed::UInt64
end

Base.show(io::IO, r::PTResult) =
    print(io, "PTResult(", length(r.points), " rungs, ",
          length(first(r.final_configs)), " sites)")

function Base.show(io::IO, ::MIME"text/plain", r::PTResult)
    println(io, "PTResult: ", length(r.points), " rungs, ",
            length(first(r.final_configs)), " sites, seed ", r.seed)
    _print_points_table(io, r.points, length(first(r.final_configs)))
    print(io, "  swap acceptance: ")
    println(io, join([@sprintf("%.2f", a) for a in r.swap_acceptance], " "))
    return nothing
end

# One parallel-tempering lane: a rung of the ladder with its chain, scratch, and
# (during measurement) accumulators.
mutable struct _PTLane
    const st::ChainState
    const sc::SweepScratch
    const kt::Float64
    const β::Float64
    accs::Vector{ObsAccumulator}
    phase_sweeps::Int              # sweeps done in the current phase
end

# Swap the replica payload between two chains (reference swaps — O(1)).
function _swap_payload!(a::ChainState, b::ChainState)
    a.config, b.config = b.config, a.config
    a.zrows, b.zrows = b.zrows, a.zrows
    a.energy, b.energy = b.energy, a.energy
    return nothing
end

# Run `n` compound sweeps of one lane (thread-confined: touches only lane state).
# In the measurement phase, adaptation is off (frozen) and measurements fire every
# `measure_interval` sweeps.
function _lane_segment!(lane::_PTLane, H::TiledHamiltonian, plan::UpdatePlan,
                        n::Int, measure::Bool)
    st = lane.st
    for _ = 1:n
        lane.phase_sweeps += 1
        _compound_sweep!(st, H, lane.β, lane.sc, plan)
        measure || (lane.phase_sweeps % plan.adapt_interval == 0 &&
                    _adapt_step!(st, plan.adapt_target))
        lane.phase_sweeps % plan.renorm_interval == 0 && _renormalize!(st, H)
        if measure && lane.phase_sweeps % plan.measure_interval == 0
            for acc in lane.accs
                _measure!(acc, st.config, st.energy, H)
            end
        end
    end
    return nothing
end

# Run all lanes for one phase (`total` sweeps each) in segments of `seglen` sweeps,
# with adjacent-pair exchange attempts between segments (alternating even/odd pair
# parity, one unconditional uniform per attempted pair in ascending order).
# `done0` resumes the phase mid-flight from a checkpoint; `ck` writes periodic
# checkpoints at segment boundaries. Returns the exchange parity to carry into the
# next phase.
function _run_pt_phase!(lanes::Vector{_PTLane}, H::TiledHamiltonian,
                        plan::UpdatePlan, total::Int, seglen::Int, measure::Bool,
                        exchange_rng::Xoshiro, swap_att::Vector{Int},
                        swap_acc::Vector{Int}, ntasks::Int, parity::Int;
                        done0::Int = 0, ck = nothing)::Int
    R = length(lanes)
    done = done0
    while done < total
        n = min(seglen, total - done)
        if ntasks <= 1
            for lane in lanes
                _lane_segment!(lane, H, plan, n, measure)
            end
        else
            chunk = cld(R, ntasks)
            @sync for lo = 1:chunk:R
                hi = min(lo + chunk - 1, R)
                Threads.@spawn for r = lo:hi
                    _lane_segment!(lanes[r], H, plan, n, measure)
                end
            end
        end
        done += n
        if done < total
            for i = (1 + parity):2:(R - 1)
                u = rand(exchange_rng)      # drawn unconditionally — determinism
                swap_att[i] += 1
                a, b = lanes[i], lanes[i + 1]
                logw = (1 / a.kt - 1 / b.kt) * (a.st.energy - b.st.energy)
                if u < exp(min(0.0, logw))
                    _swap_payload!(a.st, b.st)
                    swap_acc[i] += 1
                end
            end
            parity = 1 - parity
        end
        _ck_pt!(ck, n, H, lanes, measure ? :measure : :therm, done, parity,
                exchange_rng, swap_att, swap_acc)
    end
    return parity
end

# The shared phase driver of `run_pt` and a "pt"-kind `resume`.
function _pt_run!(lanes::Vector{_PTLane}, H::TiledHamiltonian, plan::UpdatePlan,
                  observables::Vector{Observable}, evaluables::Vector{Evaluable},
                  exchange_interval::Int, nt::Int, exchange_rng::Xoshiro,
                  swap_att::Vector{Int}, swap_acc::Vector{Int}, phase0::Symbol,
                  done0::Int, parity0::Int, ck)::PTResult
    parity = parity0
    mdone0 = 0
    if phase0 === :therm
        parity = _run_pt_phase!(lanes, H, plan, plan.sweeps_therm,
                                exchange_interval, false, exchange_rng, swap_att,
                                swap_acc, nt, parity; done0 = done0, ck = ck)
        planned = fld(plan.sweeps_measure, plan.measure_interval)
        for lane in lanes
            _renormalize!(lane.st, H)
            lane.st.frozen = true
            lane.st.acc_metro = 0
            lane.st.att_metro = 0
            lane.st.acc_or = 0
            lane.st.att_or = 0
            lane.st.max_drift = 0.0
            lane.accs = [ObsAccumulator(o, planned, plan.nbins)
                         for o in observables]
            lane.phase_sweeps = 0
        end
        # boundary checkpoint: the measurement phase starts fresh from this state
        ck === nothing ||
            _write_ckpt_pt(ck, H, lanes, :measure, 0, parity, exchange_rng,
                           swap_att, swap_acc)
    else
        mdone0 = done0
    end
    _run_pt_phase!(lanes, H, plan, plan.sweeps_measure, exchange_interval, true,
                   exchange_rng, swap_att, swap_acc, nt, parity; done0 = mdone0,
                   ck = ck)
    R = length(lanes)
    points = [let st = lane.st
                  acc_m = st.att_metro == 0 ? NaN : st.acc_metro / st.att_metro
                  acc_o = st.att_or == 0 ? NaN : st.acc_or / st.att_or
                  TempResult(lane.kt, lane.kt / KB_EV,
                             _finalize_stats(lane.accs, evaluables, lane.kt,
                                             H.n_sites),
                             acc_m, acc_o, st.step, st.max_drift)
              end
              for lane in lanes]
    swaps = [swap_att[i] == 0 ? NaN : swap_acc[i] / swap_att[i] for i = 1:(R - 1)]
    return PTResult(points, swaps, [copy(lane.st.config) for lane in lanes],
                    plan.seed)
end

"""
    run_pt(H::TiledHamiltonian; temperature = nothing, kT = nothing,
           exchange_interval = 10, ntasks = nothing, kwargs...) -> PTResult

Replica-exchange (parallel-tempering) Monte Carlo: one chain (**lane**) per rung of
a strictly monotone temperature ladder (**exactly one** of `temperature` [kelvin] /
`kT` [model energy units], length ≥ 2), all lanes sweeping concurrently over
threads. Every `exchange_interval` compound sweeps, adjacent rungs attempt to swap
their chain payloads with probability `min(1, exp((βᵢ−βⱼ)(Eᵢ−Eⱼ)))` (alternating
even/odd pairs) — so cold rungs keep escaping metastable basins through the hot end
of the ladder. Exchanges run during thermalization and measurement alike.

`ntasks` caps the concurrent lane tasks (default `min(n_rungs, nthreads())`).
Results are **bit-identical for a fixed seed regardless of `ntasks` and the thread
count** — every random decision has a dedicated RNG whose consumption order is
fixed by the segment schedule (lane RNGs inside sweeps; a coordinator exchange RNG
drawn once per attempted pair).

`checkpoint` / `checkpoint_interval` write restartable checkpoints at segment
boundaries (interval in sweeps, `0` ⇒ only at the thermalization→measurement
boundary); continue with [`resume`](@ref) — bit-identical to an uninterrupted run.

Everything else — `sweeps_therm`, `sweeps_measure`, `measure_interval`,
`or_per_metropolis`, `step` / `adapt_target` / `adapt_interval` (adaptation is
per-lane, thermalization-only), `renorm_interval`, `nbins`, `observables`,
`evaluables`, `init` (every lane starts from it; default: independent random),
`seed` — as in [`run_mc`](@ref). Lane `r`'s statistics land in `points[r]`
(ladder order); adjacent-pair swap acceptances (diagnostic: aim for O(0.2–0.5),
tighten the ladder where they collapse) in `swap_acceptance`.
"""
function run_pt(H::TiledHamiltonian; temperature = nothing, kT = nothing,
                exchange_interval::Integer = 10,
                ntasks::Union{Nothing,Integer} = nothing,
                sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                measure_interval::Integer = 1, or_per_metropolis::Integer = 0,
                step::Real = 0.6, adapt_target::Real = 0.5,
                adapt_interval::Integer = 50, renorm_interval::Integer = 1_000,
                nbins::Integer = 32,
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(),
                init = nothing, seed::Integer = rand(UInt64),
                checkpoint::Union{Nothing,AbstractString} = nothing,
                checkpoint_interval::Integer = 0)::PTResult
    kts = resolve_kt(temperature, kT)
    R = length(kts)
    R >= 2 || throw(ArgumentError("parallel tempering needs a ladder of ≥ 2 " *
                                  "temperatures; got $R"))
    (all(diff(kts) .> 0) || all(diff(kts) .< 0)) || throw(ArgumentError(
        "the temperature ladder must be strictly monotone; got $kts"))
    exchange_interval >= 1 || throw(ArgumentError(
        "exchange_interval must be ≥ 1; got $exchange_interval"))
    nt = ntasks === nothing ? min(R, Threads.nthreads()) : Int(ntasks)
    nt >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $nt"))
    plan = UpdatePlan(kts; sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = or_per_metropolis, step = step,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = false, seed = seed)
    _check_observables(observables)
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "pt", Int(exchange_interval))

    # RNG discipline: master → one Xoshiro per lane (fixed order), then the
    # exchange RNG; initial configs come from each lane's own RNG.
    master = Xoshiro(plan.seed)
    lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                         rand(master, UInt64), rand(master, UInt64)) for _ = 1:R]
    exchange_rng = Xoshiro(rand(master, UInt64), rand(master, UInt64),
                           rand(master, UInt64), rand(master, UInt64))
    lanes = [_PTLane(ChainState(H, _initial_config(H, init, lane_rngs[r]),
                                lane_rngs[r], plan.step0),
                     SweepScratch(H), kts[r], 1.0 / kts[r], ObsAccumulator[], 0)
             for r = 1:R]
    swap_att = zeros(Int, R - 1)
    swap_acc = zeros(Int, R - 1)
    return _pt_run!(lanes, H, plan, observables, evaluables,
                    Int(exchange_interval), nt, exchange_rng, swap_att, swap_acc,
                    :therm, 0, 0, ck)
end

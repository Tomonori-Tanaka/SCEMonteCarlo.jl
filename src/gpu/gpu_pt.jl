# Replica exchange (parallel tempering) over device chains (decision record
# docs/specs/gpu-prototype.md G8).
#
# Lane r owns rung r of the temperature ladder: one `GPUChainState` (the device
# chain), a host `ChainState` mirror (renormalization + measurement staging), and
# the accumulators. All rungs share one uploaded `GPUTiledHamiltonian` and run
# round-robin on the backend's single queue — the sweep kernel of a production
# supercell already saturates the device, so rung-level concurrency would buy
# nothing and cost the launch-order determinism. Between segments (every
# `exchange_interval` sweeps) the adjacent-pair swap rule of `run_pt` runs on the
# host: the incremental energies are host-side scalars already, so an exchange
# touches no device memory — `_swap_payload!` exchanges the device array
# REFERENCES (config / zrows) and the energy, while the keyed-RNG bookkeeping
# (seed, sweep_index), the step, and the counters stay with the lane, exactly
# mirroring the `ChainState` payload partition of pt.jl.
#
# Determinism (G8): every chain's trajectory is a pure function of its
# `(device seed, site, sweep)` keyed Philox coordinates — independent of how the
# rungs interleave — and the exchange uniforms come from one host Xoshiro
# consumed in the fixed serial schedule below, so a run is bitwise reproducible
# for a fixed (seed, backend, workgroupsize, package + Julia version). The
# master-seed derivation order is part of that contract (gated):
#   master = Xoshiro(seed)
#   for r = 1:R:  lane_rngs[r]  ← 4 UInt64 words        (host mirror + init)
#   exchange_rng                ← 4 UInt64 words
#   for r = 1:R:  device seed r ← 1 UInt64 word         (keys rung r's Philox)
# Distinct 64-bit keys give independent Philox streams (the ctr[4] = 0 layout of
# G2 is untouched; the reserved replica-tag word stays available for a future
# in-kernel multi-chain batch).

# One GPU parallel-tempering lane: a rung of the ladder with its device chain,
# host mirror, and (during measurement) accumulators.
mutable struct _GPUPTLane{G<:GPUChainState}
    const gst::G
    const st::ChainState           # host mirror: renorm round-trip, measurement,
                                   #   and the max_drift diagnostic
    const kt::Float64
    const β::Float64
    accs::Vector{ObsAccumulator}
    phase_sweeps::Int              # sweeps done in the current phase
end

# Swap the replica payload between two device chains (reference swaps — O(1), no
# device traffic). The moved/stayed partition mirrors `_swap_payload!` on
# `ChainState` and is pinned exhaustively in test_gpu_pt.jl.
function _swap_payload!(a::GPUChainState, b::GPUChainState)
    a.config, b.config = b.config, a.config
    a.zrows, b.zrows = b.zrows, a.zrows
    a.energy, b.energy = b.energy, a.energy
    return nothing
end

# One adjacent-pair swap attempt on the device lanes: the shared `_swap_accepts`
# rule (pt.jl) on the host-side incremental energies.
function _gpu_attempt_swap!(a::_GPUPTLane, b::_GPUPTLane, i::Int, u::Float64,
                            swap_att::Vector{Int}, swap_acc::Vector{Int})
    swap_att[i] += 1
    if _swap_accepts(a.kt, b.kt, a.gst.energy, b.gst.energy, u)
        _swap_payload!(a.gst, b.gst)
        swap_acc[i] += 1
    end
    return nothing
end

# The device half of `_adapt_step!` (updates.jl) — the same window arithmetic on
# the device chain's host-side counters. `GPUChainState` has no `frozen` flag:
# the phase driver simply never calls this in the measurement phase.
function _gpu_adapt_step!(gst::GPUChainState, target::Float64)::Float64
    if gst.att_metro > 0
        a = gst.acc_metro / gst.att_metro
        gst.step = clamp(gst.step * exp(0.5 * (a - target)), 1e-3, Float64(π))
    end
    gst.acc_metro = 0
    gst.att_metro = 0
    return gst.step
end

# Run `n` device sweeps of one lane. Same in-sweep event order as the CPU
# `_lane_segment!`: adapt (thermalization only) → renormalize → measure. The
# renormalization is the host round-trip of `gpu_run_sweeps!`; measurement
# downloads into the host mirror and reuses the ordinary accumulator machinery.
function _gpu_lane_segment!(lane::_GPUPTLane, gH::GPUTiledHamiltonian,
                            plan::UpdatePlan, n::Int, measure::Bool, ws::Int)
    gst = lane.gst
    for _ = 1:n
        lane.phase_sweeps += 1
        gpu_metropolis_sweep!(gst, gH, lane.β; workgroupsize = ws)
        measure || (lane.phase_sweeps % plan.adapt_interval == 0 &&
                    _gpu_adapt_step!(gst, plan.adapt_target))
        if lane.phase_sweeps % plan.renorm_interval == 0
            to_host!(lane.st, gst)
            _renormalize!(lane.st, gH.host)
            _from_host!(gst, lane.st)
        end
        if measure && lane.phase_sweeps % plan.measure_interval == 0
            to_host!(lane.st, gst)
            for acc in lane.accs
                _measure!(acc, lane.st.config, lane.st.energy, gH.host)
            end
        end
    end
    return nothing
end

# Run all lanes for one phase (`total` sweeps each) in segments of `seglen`
# sweeps, with adjacent-pair exchange attempts between segments — the serial
# reference schedule of `_run_pt_phase!` (alternating even/odd pair parity, one
# uniform drawn unconditionally per attempted pair in ascending order). `done0`
# resumes the phase mid-flight from a checkpoint: writes land only at segment
# boundaries (after a full segment, or the phase's final short one — a periodic
# write CAN land at done == total), so `n = min(seglen, total − done)`
# regenerates the identical segmentation from any stored `done0`; `ck` writes
# periodic checkpoints at segment boundaries. Returns the exchange parity to
# carry into the next phase.
function _gpu_pt_phase!(lanes::Vector{<:_GPUPTLane}, gH::GPUTiledHamiltonian,
                        plan::UpdatePlan, total::Int, seglen::Int, measure::Bool,
                        exchange_rng::Xoshiro, swap_att::Vector{Int},
                        swap_acc::Vector{Int}, parity::Int, ws::Int;
                        done0::Int = 0, ck = nothing)::Int
    R = length(lanes)
    done = done0
    while done < total
        n = min(seglen, total - done)
        for lane in lanes
            _gpu_lane_segment!(lane, gH, plan, n, measure, ws)
        end
        done += n
        if done < total
            for i = (1 + parity):2:(R - 1)
                u = rand(exchange_rng)  # drawn unconditionally — determinism
                _gpu_attempt_swap!(lanes[i], lanes[i + 1], i, u, swap_att,
                                   swap_acc)
            end
            parity = 1 - parity
        end
        _ck_gpu_pt!(ck, n, lanes, measure ? :measure : :therm, done, parity,
                    exchange_rng, swap_att, swap_acc, ws)
    end
    return parity
end

# The shared phase driver of `gpu_run_pt` and a "gpu_pt"-kind `resume` — the
# device mirror of `_pt_run!`. `phase0`/`done0`/`parity0` enter mid-flight from
# a checkpoint (fresh entry: `:therm`, 0, 0); `ck` writes checkpoints at segment
# boundaries plus the unconditional thermalization → measurement boundary write.
# As `_pt_run!`, there is NO end-of-run write — a completed run's file sits at
# the last mid-measure write, which is what keeps the resume gates non-vacuous.
function _gpu_pt_run!(lanes::Vector{<:_GPUPTLane}, gH::GPUTiledHamiltonian,
                      plan::UpdatePlan, observables::Vector{Observable},
                      evaluables::Vector{Evaluable}, exchange_interval::Int,
                      exchange_rng::Xoshiro, swap_att::Vector{Int},
                      swap_acc::Vector{Int}, phase0::Symbol, done0::Int,
                      parity0::Int, ck, ws::Int)::PTResult
    H = gH.host
    parity = parity0
    mdone0 = 0
    if phase0 === :therm
        parity = _gpu_pt_phase!(lanes, gH, plan, plan.sweeps_therm,
                                exchange_interval, false, exchange_rng, swap_att,
                                swap_acc, parity, ws; done0 = done0, ck = ck)
        # thermalization → measurement boundary: renormalize, freeze the counters
        # into a fresh measurement window, hand each lane its accumulators
        # (as run_pt)
        planned = fld(plan.sweeps_measure, plan.measure_interval)
        for lane in lanes
            to_host!(lane.st, lane.gst)
            _renormalize!(lane.st, H)
            _from_host!(lane.gst, lane.st)
            lane.gst.acc_metro = 0
            lane.gst.att_metro = 0
            # the mirror's `frozen` is never read on the GPU path; it is kept in
            # step with the CPU boundary (and with a restored mirror) anyway so
            # fresh and resumed lanes cannot silently diverge field-by-field
            lane.st.frozen = true
            lane.st.max_drift = 0.0    # report measurement-phase drift only
            lane.accs = [ObsAccumulator(o, planned, plan.nbins)
                         for o in observables]
            lane.phase_sweeps = 0
        end
        # boundary checkpoint: the measurement phase starts fresh from this state
        ck === nothing ||
            _write_ckpt_gpu_pt(ck, lanes, :measure, 0, parity, exchange_rng,
                               swap_att, swap_acc, ws)
    else
        mdone0 = done0
    end
    _gpu_pt_phase!(lanes, gH, plan, plan.sweeps_measure, exchange_interval, true,
                   exchange_rng, swap_att, swap_acc, parity, ws; done0 = mdone0,
                   ck = ck)

    points = [let gst = lane.gst
                  acc_m = gst.att_metro == 0 ? NaN : gst.acc_metro / gst.att_metro
                  TempResult(lane.kt, lane.kt / KB_EV,
                             _finalize_stats(lane.accs, evaluables, lane.kt,
                                             H.n_active),
                             acc_m, NaN, gst.step, lane.st.max_drift)
              end
              for lane in lanes]
    R = length(lanes)
    swaps = [swap_att[i] == 0 ? NaN : swap_acc[i] / swap_att[i] for i = 1:(R - 1)]
    finals = [begin
                  to_host!(lane.st, lane.gst)
                  copy(lane.st.config)
              end
              for lane in lanes]
    return PTResult(points, swaps, finals, plan.seed)
end

"""
    gpu_run_pt(gH::GPUTiledHamiltonian; temperature = nothing, kT = nothing,
               exchange_interval = 10, workgroupsize = 128, kwargs...) -> PTResult
    gpu_run_pt(backend, H::TiledHamiltonian; kwargs...) -> PTResult

Replica-exchange (parallel-tempering) Monte Carlo on a KernelAbstractions
backend: one device chain per rung of a strictly monotone temperature ladder
(**exactly one** of `temperature` [kelvin] / `kT` [model energy units], length
≥ 2), all rungs sharing the one uploaded table set of `gH` and sweeping
round-robin on the backend queue. Every `exchange_interval` sweeps, adjacent
rungs attempt to swap their chain payloads with the same
`min(1, exp((βᵢ−βⱼ)(Eᵢ−Eⱼ)))` rule as [`run_pt`](@ref) (alternating even/odd
pairs, during thermalization and measurement alike); the incremental energies
are host-side scalars, so an exchange moves device array references only — no
device traffic. Returns the same [`PTResult`](@ref) as `run_pt`.

Differences from [`run_pt`](@ref) (the device scope, G8):

- **Metropolis only** — no overrelaxation (`or_per_metropolis` does not exist
  here), and `acceptance_or` is `NaN` in every rung's [`TempResult`](@ref).
- **Different chains.** The device sweep draws keyed Philox noise (a pure
  function of the per-rung device seed, site, and sweep index), so a
  `gpu_run_pt` run and a `run_pt` run are different realizations of the same
  ensemble — compare them statistically, never bitwise. A `gpu_run_pt` run IS
  bitwise reproducible for a fixed (`seed`, backend, `workgroupsize`,
  package + Julia version); rung trajectories are interleaving-independent by
  construction, and the exchange uniforms come from a dedicated host RNG in a
  fixed serial schedule.
- **Host measurement.** Every `measure_interval`-th sweep downloads the chain
  into a host mirror ([`to_host!`](@ref)) and feeds the ordinary observable /
  binning machinery — size `measure_interval` for the download cost.
  Renormalization (every `renorm_interval` sweeps, plus once at the
  thermalization → measurement boundary) is the same host round-trip as
  [`gpu_run_sweeps!`](@ref); step adaptation runs on the host counters during
  thermalization only.
`checkpoint` / `checkpoint_interval` write restartable checkpoints at segment
boundaries (kind `"gpu_pt"`, interval in sweeps, `0` ⇒ only at the
thermalization → measurement boundary); continue with
`resume(path, gH::GPUTiledHamiltonian)` — bit-identical to an uninterrupted run
on the same backend. The keyed RNG makes the stored chain state tiny: per rung,
the configuration + incremental energy + the two integers `(seed, sweep_index)`
(plus step and counters) fully determine the trajectory.

Everything else — `sweeps_therm`, `sweeps_measure`, `measure_interval`,
`step` / `adapt_target` / `adapt_interval`, `renorm_interval`, `nbins`,
`observables`, `evaluables`, `init` (every rung starts from it; default:
independent random), `seed` — as in [`run_pt`](@ref). `workgroupsize` (power of
two, pinned default 128) is part of the determinism scope, as for
[`gpu_metropolis_sweep!`](@ref) — it is stored in the checkpoint and a resume
uses the stored value. The convenience form uploads the tables first
(`GPUTiledHamiltonian(backend, H)`); pass a prebuilt `gH` to reuse an upload.
"""
function gpu_run_pt(gH::GPUTiledHamiltonian; temperature = nothing, kT = nothing,
                    exchange_interval::Integer = 10, sweeps_therm::Integer = 2_000,
                    sweeps_measure::Integer = 10_000,
                    measure_interval::Integer = 1, step::Real = 0.6,
                    adapt_target::Real = 0.5, adapt_interval::Integer = 50,
                    renorm_interval::Integer = 1_000, nbins::Integer = 32,
                    observables::Vector{Observable} = standard_observables(gH.host),
                    evaluables::Vector{Evaluable} = standard_evaluables(),
                    init = nothing, seed::Integer = rand(UInt64),
                    workgroupsize::Integer = 128,
                    checkpoint::Union{Nothing,AbstractString} = nothing,
                    checkpoint_interval::Integer = 0)::PTResult
    H = gH.host
    kts = resolve_kt(temperature, kT)
    R = length(kts)
    R >= 2 || throw(ArgumentError("parallel tempering needs a ladder of ≥ 2 " *
                                  "temperatures; got $R"))
    (all(diff(kts) .> 0) || all(diff(kts) .< 0)) || throw(ArgumentError(
        "the temperature ladder must be strictly monotone; got $kts"))
    exchange_interval >= 1 || throw(ArgumentError(
        "exchange_interval must be ≥ 1; got $exchange_interval"))
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    plan = UpdatePlan(kts; sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval, or_per_metropolis = 0,
                      step = step, adapt_target = adapt_target,
                      adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = false, sweep_tasks = 1, seed = seed)
    _check_observables(observables)
    _check_evaluables(observables, evaluables)
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan,
                            observables, "gpu_pt", Int(exchange_interval))

    # RNG discipline: the derivation order documented at the top of this file is
    # a gated contract — do not reorder.
    master = Xoshiro(plan.seed)
    lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                         rand(master, UInt64), rand(master, UInt64)) for _ = 1:R]
    exchange_rng = Xoshiro(rand(master, UInt64), rand(master, UInt64),
                           rand(master, UInt64), rand(master, UInt64))
    dev_seeds = [rand(master, UInt64) for _ = 1:R]
    lanes = [begin
                 st = ChainState(H, _initial_config(H, init, lane_rngs[r]),
                                 lane_rngs[r], plan.step0)
                 _GPUPTLane(GPUChainState(gH, st; seed = dev_seeds[r]), st,
                            kts[r], 1.0 / kts[r], ObsAccumulator[], 0)
             end
             for r = 1:R]
    swap_att = zeros(Int, R - 1)
    swap_acc = zeros(Int, R - 1)
    return _gpu_pt_run!(lanes, gH, plan, observables, evaluables,
                        Int(exchange_interval), exchange_rng, swap_att, swap_acc,
                        :therm, 0, 0, ck, ws)
end

gpu_run_pt(backend::Backend, H::TiledHamiltonian; kwargs...)::PTResult =
    gpu_run_pt(GPUTiledHamiltonian(backend, H); kwargs...)

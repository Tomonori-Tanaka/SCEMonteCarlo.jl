# The single-chain run driver: one temperature or a warm-started sweep (annealing).

# Validated run configuration (internal; also the checkpointable description of a
# run). `kts` is the resolved k_B·T ladder in model energy units.
struct UpdatePlan
    kts::Vector{Float64}
    sweeps_therm::Int
    sweeps_measure::Int
    measure_interval::Int
    or_per_metropolis::Int
    step0::Float64
    adapt_target::Float64
    adapt_interval::Int
    renorm_interval::Int
    nbins::Int
    carryover::Bool
    seed::UInt64

    function UpdatePlan(kts::Vector{Float64}; sweeps_therm::Integer,
                        sweeps_measure::Integer, measure_interval::Integer,
                        or_per_metropolis::Integer, step::Real, adapt_target::Real,
                        adapt_interval::Integer, renorm_interval::Integer,
                        nbins::Integer, carryover::Bool, seed::Integer)
        isempty(kts) && throw(ArgumentError("the temperature ladder is empty"))
        sweeps_therm >= 0 ||
            throw(ArgumentError("sweeps_therm must be ≥ 0; got $sweeps_therm"))
        sweeps_measure >= 1 ||
            throw(ArgumentError("sweeps_measure must be ≥ 1; got $sweeps_measure"))
        1 <= measure_interval <= sweeps_measure || throw(ArgumentError(
            "measure_interval must be in 1:sweeps_measure; got $measure_interval"))
        or_per_metropolis >= 0 || throw(ArgumentError(
            "or_per_metropolis must be ≥ 0; got $or_per_metropolis"))
        step > 0 || throw(ArgumentError("step must be > 0; got $step"))
        0 < adapt_target < 1 || throw(ArgumentError(
            "adapt_target must be in (0, 1); got $adapt_target"))
        adapt_interval >= 1 ||
            throw(ArgumentError("adapt_interval must be ≥ 1; got $adapt_interval"))
        renorm_interval >= 1 ||
            throw(ArgumentError("renorm_interval must be ≥ 1; got $renorm_interval"))
        nbins >= 2 || throw(ArgumentError("nbins must be ≥ 2; got $nbins"))
        seed >= 0 || throw(ArgumentError("seed must be ≥ 0; got $seed"))
        return new(kts, sweeps_therm, sweeps_measure, measure_interval,
                   or_per_metropolis, Float64(step), Float64(adapt_target),
                   adapt_interval, renorm_interval, nbins, carryover, UInt64(seed))
    end
end

"""
    TempResult

One temperature's finalized results: the `kT` / `temperature` labels (model energy
units / kelvin), the observable `stats` (`Dict{Symbol,ObservableStat}` — raw
observables with binning errors and `τ_int`, evaluables jackknifed), the measured
Metropolis / overrelaxation acceptance fractions (measurement phase only; `NaN`
where not applicable), the frozen proposal `final_step`, and the worst
incremental-energy `max_drift` seen at renormalization points.
"""
struct TempResult
    kT::Float64
    temperature::Float64
    stats::Dict{Symbol,ObservableStat}
    acceptance_metropolis::Float64
    acceptance_or::Float64
    final_step::Float64
    max_drift::Float64
end

Base.show(io::IO, p::TempResult) =
    print(io, "TempResult(kT=", @sprintf("%.6g", p.kT), ", ",
          length(p.stats), " stats)")

"""
    MCResult

Result of [`run_mc`](@ref): `points` (one [`TempResult`](@ref) per temperature, in
run order), the chain's `final_config`, and the run `seed`. Prints as a summary
table.
"""
struct MCResult
    points::Vector{TempResult}
    final_config::SpinConfig
    seed::UInt64
end

Base.show(io::IO, r::MCResult) =
    print(io, "MCResult(", length(r.points), " temperatures, ",
          length(r.final_config), " sites)")

function Base.show(io::IO, ::MIME"text/plain", r::MCResult)
    println(io, "MCResult: ", length(r.points), " temperature(s), ",
            length(r.final_config), " sites, seed ", r.seed)
    _print_points_table(io, r.points, length(r.final_config))
    return nothing
end

# The shared points table of MCResult / PTResult text/plain printing.
function _print_points_table(io::IO, points::Vector{TempResult}, nsites::Int)
    @printf(io, "  %-11s %-9s %-22s %-9s %-8s %-9s %-7s %-6s\n",
            "kT", "T[K]", "E/site", "C/kB", "|m|", "chi", "U", "acc")
    for p in points
        e = get(p.stats, :energy, nothing)
        estr = e === nothing ? "-" :
               @sprintf("%.6g ± %.2g", e.mean[1] / nsites, e.err[1] / nsites)
        @printf(io, "  %-11.5g %-9.4g %-22s %-9s %-8s %-9s %-7s %-6.3f\n",
                p.kT, p.temperature, estr,
                _stat_str(p.stats, :specific_heat), _stat_str(p.stats, :absm),
                _stat_str(p.stats, :susceptibility), _stat_str(p.stats, :binder),
                p.acceptance_metropolis)
    end
    return nothing
end

function _stat_str(stats::Dict{Symbol,ObservableStat}, name::Symbol)::String
    s = get(stats, name, nothing)
    s === nothing && return "-"
    return @sprintf("%.4g", s.mean[1])
end

# One compound sweep (Metropolis; overrelaxation mixes in via `or_per_metropolis`).
function _compound_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                          sc::SweepScratch, plan::UpdatePlan)
    metropolis_sweep!(st, H, β, sc)
    return nothing
end

# Run the chain at one temperature: thermalize (with step adaptation), freeze, then
# measure. Returns the TempResult.
function _run_temperature!(st::ChainState, H::TiledHamiltonian, kt::Float64,
                           plan::UpdatePlan, observables::Vector{Observable},
                           evaluables::Vector{Evaluable})::TempResult
    β = 1.0 / kt
    sc = SweepScratch(H)
    st.frozen = false
    st.max_drift = 0.0
    for sweep = 1:plan.sweeps_therm
        _compound_sweep!(st, H, β, sc, plan)
        sweep % plan.adapt_interval == 0 && _adapt_step!(st, plan.adapt_target)
        sweep % plan.renorm_interval == 0 && _renormalize!(st, H)
    end
    _renormalize!(st, H)
    st.frozen = true
    st.acc_metro = 0
    st.att_metro = 0
    st.acc_or = 0
    st.att_or = 0

    planned = fld(plan.sweeps_measure, plan.measure_interval)
    accs = [ObsAccumulator(o, planned, plan.nbins) for o in observables]
    for sweep = 1:plan.sweeps_measure
        _compound_sweep!(st, H, β, sc, plan)
        sweep % plan.renorm_interval == 0 && _renormalize!(st, H)
        if sweep % plan.measure_interval == 0
            for acc in accs
                _measure!(acc, st.config, st.energy, H)
            end
        end
    end
    acc_m = st.att_metro == 0 ? NaN : st.acc_metro / st.att_metro
    acc_o = st.att_or == 0 ? NaN : st.acc_or / st.att_or
    stats = _finalize_stats(accs, evaluables, kt, H.n_sites)
    return TempResult(kt, kt / KB_EV, stats, acc_m, acc_o, st.step, st.max_drift)
end

"""
    run_mc(H::TiledHamiltonian; temperature = nothing, kT = nothing, kwargs...)
        -> MCResult

Run single-spin Metropolis Monte Carlo on the tiled Hamiltonian at one absolute
temperature or a ladder of them. Provide **exactly one** of `temperature` (kelvin,
converted with [`KB_EV`](@ref) — assumes an eV-fitted model) or `kT` (`k_B·T` in the
model's energy units), scalar or collection. A collection runs **in the given
order** with the chain carried over (fresh thermalization at each value), so
ordering high → low is an annealing run; pass `carryover = false` for an
independent random restart per temperature.

# Keyword arguments
- `sweeps_therm = 2_000`: equilibration sweeps per temperature (one sweep =
  `n_sites` single-spin attempts). The proposal step adapts only here.
- `sweeps_measure = 10_000`: measurement sweeps per temperature.
- `measure_interval = 1`: measure every k-th sweep.
- `or_per_metropolis = 0`: overrelaxation sweeps mixed after each Metropolis sweep.
- `step = 0.6`: initial proposal rotation scale (radians).
- `adapt_target = 0.5`, `adapt_interval = 50`: acceptance target and window (in
  sweeps) of the thermalization-only step adaptation; the adapted step is frozen
  during measurement (a history-dependent kernel would bias expectations) and
  reported as `final_step`.
- `renorm_interval = 1_000`: sweeps between renormalize + full-energy re-anchoring
  (drift is recorded in `max_drift`).
- `nbins = 32`: jackknife bins per temperature for the evaluables.
- `observables = standard_observables(H)`, `evaluables = standard_evaluables()`.
- `init = nothing`: chain start — a `3 × n_sites` matrix or a vector of 3-vectors
  (normalized), else uniform random.
- `carryover = true`: carry the chain state across the temperature ladder.
- `seed = 0`: the run is bit-reproducible for a fixed seed.
"""
function run_mc(H::TiledHamiltonian; temperature = nothing, kT = nothing,
                sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                measure_interval::Integer = 1, or_per_metropolis::Integer = 0,
                step::Real = 0.6, adapt_target::Real = 0.5,
                adapt_interval::Integer = 50, renorm_interval::Integer = 1_000,
                nbins::Integer = 32,
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(),
                init = nothing, carryover::Bool = true,
                seed::Integer = 0)::MCResult
    plan = UpdatePlan(resolve_kt(temperature, kT); sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = or_per_metropolis, step = step,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = carryover, seed = seed)
    _check_observables(observables)
    rng = Xoshiro(plan.seed)
    st = ChainState(H, _initial_config(H, init, rng), rng, plan.step0)
    points = Vector{TempResult}(undef, 0)
    for (i, kt) in enumerate(plan.kts)
        if i > 1 && !plan.carryover
            _reset_config!(st, H, _initial_config(H, nothing, rng))
            st.step = plan.step0
        end
        push!(points, _run_temperature!(st, H, kt, plan, observables, evaluables))
    end
    return MCResult(points, copy(st.config), plan.seed)
end

function _check_observables(observables::Vector{Observable})
    isempty(observables) && throw(ArgumentError("the observable list is empty"))
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    return nothing
end

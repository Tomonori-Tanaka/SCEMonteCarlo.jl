# Constrained Monte Carlo (CMC), part 3 of 4: the single-chain driver — the default
# observables and evaluables, `CMCResult`, the constrained mover of run.jl's
# temperature loop, and `run_cmc` for one constraint
# (`docs/specs/constrained-mc.md` D4, D5).

# --- observables ---------------------------------------------------------------------

# The observable function of a constraint-dependent default observable (`:M_par`,
# `:constraint_residual`, `:torque_frame`): the `_bound_to` closure `f`, tagged with
# the Hamiltonian `H` and the resolved constraint `rc` it was built for, so the
# start-time door `_check_cmc_observables` can compare them with the run's without
# evaluating anything. A call is exactly a call of `f`.
struct _ConstraintBound{F} <: Function
    H::TiledHamiltonian
    rc::_ResolvedConstraint
    f::F
end

(b::_ConstraintBound)(config, E, Hrun) = b.f(config, E, Hrun)

# Tolerance of the start-time door, on the direction (the chord `|n̂_obs − n̂_run|`,
# the angle between them to first order) and on the per-site weights (relative to
# `max |w|` of the run). A set built for the run's own constraint differs by
# nothing or by re-normalization ulps (~1e-16); a set built for another one differs
# by the scan's direction spacing. Why 1e-6: on a configuration that satisfies the
# run's constraint, observables built for a direction δ away report
# `:constraint_residual ≈ δ`, while a correctly matched chain keeps
# `|M⊥| ≤ 1e-10 Σ|w|` (the drift bound), i.e. a residual ≤ 2e-9 whenever
# `M∥/Σ|w| ≥ 0.05` (not a weak constraint). 1e-6 sits ~500× above that, and a
# mismatch below it moves `:M_par` by O(δ²) ≤ 1e-12 relative and rotates the
# `:torque_frame` axes by ≤ 1e-6 rad — below any statistical error of a run.
const _CMC_OBS_MATCH_TOL = 1e-6

# The start-time door of a constrained run (`docs/specs/constrained-mc.md` D5):
# every observable of the run's set whose function came from `cmc_observables` must
# have been built for the run's `H` (the same object) and for the run's constraint
# `rc` (direction and per-site weights within `_CMC_OBS_MATCH_TOL`). Observables
# that do not depend on the constraint, and observables of the user's own, are not
# inspected. Throws an `ArgumentError` naming `context`; reads no configuration and
# consumes no RNG.
function _check_cmc_observables(observables::Vector{Observable}, H::TiledHamiltonian,
                                rc::_ResolvedConstraint, context::String)
    for o in observables
        b = o.f
        b isa _ConstraintBound || continue
        b.H === H || throw(ArgumentError(
            "$context: observable :$(o.name) was built for a different " *
            "TiledHamiltonian than the one this run samples; build it with the " *
            "run's own H"))
        orc = b.rc
        Δn = norm(orc.direction - rc.direction)
        # per-site weights (same H, so same length): a weight on an atom with no
        # active site enters no measurement and may differ
        Δw = maximum(abs(x - y) for (x, y) in zip(orc.site_weights, rc.site_weights))
        (Δn <= _CMC_OBS_MATCH_TOL &&
         Δw <= _CMC_OBS_MATCH_TOL * maximum(abs, rc.site_weights)) ||
            throw(ArgumentError(
                "$context: observable :$(o.name) was built by cmc_observables for " *
                "another constraint (n̂ = $(Tuple(orc.direction)), weights = " *
                "$(_weights_str(orc.cell_weights))) than the run's (n̂ = " *
                "$(Tuple(rc.direction)), weights = $(_weights_str(rc.cell_weights))); " *
                "it would measure along the wrong direction or with the wrong " *
                "weights. Build the observables with the run's constraint (over a " *
                "vector of constraints: c -> cmc_observables(H, c))"))
    end
    return nothing
end

# The pole threshold of the torque frame: below this `|ẑ × n̂|` (= sinθ) the azimuth
# of `n̂` is rounding noise (a direction built from the Float64 θ = π carries
# sin θ ≈ 1.2e-16 in its x/y components), and the frame takes its pole convention.
const _CMC_FRAME_POLE_TOL = 1e-12

# The tangent frame `(ê_φ, ê_θ)` of the `:torque_frame` observable at the unit
# direction `n̂` (D5). Away from the poles, `ê_φ = normalize(ẑ × n̂)` and
# `ê_θ = ê_φ × n̂`. By hand, for `n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`:
# `ẑ × n̂ = (−n_y, n_x, 0) = sinθ (−sinφ, cosφ, 0)`, so `ê_φ = (−sinφ, cosφ, 0)`, and
# `ê_φ × n̂ = (cosφ cosθ, sinφ cosθ, −sin²φ sinθ − cos²φ sinθ)
#          = (cosθ cosφ, cosθ sinφ, −sinθ)` — the standard `ê_θ = ∂n̂/∂θ`. The triple
# `(ê_θ, ê_φ, n̂)` is right-handed (`ê_θ × ê_φ = n̂`). At a pole (`|ẑ × n̂|` below
# `_CMC_FRAME_POLE_TOL`) `ê_φ` is `ŷ` made orthogonal to `n̂` and `ê_θ = ê_φ × n̂`: at
# the north pole `(ê_φ, ê_θ) = (ŷ, x̂)`, at the south pole `(ŷ, −x̂)` — in both cases
# the `φ = 0` limit of the frame (`ê_θ(θ → π, φ = 0) = (−1, 0, 0)`), whatever `φ` the
# direction was built from (at θ = 0 the stored direction no longer carries `φ`).
function _cmc_torque_frame(n̂::SVector{3,Float64})::NTuple{2,SVector{3,Float64}}
    a = SVector(-n̂[2], n̂[1], 0.0)          # ẑ × n̂, exact (no arithmetic)
    s = norm(a)
    eφ = if s < _CMC_FRAME_POLE_TOL
        ŷ = SVector(0.0, 1.0, 0.0)
        normalize(ŷ - dot(ŷ, n̂) * n̂)
    else
        a / s
    end
    return (eφ, cross(eφ, n̂))
end

# A memo of the last total torque, shared by the `:torque` and `:torque_frame`
# observables of one `cmc_observables` call so that a measurement evaluates the
# energy gradient once. It returns the stored `T` only for a configuration that is
# bitwise equal (`===` per site) to the one `T` was computed for, so its values are
# those of the pure function `total_torque(H, config)`; the comparison and copy
# cost O(n_sites), against the full gradient they save. Guarded by a lock: a set
# shared by concurrent lanes (`c -> obs`) stays correct, only slower.
struct _TorqueMemo
    lock::ReentrantLock
    config::SpinConfig                         # what `T` was computed for (a copy)
    T::Base.RefValue{SVector{3,Float64}}
end

_TorqueMemo() = _TorqueMemo(ReentrantLock(), SpinConfig(), Ref(zero(SVector{3,Float64})))

function _memo_torque!(memo::_TorqueMemo, H::TiledHamiltonian,
                       config::SpinConfig)::SVector{3,Float64}
    lock(memo.lock)
    try
        c = memo.config
        hit = length(c) == length(config)
        if hit
            @inbounds for s in eachindex(config)
                c[s] === config[s] || (hit = false; break)
            end
        end
        hit && return memo.T[]
        T = total_torque(H, config)
        resize!(c, length(config))
        copyto!(c, config)
        memo.T[] = T
        return T
    finally
        unlock(memo.lock)
    end
end

"""
    cmc_observables(H::TiledHamiltonian, constraint::MagnetizationConstraint)
        -> Vector{Observable}

The default observables of a constrained run ([`run_cmc`](@ref)) on `H` under
`constraint` (conventions: `docs/specs/binning-observables.md` B5; torque
conventions: `docs/specs/constrained-mc.md` D5). Every observable is per cell of
`H`: `n_cells = n_sites ÷ n_cell_atoms` is the number of cells of `H` in the
supercell (the cells of the reduced cell after [`reduce_cell`](@ref)):

- `:energy`, `:energy2` — total energy and its square (model units, `j0` excluded),
  as in [`standard_observables`](@ref);
- `:torque` — the total torque per cell of `H`,
  [`total_torque`](@ref)`(H, config) / n_cells`, Cartesian (3 components;
  unweighted, Zeeman part included). Averaged, it is the free-energy gradient of
  the constrained direction: `⟨T⟩ = −n̂ × ∂f/∂n̂` with `f = F/n_cells`;
- `:M_par` — the constrained magnetization along the constraint,
  `M·n̂ / n_cells` with `M = Σ_{s∈𝒞} w_{a(s)} e_s` (units of the weights per cell of
  `H` — μ_B per cell for `weights = :magmoms`);
- `:constraint_residual` — `|M⊥| / M∥` (radians: the angle between `M` and `n̂`,
  to first order), a rounding-level diagnostic of the pair moves;
- `:torque_frame` — the same torque in the spherical frame of the constraint
  direction, `(ê_φ·T, ê_θ·T, n̂·T) / n_cells` (3 components), from the SAME
  gradient evaluation as `:torque` (one gradient per measurement serves both).
  For `n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`, `ê_φ = (−sinφ, cosφ, 0)` and
  `ê_θ = (cosθ cosφ, cosθ sinφ, −sinθ)`, so averaged
  `⟨ê_φ·T⟩ = −∂f/∂θ`, `⟨ê_θ·T⟩ = (1/sinθ) ∂f/∂φ`, `⟨n̂·T⟩ = 0`. At the poles the
  frame is the `φ = 0` limit (`ê_φ = ŷ`; `ê_θ = x̂` at `+ẑ`, `−x̂` at `−ẑ`). Its error
  bars are binned directly: the error of a projection cannot be built from the
  per-component errors of `:torque` (their covariance is not kept);
- `:torque_zeeman` (only when `H` carries a field, [`has_field`](@ref)) — the
  analytic Zeeman part of `:torque`, `MU_B_EV_T (M_μ × B) / n_cells` with
  `M_μ = Σ_{active s} m_{a(s)} e_s` — always the moments of `H`, whatever the
  constraint weights.

Build them with the SAME `constraint` the run uses: `:M_par`,
`:constraint_residual`, and `:torque_frame` measure along the direction (and with
the weights) of the constraint they were built for. [`run_cmc`](@ref) and
[`resume`](@ref) check this when the run starts and throw an `ArgumentError` for a
set built for another constraint or another `H` (observables of your own are not
inspected). The weights are resolved against `H` here (the errors of the constraint
resolution surface at this call). The measuring functions are bound to this `H`
object: measuring them in a run on any other `TiledHamiltonian` throws an
`ArgumentError`. They hold no state that changes a value: `:torque` and
`:torque_frame` share a lock-guarded memo of the last torque, reused only for a
bitwise-identical configuration. The torque costs one full energy gradient per
measurement; thin it with `measure_interval`.
"""
function cmc_observables(H::TiledHamiltonian,
                         constraint::MagnetizationConstraint)::Vector{Observable}
    rc = _resolve_constraint(H, constraint)
    n_cells = _n_cells(H)
    n̂ = rc.direction
    eφ, eθ = _cmc_torque_frame(n̂)
    memo = _TorqueMemo()
    torque_per_cell(config) = _memo_torque!(memo, H, config) / n_cells
    bound(name, f) = _bound_to(H, "cmc observable :$name", f)
    tagged(name, f) = _ConstraintBound(H, rc, bound(name, f))
    obs = [Observable(:energy, 1, (config, E, Hrun) -> E),
           Observable(:energy2, 1, (config, E, Hrun) -> E^2),
           Observable(:torque, 3, bound(:torque, torque_per_cell)),
           Observable(:M_par, 1,
                      tagged(:M_par,
                             config -> dot(_cmc_magnetization(rc, config),
                                           rc.direction) / n_cells)),
           Observable(:constraint_residual, 1,
                      tagged(:constraint_residual, config -> _cmc_residual(rc, config))),
           Observable(:torque_frame, 3,
                      tagged(:torque_frame,
                             config -> begin
                                 T = torque_per_cell(config)
                                 SVector(dot(eφ, T), dot(eθ, T), dot(n̂, T))
                             end))]
    has_field(H) && push!(obs, Observable(:torque_zeeman, 3,
                                          bound(:torque_zeeman,
                                                config -> _zeeman_torque(H, config) /
                                                          n_cells)))
    return obs
end

# |M⊥| / M∥ of the constrained magnetization (`M` from scratch).
function _cmc_residual(rc::_ResolvedConstraint, config::SpinConfig)::Float64
    M = _cmc_magnetization(rc, config)
    Mpar = dot(M, rc.direction)
    return norm(M - Mpar * rc.direction) / Mpar
end

"""
    cmc_evaluables() -> Vector{Evaluable}

The default evaluables of a constrained run: `:specific_heat` only, with the
definition of [`standard_evaluables`](@ref) (per active site, units of `k_B`). The
fluctuation formula stays exact in the constrained ensemble — its reference measure
does not depend on the temperature — while the susceptibility and the Binder
cumulant change meaning under a fixed direction of `M`, so they are left out.
"""
cmc_evaluables()::Vector{Evaluable} =
    filter(e -> e.name === :specific_heat, standard_evaluables())

# --- the result ----------------------------------------------------------------------

"""
    CMCResult

Result of [`run_cmc`](@ref): `points` (one [`TempResult`](@ref) per temperature, in
run order), the `constraint` — a [`MagnetizationConstraint`](@ref) with the run's
direction (bitwise) and its weights **resolved** against the Hamiltonian (a
per-cell-atom vector, also when the run was given `:auto` / `:uniform` /
`:magmoms`; signed zeros folded to `+0.0`) — the chain's `final_config`, and the
run `seed`.

In a constrained run `acceptance_metropolis` is the accepted fraction of ALL
attempts of the measurement phase: pair attempts (null moves — proposals the
compensation cannot realize — count as rejected attempts) and the single-spin
attempts of weight-0 sites, pooled. `acceptance_or` is `NaN` (no overrelaxation in
CMC). Prints as a summary table: the mean Cartesian torque (`torque_x`,
`torque_y`, `torque_z`) and `M∥`, per cell of `H`. The error bar of a projection of
the torque (onto `ê_θ`, `ê_φ`, or any fixed axis) cannot be built from the
per-component errors (their covariance is not kept): read it from the
`:torque_frame` statistics, which are binned directly.
"""
struct CMCResult
    points::Vector{TempResult}
    constraint::MagnetizationConstraint
    final_config::SpinConfig
    seed::UInt64
end

Base.show(io::IO, r::CMCResult) =
    print(io, "CMCResult(", length(r.points), " temperatures, ",
          length(r.final_config), " sites)")

# The weight vector of a result constraint for printing: in full up to 8 cell atoms,
# else the first 6 and the count.
function _weights_str(w)::String
    w isa Symbol && return ":$w"
    length(w) <= 8 && return string(w)
    return "[" * join((@sprintf("%.4g", x) for x in w[1:6]), ", ") *
           ", … ($(length(w)) cell atoms)]"
end

function Base.show(io::IO, ::MIME"text/plain", r::CMCResult)
    n = length(r.final_config)
    n̂ = r.constraint.direction
    println(io, "CMCResult: ", length(r.points), " temperature(s), ", n,
            " sites, seed ", r.seed)
    println(io, "  constraint n̂ = (", join((@sprintf("%.6g", x) for x in n̂), ", "),
            "), weights = ", _weights_str(r.constraint.weights))
    @printf(io, "  %-11s %-9s %-22s %-11s %-11s %-11s %-9s %-6s\n",
            "kT", "T[K]", "E/site", "torque_x", "torque_y", "torque_z", "M_par",
            "acc")
    for p in r.points
        e = get(p.stats, :energy, nothing)
        estr = e === nothing ? "-" :
               @sprintf("%.6g ± %.2g", e.mean[1] / n, e.err[1] / n)
        t = get(p.stats, :torque, nothing)
        tstr = (t === nothing || length(t.mean) != 3) ? ("-", "-", "-") :
               Tuple(@sprintf("%.4g", t.mean[c]) for c = 1:3)
        @printf(io, "  %-11.5g %-9.4g %-22s %-11s %-11s %-11s %-9s %-6.3f\n",
                p.kT, p.temperature, estr, tstr..., _stat_str(p.stats, :M_par),
                p.acceptance_metropolis)
    end
    print(io, "  (torque and M_par per cell of H, torque Cartesian; error bars of " *
              "torque projections: :torque_frame)")
    return nothing
end

# --- the constrained mover -----------------------------------------------------------

# The constrained mover of the single-chain driver (run.jl): the CMC sweep under the
# resolved constraint `rc` with its lane-private `scratch`, and the constrained
# start `start` that a `carryover = false` ladder returns to at every temperature —
# a private copy, so the chain can never alias it.
struct _CMCMover <: _Mover
    rc::_ResolvedConstraint
    scratch::_CMCScratch
    start::SpinConfig

    function _CMCMover(H::TiledHamiltonian, rc::_ResolvedConstraint,
                       scratch::_CMCScratch, start::SpinConfig)
        @assert length(scratch.lc) == H.nlm "CMC scratch built for another Hamiltonian"
        @assert(length(rc.site_weights) == H.n_sites,
                "constraint resolved for another Hamiltonian")
        length(start) == H.n_sites || throw(DimensionMismatch(
            "start has $(length(start)) sites; expected $(H.n_sites)"))
        return new(rc, scratch, copy(start))
    end
end

# One CMC sweep (D3). The sweep's returned `M` is dropped: every sweep recomputes
# it from the configuration, so the chain is a function of (config, streams) alone.
function _mover_sweep!(mover::_CMCMover, st::ChainState, H::TiledHamiltonian,
                       β::Float64, scs::Vector{SweepScratch}, plan::UpdatePlan)
    _cmc_sweep!(st, H, β, scs[1], mover.rc, mover.scratch)
    return nothing
end

# The configuration of the `carryover = false` restart (`_restart_chain!`, run.jl):
# back to the constrained start (the default pattern or the rotated `init`) —
# deterministic, and no RNG is consumed (D4).
function _mover_reinit!(mover::_CMCMover, st::ChainState, H::TiledHamiltonian)
    _reset_config!(st, H, mover.start)
    return nothing
end

# The two end-of-thermalization checks (D3), once per temperature.
function _mover_after_thermalization(mover::_CMCMover, st::ChainState,
                                     H::TiledHamiltonian, kT::Float64)
    _cmc_check_step_floor(st, kT, mover.rc.direction)
    _cmc_check_weak_constraint(mover.rc, st.config, kT)
    return nothing
end

# The result: `_constraint_record` (constraint.jl) carries the direction bitwise.
_mover_result(mover::_CMCMover, points::Vector{TempResult}, st::ChainState,
              plan::UpdatePlan)::CMCResult =
    CMCResult(points, _constraint_record(mover.rc), copy(st.config), plan.seed)

# --- the single-chain driver ---------------------------------------------------------

"""
    run_cmc(H::TiledHamiltonian, constraint::MagnetizationConstraint;
            temperature = nothing, kT = nothing, kwargs...) -> CMCResult

Constrained Monte Carlo (`docs/specs/constrained-mc.md`): sample the canonical
ensemble of `H` restricted to a fixed direction `n̂` of the weighted magnetization
of `constraint` (see [`MagnetizationConstraint`](@ref)), and measure the torque,
whose average is the free-energy gradient — the finite-temperature magnetic
anisotropy. Per cell of `H`, `⟨T⟩ = −n̂ × ∂f/∂n̂` with `f = F/n_cells` (the
`:torque` observable; `F` the free energy of the direction of `M`). Provide
**exactly one** of `temperature` (kelvin, converted with [`KB_EV`](@ref) — assumes
an eV-fitted model) or `kT` (model energy units), scalar or collection; a
collection runs **in the given order** with the chain carried over (an annealing
ladder when ordered high → low), or restarted from the constrained start at each
temperature with `carryover = false`.

One call samples ONE constraint direction. To scan directions, repeat the call or
pass a vector of constraints, which runs them as independent lanes over threads:

    run_cmc(H, constraints::AbstractVector{MagnetizationConstraint};
            observables = c -> cmc_observables(H, c), ntasks = nothing, seed, …)
        -> Vector{CMCResult}

One sweep is one attempt per **active** site, in site order, serial: a constrained
site (nonzero weight) is the source of a pair move with a uniformly drawn partner
that compensates the transverse change of `M` (Asselin et al., PRB 82, 054415); a
weight-0 active site gets a plain single-spin Metropolis attempt; inactive sites
stay frozen. The run is bit-reproducible for a fixed `seed`, independently of the
thread count.

```julia
H = TiledHamiltonian(model; dims = (4, 4, 4))
r = run_cmc(H, MagnetizationConstraint(π / 4, 0.0); temperature = 300.0,
            measure_interval = 10, seed = 1)
r.points[1].stats[:torque].mean    # ⟨T⟩ per cell, Cartesian (model energy units)
```

# Keyword arguments
- `sweeps_therm = 2_000`, `sweeps_measure = 10_000`, `measure_interval = 1`,
  `step = 0.6`, `adapt_target = 0.5`, `adapt_interval = 50`,
  `renorm_interval = 1_000`, `nbins = 32`, `seed = rand(UInt64)` — as for
  [`run_mc`](@ref). The acceptance that drives the thermalization-only step
  adaptation counts null moves as rejected attempts (see [`CMCResult`](@ref)).
  Each `:torque` measurement costs one full energy gradient — about one sweep — so
  a `measure_interval` of a few sweeps is usually the better use of the time.
- `observables = cmc_observables(H, constraint)`, `evaluables = cmc_evaluables()`.
  A set from [`cmc_observables`](@ref) must be built for the same `H` and the same
  `constraint` (direction and resolved weights): a set built for another one is
  refused with an `ArgumentError` before the run starts.
- `init = nothing`: the constrained start. `nothing` — every constrained site along
  `sign(w) n̂`, every other site along `n̂` (deterministic, no RNG); otherwise a
  `3 × n_sites` matrix or a vector of 3-vectors, rotated rigidly so that its
  weighted magnetization points along `n̂` (refused when that magnetization is
  below `1e-8 Σ|w|`). Pass a ground state ([`find_ground_state`](@ref)) for a
  model with antiparallel sublattices.
- `carryover = true`: carry the chain across the temperature ladder; `false`
  returns to the constrained start at every temperature (no RNG is consumed).
- `checkpoint = nothing`, `checkpoint_interval = 0`: a file path to write
  restartable checkpoints to, and the sweeps between periodic writes (`0` ⇒ only
  at temperature boundaries) — as for [`run_mc`](@ref). The file (kind `"cmc"`,
  `docs/specs/checkpoint-schema.md`) also stores the constraint (direction
  bitwise, resolved weights) and the constrained start; continue with
  [`resume`](@ref)`(path, H)`, which returns the full run's `CMCResult`,
  bit-identical to an uninterrupted run.
- `or_per_metropolis`, `sweep_tasks`: not supported — giving either throws an
  `ArgumentError`. There is no overrelaxation in CMC, and the sweep is serial (the
  pair move couples every pair through `M∥`, so colour classes do not commute).

At the end of every temperature's thermalization the run warns when the adapted
step sits at its lower clamp (many null moves — a large weight ratio) and when the
constraint is weak, `M∥ / Σ|w| < 0.05` (above the ordering temperature, or
nearly cancelling weights); every warning names `kT` and `n̂`.
"""
function run_cmc(H::TiledHamiltonian, constraint::MagnetizationConstraint;
                 temperature = nothing, kT = nothing,
                 sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                 measure_interval::Integer = 1, step::Real = 0.6,
                 adapt_target::Real = 0.5, adapt_interval::Integer = 50,
                 renorm_interval::Integer = 1_000, nbins::Integer = 32,
                 observables::Vector{Observable} = cmc_observables(H, constraint),
                 evaluables::Vector{Evaluable} = cmc_evaluables(),
                 init = nothing, carryover::Bool = true,
                 seed::Integer = rand(UInt64),
                 checkpoint::Union{Nothing,AbstractString} = nothing,
                 checkpoint_interval::Integer = 0,
                 or_per_metropolis = nothing, sweep_tasks = nothing)::CMCResult
    _cmc_refuse_unsupported(or_per_metropolis, sweep_tasks)
    plan = UpdatePlan(resolve_kt(temperature, kT); sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval, or_per_metropolis = 0,
                      step = step, adapt_target = adapt_target,
                      adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = carryover, sweep_tasks = 1, seed = seed)
    _check_observables(observables)
    _check_evaluables(observables, evaluables)
    rc, start = _cmc_setup(H, constraint, init)
    _check_cmc_observables(observables, H, rc, "run_cmc")
    # the file stores the constraint record and the start: a resumed
    # `carryover = false` ladder needs the start, which for a user `init` cannot be
    # rebuilt from the rest of the file (docs/specs/checkpoint-schema.md, kind
    # "cmc")
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "cmc", 0; constraint = _constraint_record(rc),
                            start = start)
    mover = _CMCMover(H, rc, _CMCScratch(H), start)
    st = ChainState(H, copy(start), Xoshiro(plan.seed), plan.step0)
    return _mc_loop!(mover, TempResult[], st, H, plan, observables, evaluables, 1,
                     :therm, 0, nothing, ck)
end

# The two keywords of `run_mc` that constrained MC refuses, with the reason (D4);
# shared by the single-chain and the lane method.
function _cmc_refuse_unsupported(or_per_metropolis, sweep_tasks)
    or_per_metropolis === nothing || throw(ArgumentError(
        "run_cmc: or_per_metropolis is not supported — constrained MC has no " *
        "overrelaxation move (a reflection would change M and break the constraint)"))
    sweep_tasks === nothing || throw(ArgumentError(
        "run_cmc: sweep_tasks is not supported — the constrained sweep is serial (the " *
        "pair move couples every pair through M∥); parallelize over constraint " *
        "directions instead"))
    return nothing
end

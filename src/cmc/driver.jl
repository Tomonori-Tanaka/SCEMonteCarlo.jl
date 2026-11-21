# Constrained Monte Carlo (CMC), part 3 of 4: the single-chain driver — the default
# observables and evaluables, `CMCResult`, the constrained mover of run.jl's
# temperature loop, and `run_cmc` for one constraint
# (`docs/specs/constrained-mc.md` D4, D5).

# --- observables ---------------------------------------------------------------------

"""
    cmc_observables(H::TiledHamiltonian, constraint::MagnetizationConstraint)
        -> Vector{Observable}

The default observables of a constrained run ([`run_cmc`](@ref)) on `H` under
`constraint` (conventions: `docs/specs/binning-observables.md` B5; torque
conventions: `docs/specs/constrained-mc.md` D5). `n_cells` is the number of cells of
`H` in the supercell (`n_sites / n_cell_atoms`; the cells of the reduced cell after
[`reduce_cell`](@ref)):

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
- `:torque_zeeman` (only when `H` carries a field, [`has_field`](@ref)) — the
  analytic Zeeman part of `:torque`, `MU_B_EV_T (M_μ × B) / n_cells` with
  `M_μ = Σ_{active s} m_{a(s)} e_s` — always the moments of `H`, whatever the
  constraint weights.

Build them with the SAME `constraint` the run uses: `:M_par` and
`:constraint_residual` measure along the direction and with the weights of the
constraint they were built for, and nothing checks that it is the run's (a mismatch
gives wrong values, not an error). The weights are resolved against `H` here (the
errors of the constraint resolution surface at this call). The measuring functions
are pure — no mutable state is captured — and bound to this `H` object: measuring
them in a run on any other `TiledHamiltonian` throws an `ArgumentError`. The torque
costs one full energy gradient per measurement; thin it with `measure_interval`.
"""
function cmc_observables(H::TiledHamiltonian,
                         constraint::MagnetizationConstraint)::Vector{Observable}
    rc = _resolve_constraint(H, constraint)
    n_cells = _n_cells(H)
    bound(name, f) = _bound_to(H, "cmc observable :$name", f)
    obs = [Observable(:energy, 1, (config, E, Hrun) -> E),
           Observable(:energy2, 1, (config, E, Hrun) -> E^2),
           Observable(:torque, 3,
                      bound(:torque, config -> total_torque(H, config) / n_cells)),
           Observable(:M_par, 1,
                      bound(:M_par,
                            config -> dot(_cmc_magnetization(rc, config),
                                          rc.direction) / n_cells)),
           Observable(:constraint_residual, 1,
                      bound(:constraint_residual, config -> _cmc_residual(rc, config)))]
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
CMC). Prints as a summary table (torque and `M∥` per cell of `H`).
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
            "kT", "T[K]", "E/site", "T_x/cell", "T_y/cell", "T_z/cell", "M_par",
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
  Observables of your own must be built for the same `constraint` (see
  [`cmc_observables`](@ref)).
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

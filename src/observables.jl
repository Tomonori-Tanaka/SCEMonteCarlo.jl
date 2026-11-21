# Composable observables (see `docs/specs/binning-observables.md` for the C/χ/U
# conventions — that file is authoritative).
#
# An `Observable` is measured every measurement sweep and accumulated in a
# `LogBinner` (mean/error/τ_int) and a `BinStore` (bin means for jackknife). An
# `Evaluable` is a derived quantity — a function of the means of scalar observables —
# jackknifed over the stored bins. Nothing here knows about Markov chains; the run
# drivers feed `(config, energy, H)` in.

"""
    Observable(name::Symbol, ncomp::Integer, f)

A raw observable measured on every stored sweep: `f(config, energy, H)` returns a
`Real` (`ncomp == 1`) or an `ncomp`-component vector. `energy` is the configuration's
current total SCE energy (model units, `j0` excluded) — so `:energy` costs nothing
extra. Accumulated with autocorrelation-aware errors ([`LogBinner`](@ref)) and bin
means for derived [`Evaluable`](@ref)s.
"""
struct Observable
    name::Symbol
    ncomp::Int
    f::Function

    function Observable(name::Symbol, ncomp::Integer, f)
        ncomp >= 1 || throw(ArgumentError("ncomp must be ≥ 1; got $ncomp"))
        return new(name, ncomp, f)
    end
end

"""
    Evaluable(name::Symbol, inputs::Vector{Symbol}, f)

A derived quantity `f(means::NamedTuple, kT, n) -> Real` of the means of
**scalar** raw observables named in `inputs` (e.g. specific heat from `:energy` and
`:energy2`); `n` is the number of **active** (magnetic) sites, so per-site
quantities are per active site. Estimated by leave-one-bin-out [`jackknife`](@ref)
over the stored bin means, which propagates the nonlinearity correctly.
"""
struct Evaluable
    name::Symbol
    inputs::Vector{Symbol}
    f::Function

    function Evaluable(name::Symbol, inputs::Vector{Symbol}, f)
        isempty(inputs) && throw(ArgumentError("an Evaluable needs ≥ 1 input"))
        return new(name, copy(inputs), f)
    end
end

"""
    ObservableStat

One observable's finalized statistics at one temperature: component-wise `mean`,
standard `err`, and integrated autocorrelation time `tau_int` (`NaN` for jackknifed
evaluables), plus the measurement `count` (raw) or bin count (evaluables).
"""
struct ObservableStat
    name::Symbol
    mean::Vector{Float64}
    err::Vector{Float64}
    tau_int::Vector{Float64}
    count::Int
end

Base.show(io::IO, s::ObservableStat) =
    length(s.mean) == 1 ?
    print(io, "ObservableStat(", s.name, " = ", @sprintf("%.6g ± %.2g", s.mean[1],
          s.err[1]), ")") :
    print(io, "ObservableStat(", s.name, ", ", length(s.mean), " comps)")

"""
    standard_observables(H::TiledHamiltonian) -> Vector{Observable}

The standard set: `:energy`, `:energy2` (total, model units), the magnetization
vector `:m = Σ_s e_s / n_active` (**active sites only** — an inactive, non-magnetic
site's frozen direction is not a magnetic moment), `:absm = |m|`, its powers `:m2`,
`:m4`, and the per-sublattice magnetization `:sublattice_m` (training-cell atom `a`'s
cell-averaged vector, flattened `(x₁,y₁,z₁, x₂,…)`, `3·n_cell_atoms` components;
inactive sublattices report exactly zero). Spin directions only — magnetic-moment
magnitudes are not part of the fitted model. When `H` carries `magmoms` the set
grows by `:M = Σ_active m_{a(s)} e_s / n_cells` (3-vector, μ_B per cell of `H` —
the reduced cell after [`reduce_cell`](@ref); the `:sublattice_m` normalization)
and, with a nonzero field ([`has_field`](@ref)), `:M_B = M · B̂`
(docs/specs/zeeman-field.md).
"""
function standard_observables(H::TiledHamiltonian)::Vector{Observable}
    obs = [Observable(:energy, 1, (cfg, E, H) -> E),
           Observable(:energy2, 1, (cfg, E, H) -> E^2),
           Observable(:m, 3, _mean_spin),
           Observable(:absm, 1, (cfg, E, H) -> norm(_mean_spin(cfg, E, H))),
           Observable(:m2, 1, (cfg, E, H) -> sum(abs2, _mean_spin(cfg, E, H))),
           Observable(:m4, 1, (cfg, E, H) -> sum(abs2, _mean_spin(cfg, E, H))^2),
           Observable(:sublattice_m, 3 * H.n_cell_atoms, _sublattice_m)]
    if H.magmoms !== nothing
        push!(obs, Observable(:M, 3, _moment_sum))
        has_field(H) && push!(obs, Observable(:M_B, 1, _moment_along_field))
    end
    return obs
end

# Magnetization in μ_B per cell of `H`: Σ_active m_{a(s)} e_s / n_cells. Active
# sites only, like `_sublattice_m` — a frozen direction is not a moment (B3). With a
# field every moment-carrying site owns a Zeeman instance and is active, so nothing
# is dropped; without one a moment-carrying SCE-inactive sublattice stays excluded.
function _moment_sum(config::SpinConfig, E, H::TiledHamiltonian)::SVector{3,Float64}
    H.magmoms === nothing && return zero(SVector{3,Float64})
    return _moment_total(H, config) / _n_cells(H)
end

# The total moment `M_μ = Σ_{active s} m_{a(s)} e_s` (μ_B; no `1/n_cells`), summed in
# site order — shared by `:M` and the analytic Zeeman torque, so the two agree
# bitwise. Precondition (the callers'): `H.magmoms !== nothing` and
# `length(config) == H.n_sites`. `@inline` is load-bearing: LLVM may contract the
# multiply-adds of the callers' follow-up arithmetic (`dot` in `:M_B`) differently
# across a call boundary, and inlined is how the sum compiled before it was shared
# (an outlined version moves `:M_B` by an ulp in about 30 % of configurations).
@inline function _moment_total(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    mm = H.magmoms::Vector{Float64}
    M = zero(SVector{3,Float64})
    for s = 1:H.n_sites
        H.site_active[s] || continue
        M += mm[site_atom(H, s)] * config[s]
    end
    return M
end

_moment_along_field(config::SpinConfig, E, H::TiledHamiltonian)::Float64 =
    dot(_moment_sum(config, E, H), normalize(H.field))

"""
    total_torque(H::TiledHamiltonian, config::SpinConfig) -> SVector{3,Float64}

Total torque of the full system, `T = Σ_{s active} G_s × e_s = −Σ_s e_s × ∂E/∂e_s`,
with `G` the tangent-projected gradient of [`energy_gradient!`](@ref) — so the
Zeeman part of a field-carrying `H` is included, and every term counts with unit
weight (never the constraint weights). Model energy units (eV for an eV-fitted
model); a full-system sum, not per cell or per site.

Meaning: for a rigid rotation of every spin by the angle `α` about a unit axis `w`,
`T · w = −dE/dα`. Averaged in the constrained ensemble it is the free-energy
gradient `⟨T⟩ = −n̂ × ∂F/∂n̂` (`docs/specs/constrained-mc.md`). Inactive sites have
`G_s = 0` and contribute nothing. For an isotropic (rotation-invariant) model the
sum vanishes configuration by configuration.

One call costs about one Metropolis sweep (it rebuilds the tesseral rows and
allocates the gradient vector).
"""
function total_torque(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    G = energy_gradient!(Vector{SVector{3,Float64}}(undef, H.n_sites), H, config)
    T = zero(SVector{3,Float64})
    for s = 1:H.n_sites
        H.site_active[s] || continue
        T += cross(G[s], config[s])
    end
    return T
end

# Analytic Zeeman torque of the full system, `MU_B_EV_T · (M_μ × B)` with
# `M_μ = Σ_{active s} m_{a(s)} e_s` (μ_B; no `1/n_cells`). Sign: the Zeeman energy
# is `E_Z = −MU_B_EV_T Σ m (e·B)`; a global rotation by `α` about the unit axis `w`
# moves `e → e + α w × e`, so `dE_Z/dα = −MU_B_EV_T Σ m (w × e)·B
# = −w · MU_B_EV_T (M_μ × B)`, and the torque convention `T·w = −dE/dα`
# (`docs/specs/constrained-mc.md` D5) gives `T_Z = +MU_B_EV_T (M_μ × B)` — the
# moment is pulled toward `B`. Zero without a field. This is the part of
# [`total_torque`](@ref) the external field contributes, which partial
# (field-free) Hamiltonians leave out.
function _zeeman_torque(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    (H.magmoms === nothing || !has_field(H)) && return zero(SVector{3,Float64})
    return MU_B_EV_T * cross(_moment_total(H, config), H.field)
end

# The door of an observable bound to the `H` object it was built for: measuring it
# in a run on any other `TiledHamiltonian` (even an equal one) throws an
# `ArgumentError` naming `label`.
function _check_bound(H::TiledHamiltonian, Hrun, label::String)
    Hrun === H || throw(ArgumentError(
        "$label was built for a different TiledHamiltonian than the one this run " *
        "samples; build it with the run's own H"))
    return nothing
end

# An observable function `(config, E, Hrun) -> f(config)` bound to the `H` object it
# was built for (`_check_bound`). Captures no mutable state.
_bound_to(H::TiledHamiltonian, label::String, f) =
    (config, E, Hrun) -> begin
        _check_bound(H, Hrun, label)
        f(config)
    end

# --- torque observables: the public call and the chain-state path ------------------
#
# The torque observables (`:torque` and `:torque_frame` of `cmc_observables`, and
# `torque_observable`) are `_TorqueObservable`s. Called as a function
# `(config, E, H)` — the public `Observable` contract, used by `run_pt`,
# `gpu_run_pt`, a wrapper of the user's, or a direct call — each one evaluates
# `total_torque` of its own Hamiltonian from scratch. The single-chain measurement of
# `run_mc` / `run_cmc` (`_measure_chain!`, run.jl) evaluates them through
# `_observe` instead: from the chain's own tesseral rows (column `s` = `Z_lm(e_s)`,
# kept bitwise equal to a fresh `_zrows` of the configuration by every update,
# renormalization, and reset) with the chain's `_MeasureScratch` — one
# allocation-free gradient pass per Hamiltonian and measurement, shared by every
# torque observable of that Hamiltonian (`:torque` and `:torque_frame` share one).
# The two paths are bitwise equal (`_torque_from_rows` repeats the arithmetic of
# `total_torque`: the same `_site_grad` per active site, the same site-order sum).

# A torque observable bound to the run's `H`: `project(T / n_cells)` with
# `T = total_torque(Hgrad, config)` — `Hgrad` is `H` itself or a term subset of it
# (same sites), `project` is `identity` or the frame projection of `:torque_frame`.
struct _TorqueObservable{P} <: Function
    H::TiledHamiltonian
    Hgrad::TiledHamiltonian
    n_cells::Int
    project::P
    label::String
end

function (o::_TorqueObservable)(config, E, Hrun)
    _check_bound(o.H, Hrun, o.label)
    return o.project(total_torque(o.Hgrad, config) / o.n_cells)
end

# The full-system torque of `Hgrad` from precomputed tesseral rows `zrows` (column
# `s` = `Z_lm(config[s])`, at least `Hgrad.nlm` rows in `lm_index` order) and
# caller-owned scratch `c` (length `Hgrad.nlm`) and `plm` (length ≥ `Hgrad.lmax + 1`).
# Bitwise `total_torque(Hgrad, config)` when `zrows` are the rows of `config`: the
# row values do not depend on the `lmax` they were tabulated with (one independent
# `Zlm_unsafe` per `(l, m)`), and the loop is that of `total_torque` over
# `energy_gradient!` without the gradient vector. Allocation-free.
function _torque_from_rows(Hgrad::TiledHamiltonian, config::SpinConfig,
                           zrows::Matrix{Float64}, c::Vector{Float64},
                           plm::Vector{Float64})::SVector{3,Float64}
    T = zero(SVector{3,Float64})
    for s = 1:Hgrad.n_sites
        Hgrad.site_active[s] || continue
        T += cross(_site_grad(Hgrad, s, config[s], zrows, c, plm), config[s])
    end
    return T
end

# One gradient slot of a `_MeasureScratch`: the scratch of a pass over `Hgrad` and
# the torque it gave at the measurement `stamp`. `evaluations` counts the passes
# (the one-pass-per-measurement contract is tested on it).
mutable struct _TorqueSlot
    const Hgrad::TiledHamiltonian
    const c::Vector{Float64}
    const plm::Vector{Float64}
    T::SVector{3,Float64}
    stamp::Int
    evaluations::Int
end

# The measurement scratch of ONE chain (created per chain and temperature by the
# single-chain driver, never shared between chains): the measurement counter
# `stamp`, advanced once per measurement, and one torque slot per Hamiltonian
# measured. Holds no state that outlives a measurement — a slot is valid only for
# the `stamp` it was computed at — so nothing of it is checkpointed.
mutable struct _MeasureScratch
    stamp::Int
    const slots::Vector{_TorqueSlot}
end

_MeasureScratch() = _MeasureScratch(0, _TorqueSlot[])

function _torque_slot!(ms::_MeasureScratch, Hgrad::TiledHamiltonian)::_TorqueSlot
    for slot in ms.slots
        slot.Hgrad === Hgrad && return slot
    end
    slot = _TorqueSlot(Hgrad, zeros(Hgrad.nlm), Vector{Float64}(undef, Hgrad.lmax + 1),
                       zero(SVector{3,Float64}), ms.stamp - 1, 0)
    push!(ms.slots, slot)
    return slot
end

# The torque of `Hgrad` on the chain whose configuration is `config`, with tesseral
# rows `zrows` of the run's `H`, at the current measurement of `ms`: computed once
# per measurement and Hamiltonian, then reused. A Hamiltonian whose rows are not a
# prefix of `H`'s (another site count, a larger `lmax` — `torque_observable` admits
# neither) takes the public path.
function _chain_torque!(ms::_MeasureScratch, Hgrad::TiledHamiltonian,
                        H::TiledHamiltonian, config::SpinConfig,
                        zrows::Matrix{Float64})::SVector{3,Float64}
    (Hgrad.n_sites == H.n_sites && Hgrad.lmax <= H.lmax) ||
        return total_torque(Hgrad, config)
    slot = _torque_slot!(ms, Hgrad)
    if slot.stamp != ms.stamp
        slot.T = _torque_from_rows(Hgrad, config, zrows, slot.c, slot.plm)
        slot.stamp = ms.stamp
        slot.evaluations += 1
    end
    return slot.T
end

# One observable's value at a measurement of a single chain (`_measure_chain!`):
# `f(config, energy, H)` — except for a torque observable, read from the chain state.
_observe(f, config::SpinConfig, energy::Float64, zrows::Matrix{Float64},
         H::TiledHamiltonian, ms::_MeasureScratch) = f(config, energy, H)

function _observe(o::_TorqueObservable, config::SpinConfig, energy::Float64,
                  zrows::Matrix{Float64}, H::TiledHamiltonian, ms::_MeasureScratch)
    _check_bound(o.H, H, o.label)
    return o.project(_chain_torque!(ms, o.Hgrad, H, config, zrows) / o.n_cells)
end

# Value identity of two templates on every field that fixes their energy
# (atoms, shifts, ls, folded, coef). `ScaledTerm` holds arrays, so the default
# `==` (identity) would never match two separately constructed templates.
_term_key(t::ScaledTerm) = (t.coef, t.atoms, t.shifts, t.ls, t.folded)

"""
    torque_observable(name::Symbol, Hpart::TiledHamiltonian, H::TiledHamiltonian)
        -> Observable

A 3-component [`Observable`](@ref) named `name` that measures the torque of the
term-subset Hamiltonian `Hpart` on the configuration of a run on `H`:
`total_torque(Hpart, config) / n_cells` (Cartesian, per cell of `H`, unweighted —
the normalization of `:torque`; see [`total_torque`](@ref) and
`docs/specs/constrained-mc.md`).

Build `Hpart` with the public constructor from the SAME term list that built `H`
(after [`reduce_cell`](@ref): `red.terms`), filtered to the subset of interest and
without a field — e.g. `TiledHamiltonian(n, filter(t -> t.body == 1, terms); dims)`
for the on-site part. Over a disjoint cover of the fitted terms the partial
torques plus the analytic Zeeman torque sum to the total torque, configuration by
configuration; an individual partial torque is a free-energy gradient only to
first order in the anisotropy.

Doors (`ArgumentError`, every message naming the argument order — the term subset
`Hpart` second, the run's `H` third): `Hpart` and `H` must agree on
`n_cell_atoms`, `dims`, and `n_sites`; `Hpart` must carry no field (the Zeeman part
is measured analytically);
every template of `Hpart` must equal — in atoms, shifts, `ls`, `folded`, and the
scaled coefficient — one of the fitted templates `H.terms[1:H.n_fitted_terms]`, so
a subset of a different term list (another cell, another fit) is refused even when
the atom counts happen to agree. Disjointness and coverage are not enforced.

The observable is bound to the `H` object it was built with: measuring it in a run
on any other `TiledHamiltonian` (even an equal one) throws an `ArgumentError`, so
build it with the same `H` the run uses. `Hpart` and `H` are immutable and the
returned function captures no mutable state, so the observable may be shared across
lanes that sample the same `H`.

Cost: one gradient pass over the terms of `Hpart` per measurement. In
[`run_mc`](@ref) and [`run_cmc`](@ref) it reads the chain's cached tesseral rows
and allocates nothing (bitwise the value of a direct call, which rebuilds the rows
and allocates the gradient vector); see `measure_interval` of [`run_cmc`](@ref)
for measured costs.
"""
function torque_observable(name::Symbol, Hpart::TiledHamiltonian,
                           H::TiledHamiltonian)::Observable
    # every refusal names the argument order: swapping Hpart and H is the easy mistake
    door = "torque_observable(name, Hpart, H) — Hpart the term subset, H the run's " *
           "Hamiltonian:"
    Hpart.n_cell_atoms == H.n_cell_atoms || throw(ArgumentError(
        "$door Hpart has $(Hpart.n_cell_atoms) cell atoms, H has $(H.n_cell_atoms)"))
    Hpart.dims == H.dims || throw(ArgumentError(
        "$door Hpart.dims = $(Tuple(Hpart.dims)) differs from " *
        "H.dims = $(Tuple(H.dims))"))
    Hpart.n_sites == H.n_sites || throw(ArgumentError(
        "$door Hpart has $(Hpart.n_sites) sites, H has $(H.n_sites)"))
    has_field(Hpart) && throw(ArgumentError(
        "$door Hpart carries an external field; build partial Hamiltonians " *
        "field-free (the Zeeman torque is measured analytically)"))
    fitted = Set(_term_key(t) for t in view(H.terms, 1:H.n_fitted_terms))
    for (k, t) in enumerate(Hpart.terms)
        _term_key(t) in fitted || throw(ArgumentError(
            "$door template $k of Hpart (atoms = $(t.atoms), shifts = $(t.shifts), " *
            "ls = $(t.ls)) matches no fitted template of H; build Hpart from the " *
            "same term list as H"))
    end
    return Observable(name, 3,
                      _TorqueObservable(H, Hpart, _n_cells(H), identity,
                                        "torque_observable $name"))
end

function _mean_spin(config::SpinConfig, E, H::TiledHamiltonian)::SVector{3,Float64}
    # All-active fast path: pairwise `sum` — byte-identical to the pre-inactive-site
    # convention (a sequential loop differs by ULPs on large lattices) and slightly
    # more accurate than sequential accumulation.
    H.n_active == H.n_sites && return sum(config) / H.n_active
    m = zero(SVector{3,Float64})
    @inbounds for s in eachindex(config)
        H.site_active[s] && (m += config[s])
    end
    return m / H.n_active
end

function _sublattice_m(config::SpinConfig, E, H::TiledHamiltonian)::Vector{Float64}
    out = zeros(3, H.n_cell_atoms)
    for s in eachindex(config)
        H.site_active[s] || continue     # inactive sublattices stay exactly zero
        a = mod1(s, H.n_cell_atoms)
        e = config[s]
        out[1, a] += e[1]
        out[2, a] += e[2]
        out[3, a] += e[3]
    end
    return vec(out) ./ _n_cells(H)
end

"""
    standard_evaluables() -> Vector{Evaluable}

The standard derived quantities (conventions: `docs/specs/binning-observables.md`):

- `:specific_heat` — per active site, in units of ``k_B``:
  ``C/k_B = (⟨E²⟩ − ⟨E⟩²) / (n_{active}\\, (k_BT)²)`` (intensive — comparable across
  supercell sizes).
- `:susceptibility` — |m|-connected, per active site:
  ``χ = n_{active} (⟨m²⟩ − ⟨|m|⟩²) / k_BT``. On a finite system with continuous
  symmetry ``⟨\\boldsymbol m⟩ = 0`` exactly, so the naive connected form degenerates
  and grows ∝ N below the transition; the |m|-connected form peaks at it (the
  finite-size-scaling standard).
- `:binder` — the plain ratio ``U = ⟨m⁴⟩/⟨m²⟩²`` (→ 1 ordered, → 5/3 disordered for
  3-component spins; U(T) crossings locate ``T_c``).
"""
function standard_evaluables()::Vector{Evaluable}
    return [Evaluable(:specific_heat, [:energy, :energy2],
                      (m, kT, n) -> (m.energy2 - m.energy^2) / (n * kT^2)),
            Evaluable(:susceptibility, [:m2, :absm],
                      (m, kT, n) -> n * (m.m2 - m.absm^2) / kT),
            Evaluable(:binder, [:m2, :m4], (m, kT, n) -> m.m4 / m.m2^2)]
end

# --- accumulation ------------------------------------------------------------------

# One observable's accumulators for one temperature / lane.
struct ObsAccumulator
    obs::Observable
    binner::LogBinner
    store::BinStore
    val::Vector{Float64}    # scratch: the current measurement
end

# `planned` = number of measurements this accumulator will receive; the bin size is
# fixed up front so every bin is equal-weight (a trailing remainder is dropped).
function ObsAccumulator(obs::Observable, planned::Integer, nbins::Integer)
    bin_size = max(1, fld(planned, nbins))
    return ObsAccumulator(obs, LogBinner(obs.ncomp),
                          BinStore(obs.ncomp, bin_size, nbins), zeros(obs.ncomp))
end

_measure!(acc::ObsAccumulator, config::SpinConfig, energy::Float64,
          H::TiledHamiltonian) = _record!(acc, acc.obs.f(config, energy, H))

# Check one measured value `v` against the observable's declared components and
# accumulate it.
function _record!(acc::ObsAccumulator, v)
    if v isa Real
        acc.obs.ncomp == 1 || throw(DimensionMismatch(
            "observable $(acc.obs.name) returned a scalar but declares " *
            "$(acc.obs.ncomp) components"))
        acc.val[1] = Float64(v)
    else
        length(v) == acc.obs.ncomp || throw(DimensionMismatch(
            "observable $(acc.obs.name) returned $(length(v)) components but " *
            "declares $(acc.obs.ncomp)"))
        copyto!(acc.val, v)
    end
    push!(acc.binner, acc.val)
    push!(acc.store, acc.val)
    return acc
end

# Finalize one temperature: raw stats from the binners, evaluables jackknifed over
# the stored bins of their (scalar) inputs.
function _finalize_stats(accs::Vector{ObsAccumulator}, evals::Vector{Evaluable},
                         kT::Float64, n_active::Int)::Dict{Symbol,ObservableStat}
    stats = Dict{Symbol,ObservableStat}()
    byname = Dict(acc.obs.name => acc for acc in accs)
    for acc in accs
        stats[acc.obs.name] = ObservableStat(acc.obs.name, mean(acc.binner),
                                             std_error(acc.binner),
                                             tau_int(acc.binner), acc.binner.n)
    end
    for ev in evals
        cols = Vector{Vector{Float64}}(undef, length(ev.inputs))
        nb = typemax(Int)
        ok = true
        for (q, name) in enumerate(ev.inputs)
            acc = get(byname, name, nothing)
            acc === nothing && throw(ArgumentError(
                "evaluable $(ev.name) needs observable :$name, which is not measured"))
            acc.obs.ncomp == 1 || throw(ArgumentError(
                "evaluable $(ev.name) input :$name is not a scalar observable"))
            cols[q] = vec(bin_means(acc.store))
            nb = min(nb, length(cols[q]))
            ok &= nb >= 2
        end
        if !ok
            stats[ev.name] = ObservableStat(ev.name, [NaN], [NaN], [NaN], 0)
            continue
        end
        # One temperature's accumulators are built alike and receive the same
        # `_measure!` calls, so the input columns are equal-length by construction —
        # and `jackknife` requires it. Assert rather than truncate to `nb`: a ragged
        # set here means a new accumulation path broke that lockstep, and silent
        # truncation would hide it behind a slightly-wrong error bar.
        # [Backported from SLCEMonteCarlo.jl 0278973.]
        all(length(c) == nb for c in cols) || throw(ArgumentError(
            "evaluable $(ev.name): input bin columns have unequal lengths " *
            "$(map(length, cols)) — the accumulators' lockstep is broken"))
        keys_tuple = Tuple(ev.inputs)
        f = (ms...) -> ev.f(NamedTuple{keys_tuple}(ms), kT, n_active)
        est, err = jackknife(f, cols)
        stats[ev.name] = ObservableStat(ev.name, [est], [err], [NaN], nb)
    end
    return stats
end

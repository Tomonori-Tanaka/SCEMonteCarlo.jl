# Constrained Monte Carlo (CMC) for finite-temperature anisotropy: the
# Asselin-type pair move that holds the direction of a (weighted) magnetization
# fixed, plus the torque observables that read the free-energy gradient off the
# constrained ensemble. The standing contract — constrained measure, acceptance
# rule, null moves, partner rule, sweep and RNG consumption, torque conventions —
# is `docs/specs/constrained-mc.md`.
#
# This file only CALLS the energy kernels of energy.jl (`site_coeffs!`,
# `delta_energy`, `_zlm_row!`) and the public gradient (`energy_gradient!`, via
# `total_torque` in observables.jl); it changes none of them.

"""
    _CMCScratch(H::TiledHamiltonian)

Pre-allocated buffers of one CMC lane (never shared across tasks), sized like
[`SweepScratch`](@ref) from `H.nlm` / `H.lmax`:

- `lc` — the leave-one-out coefficient vector of the site being evaluated (the
  `c` of [`site_coeffs!`](@ref); named `lc` to keep it apart from the constraint);
- `zi_new` — the tesseral row `Z(e_i′)` of the source spin's proposal;
- `zi_old` — the saved column `i` of `zrows` while `Z(e_i′)` is written there
  tentatively (restored bitwise before [`_pair_delta_energy!`](@ref) returns);
- `zj_new` — the tesseral row `Z(e_j′)` of the compensating partner's proposal;
- `plm` — the associated-Legendre workspace of the internal `_zlm_row!`.

After a `_pair_delta_energy!` call `zi_new`/`zj_new` hold the proposed rows, so an
accepted move copies them into `zrows` without re-evaluating the harmonics.
"""
struct _CMCScratch
    lc::Vector{Float64}
    zi_new::Vector{Float64}
    zi_old::Vector{Float64}
    zj_new::Vector{Float64}
    plm::Vector{Float64}
end

_CMCScratch(H::TiledHamiltonian) =
    _CMCScratch(zeros(H.nlm), zeros(H.nlm), zeros(H.nlm), zeros(H.nlm),
                Vector{Float64}(undef, H.lmax + 1))

"""
    _pair_delta_energy!(scratch::_CMCScratch, zrows, H, i, j, e_i_new, e_j_new)
        -> Float64

Exact energy change of moving the two distinct sites `i ≠ j` jointly to
`e_i_new`, `e_j_new`, by sequential leave-one-out (`docs/specs/constrained-mc.md`):

    ΔE = [E(e_i′, e_j) − E(e_i, e_j)] + [E(e_i′, e_j′) − E(e_i′, e_j)]

1. `ΔE_i = c_i · (Z(e_i′) − Z(e_i))` with `c_i` from the current `zrows`;
2. column `i` of `zrows` is saved to `scratch.zi_old` and `Z(e_i′)` is written
   there tentatively;
3. `ΔE_j = c_j · (Z(e_j′) − Z(e_j))` with `c_j` built from the tentative `zrows`,
   so every instance that contains both sites sees `e_i′`;
4. column `i` is restored bitwise from `scratch.zi_old`.

Each bracket is a single-site move, exact at any body order under the
distinct-member-sites invariant of the constructor. When `i` and `j` share no
instance, `c_j` does not read column `i` and the result is bitwise the naive sum
of two independent single-site differences.

On return `zrows` is bitwise unchanged, `scratch.zi_new == Z(e_i′)` and
`scratch.zj_new == Z(e_j′)`; nothing else is written. Allocation-free. The spins
of sites `i`, `j` in the configuration are never read — `zrows` is the state.

Preconditions (the caller's): `i, j ∈ 1:H.n_sites`, `zrows` of size
`H.nlm × H.n_sites`, `scratch` built for this `H`, and `e_i_new`, `e_j_new` unit
vectors whose components lie in `[-1, 1]` (the harmonics throw otherwise — before
`zrows` is touched, so a throw leaves it intact).
"""
function _pair_delta_energy!(scratch::_CMCScratch, zrows::Matrix{Float64},
                             H::TiledHamiltonian, i::Int, j::Int,
                             e_i_new::SVector{3,Float64},
                             e_j_new::SVector{3,Float64})::Float64
    @assert i != j "pair move needs two distinct sites; got i = j = $i"
    lc = scratch.lc
    # Both proposed rows are tabulated BEFORE column i is touched: the harmonics
    # throw a DomainError on a component a few ulp outside [-1, 1], and a throw
    # inside the tentative-write window would leave `zrows` inconsistent.
    _zlm_row!(scratch.zi_new, e_i_new, H.lmax, scratch.plm)
    _zlm_row!(scratch.zj_new, e_j_new, H.lmax, scratch.plm)
    # (1) source site against the current rows
    fill!(lc, 0.0)
    site_coeffs!(lc, H, i, zrows)
    ΔE_i = delta_energy(lc, view(zrows, :, i), scratch.zi_new)
    # (2) tentatively place e_i′ — every instance shared with j must see it
    zcol_i = view(zrows, :, i)
    copyto!(scratch.zi_old, zcol_i)
    copyto!(zcol_i, scratch.zi_new)
    # (3) partner site against the tentative rows
    fill!(lc, 0.0)
    site_coeffs!(lc, H, j, zrows)
    ΔE_j = delta_energy(lc, view(zrows, :, j), scratch.zj_new)
    # (4) restore column i bitwise. The only calls inside the window are
    # `fill!`/`site_coeffs!`/`delta_energy` on the scratch and rows, which cannot
    # throw for in-range sites `i, j ∈ 1:H.n_sites` (the caller's precondition) and
    # a `zrows` of size `H.nlm × H.n_sites`; this is the only exit, so it always
    # runs.
    copyto!(zcol_i, scratch.zi_old)
    return ΔE_i + ΔE_j
end

# Analytic Zeeman torque of the full system, `MU_B_EV_T · (M_μ × B)` with
# `M_μ = Σ_{active s} m_{a(s)} e_s` (μ_B; no `1/n_cells`). Sign: the Zeeman energy
# is `E_Z = −MU_B_EV_T Σ m (e·B)`; a global rotation by `α` about the unit axis `w`
# moves `e → e + α w × e`, so `dE_Z/dα = −MU_B_EV_T Σ m (w × e)·B
# = −w · MU_B_EV_T (M_μ × B)`, and the torque convention `T·w = −dE/dα`
# (`docs/specs/constrained-mc.md`) gives `T_Z = +MU_B_EV_T (M_μ × B)` — the moment
# is pulled toward `B`. Zero without a field. This is the part of
# [`total_torque`](@ref) the external field contributes, which partial
# (field-free) Hamiltonians leave out.
function _zeeman_torque(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    mm = H.magmoms
    (mm === nothing || !has_field(H)) && return zero(SVector{3,Float64})
    M = zero(SVector{3,Float64})
    for s = 1:H.n_sites
        H.site_active[s] || continue
        M += mm[site_atom(H, s)] * config[s]
    end
    return MU_B_EV_T * cross(M, H.field)
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
`total_torque(Hpart, config) / n_cells` (Cartesian, per training cell, unweighted —
the normalization of `:torque`; see [`total_torque`](@ref) and
`docs/specs/constrained-mc.md`).

Build `Hpart` with the public constructor from the SAME term list that built `H`
(after [`reduce_cell`](@ref): `red.terms`), filtered to the subset of interest and
without a field — e.g. `TiledHamiltonian(n, filter(t -> t.body == 1, terms); dims)`
for the on-site part. Over a disjoint cover of the fitted terms the partial
torques plus the analytic Zeeman torque sum to the total torque, configuration by
configuration; an individual partial torque is a free-energy gradient only to
first order in the anisotropy.

Doors (`ArgumentError`): `Hpart` and `H` must agree on `n_cell_atoms`, `dims`, and
`n_sites`; `Hpart` must carry no field (the Zeeman part is measured analytically);
every template of `Hpart` must equal — in atoms, shifts, `ls`, `folded`, and the
scaled coefficient — one of the fitted templates `H.terms[1:H.n_fitted_terms]`, so
a subset of a different term list (another cell, another fit) is refused even when
the atom counts happen to agree. Disjointness and coverage are not enforced.

The observable is bound to the `H` object it was built with: measuring it in a run
on any other `TiledHamiltonian` (even an equal one) throws an `ArgumentError`, so
build it with the same `H` the run uses. `Hpart` and `H` are immutable and the
returned function captures no mutable state, so the observable may be shared across
lanes that sample the same `H`.
"""
function torque_observable(name::Symbol, Hpart::TiledHamiltonian,
                           H::TiledHamiltonian)::Observable
    Hpart.n_cell_atoms == H.n_cell_atoms || throw(ArgumentError(
        "torque_observable: Hpart has $(Hpart.n_cell_atoms) cell atoms, H has " *
        "$(H.n_cell_atoms)"))
    Hpart.dims == H.dims || throw(ArgumentError(
        "torque_observable: Hpart.dims = $(Tuple(Hpart.dims)) differs from " *
        "H.dims = $(Tuple(H.dims))"))
    Hpart.n_sites == H.n_sites || throw(ArgumentError(
        "torque_observable: Hpart has $(Hpart.n_sites) sites, H has $(H.n_sites)"))
    has_field(Hpart) && throw(ArgumentError(
        "torque_observable: Hpart carries an external field; build partial " *
        "Hamiltonians field-free (the Zeeman torque is measured analytically)"))
    fitted = Set(_term_key(t) for t in view(H.terms, 1:H.n_fitted_terms))
    for (k, t) in enumerate(Hpart.terms)
        _term_key(t) in fitted || throw(ArgumentError(
            "torque_observable: template $k of Hpart (atoms = $(t.atoms), " *
            "shifts = $(t.shifts), ls = $(t.ls)) matches no fitted template of " *
            "H; build Hpart from the same term list as H"))
    end
    ncells = prod(H.dims)
    return Observable(name, 3, (config, E, Hrun) -> begin
        Hrun === H || throw(ArgumentError(
            "torque_observable $name was built for a different TiledHamiltonian " *
            "than the one this run samples; build it with the run's own H"))
        total_torque(Hpart, config) / ncells
    end)
end

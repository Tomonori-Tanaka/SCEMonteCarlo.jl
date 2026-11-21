# Decision record — binning error analysis and observable conventions

Status: landed (M2); B5 added with the constrained-MC driver (2026-11-21),
`:torque_frame` and the start-time constraint check added to it the same day. Owner:
`src/binning.jl`, `src/observables.jl`, `src/cmc/driver.jl` (B5); gates in
`test/unit/test_binning.jl`, `test/unit/test_observables.jl`,
`test/unit/test_cmc_driver.jl` (B5).
**This file is the authoritative statement of the C/χ/U conventions** — the code
and the guide pages follow it.

## B1 — streaming log-binning, no stored time series

`LogBinner` keeps, per cascade level `k`, the `(count, Σ, Σ²)` of bin means of size
`2^(k-1)` plus one pending half-pair — O(levels) memory. The reported error is the
naive standard error at the **deepest level with ≥ 32 bins** (the plateau proxy);
`τ_int = ((err_pl/err_naive)² − 1)/2`.

Known accuracy: with 32 bins the plateau error itself fluctuates by
`≈ 1/√(2·31) ≈ 13%` (τ, quadratic in it, by ≈ 26%) — fine for error bars, and the
reason the AR(1) test gate uses matching tolerances. Rejected alternative: storing
the full time series for a windowed autocorrelation estimator — accuracy is not
needed at that price (a 32-replica PT run × 10⁶ measurements would cost GB).

## B2 — jackknife over a fixed bin layout for derived quantities

Nonlinear evaluables (C, χ, U) are estimated by leave-one-bin-out jackknife over
`BinStore` bin means: `nbins` (default 32) equal-size bins with
`bin_size = max(1, planned ÷ nbins)` fixed **up front**, so every bin has equal
weight; a trailing remainder (< 1 bin) is dropped. Bias-corrected estimate
`n_b·f(m̄) − (n_b−1)·θ̄`, error `√((n_b−1)/n_b · Σ(θᵢ−θ̄)²)`. For a linear `f` this
reproduces the plain mean and error exactly (machine-precision gate).

## B3 — observable conventions (authoritative)

**Active sites.** A site no cluster instance touches (a species with `lmax = 0`,
e.g. boron, or one whose SALC coefficients all fitted to zero) has a
spin-independent energy: it is flagged inactive (`TiledHamiltonian.site_active`,
count `n_active`), skipped by the update sweeps (frozen at its initial direction;
attempting it would always accept, biasing the acceptance statistics and the step
adaptation), **excluded from every standard observable**, and per-site
normalizations use `n_active`. *Why exclusion and skipping come together*: a frozen
spin included in `:m` would add a constant bias, a free-walking one dilution noise —
either contaminates; excluded, its updates are pure waste. `:sublattice_m` reports
exactly zero for inactive sublattices (a frozen random unit vector would look like
perfect order). The `f(config, energy, H)` signature hands custom observables
`H.site_active` for the same masking.

Raw set (`standard_observables`): `:energy`, `:energy2` (total, model units, `j0`
excluded), `:m` (3-vector `Σ_active e / n_active`), `:absm`, `:m2`, `:m4`,
`:sublattice_m` (per training-cell atom, cell-averaged 3-vector, flattened).
Directions only by default — moment magnitudes (μ_B) are not part of the fitted
model. With `magmoms` (`zeeman-field.md`): `:M = Σ_active m_{a(s)} e_s / n_cells`
(3-vector, μ_B per cell of `H` — the reduced cell after `reduce_cell`; the
`:sublattice_m` normalization, not `:m`'s per-site one) and, with a nonzero field, `:M_B = M · B̂`. A field is the one way
a site no fitted cluster touches becomes magnetic: its Zeeman template is an
instance, so it is active and counts in `:m` / `:sublattice_m` / `:M` — a genuine
field-induced polarization. Without a field, `magmoms` changes none of the above
(the moment-carrying SCE-inactive sublattice stays frozen and excluded).

Derived (`standard_evaluables`, jackknifed):

- **Specific heat, per active site, in units of k_B**:
  `C/k_B = (⟨E²⟩ − ⟨E⟩²) / (n_active (k_BT)²)`.
  *Why*: intensive (comparable across `dims`); k_B units avoid eV/K clutter and are
  the lattice-MC standard.
- **Susceptibility, |m|-connected, per active site**:
  `χ = n_active (⟨m²⟩ − ⟨|m|⟩²) / k_BT`.
  *Why*: on a finite system with continuous symmetry `⟨m⟩ = 0` exactly, so the
  textbook connected form degenerates to `n⟨m²⟩/kT`, which grows ∝ N below T_c
  instead of peaking; the |m|-connected form peaks at the transition in both phases
  (the finite-size-scaling standard). The high-T form is user-composable from `:m2`.
- **Binder cumulant, plain ratio**: `U = ⟨m⁴⟩/⟨m²⟩²` — → 1 (ordered), → 5/3
  (disordered, 3-component Gaussian). Chosen over `1 − U/3`-style variants because
  there is no convention factor to get wrong; `U(T)` crossings locate `T_c`
  identically.

## B4 — composability

`Observable(name, ncomp, f(config, energy, H))` and
`Evaluable(name, inputs, f(means::NamedTuple, kT, n_active))` are plain structs the
run drivers accept as vectors — nothing is hard-coded into the sweep (the
SpinClusterMC pain point). Evaluable inputs must be scalar observables (validated).
Ferrimagnetic order parameters compose from `:sublattice_m` components.

## B5 — constrained-MC observables (`cmc_observables`, `cmc_evaluables`)

The defaults of `run_cmc` (`docs/specs/constrained-mc.md`), with `n_cells =
n_sites ÷ n_cell_atoms = prod(dims)` (cells of `H` in the supercell — the cell of
`H` itself, i.e. the reduced cell after `reduce_cell`, not the training cell);
every observable below is per cell of `H` in this sense. `𝒞` the constrained
sites (active, nonzero weight), `M = Σ_{s∈𝒞} w_{a(s)} e_s` the weighted
magnetization, `n̂` the constraint direction:

- `:energy`, `:energy2` — as in B3 (total, model units, `j0` excluded).
- `:torque` — `total_torque(H, config) / n_cells`, 3 Cartesian components, model
  energy units per cell of `H`. `total_torque = Σ_{s active} G_s × e_s =
  −Σ_s e_s × ∂E/∂e_s` with `G` the tangent-projected gradient of
  `energy_gradient!` (Zeeman part included, every term with unit weight — never
  the constraint weights). Sign: for a rigid rotation of every spin by the angle
  `α` about the unit axis `w`, `T·w = −dE/dα`. Identity, exact at finite N:
  `⟨T⟩ = −n̂ × ∂F/∂n̂` for the free energy `F(n̂)` of the canonical distribution of
  `M̂`, i.e. `⟨T⟩·ê_φ = −∂F/∂θ`, `⟨T⟩·ê_θ = (1/sinθ) ∂F/∂φ`, `⟨T⟩·n̂ = 0`
  (`ê_φ = (−sinφ, cosφ, 0)`, `ê_θ = (cosθ cosφ, cosθ sinφ, −sinθ)`); per cell,
  the same identity holds for `F/n_cells`. Cartesian; the polar projection is
  `:torque_frame` below.
- `:M_par` — `M·n̂ / n_cells`, in the units of the weights per cell of `H`
  (μ_B per cell for `weights = :magmoms`, a pure number for `:uniform`).
  Inactive sites and weight-0 sites are not in `M`.
- `:constraint_residual` — `|M_⊥| / M_∥`, `M_⊥ = M − (M·n̂) n̂`: the angle between
  `M` and `n̂` to first order (radians), a **relative** rounding diagnostic
  (rounding of the pair moves and renormalizations makes it a random walk of
  order 1e-13 over 10⁵ sweeps). The drift warning of the sweep is a different,
  **absolute** test, `|M_⊥| > 1e-10 Σ_{s∈𝒞}|w_s|` at every sweep start: a relative
  bound would misfire on starts near the `|M_init|` door (`constrained-mc.md` D3).
- `:torque_frame` — `(ê_φ·T, ê_θ·T, n̂·T) / n_cells` with `T` the total torque
  of the SAME evaluation as `:torque` within a run: the single-chain measurement
  of `run_mc` / `run_cmc` (lanes and `resume` included) evaluates the torque once
  per measurement from the chain state, with per-chain scratch, and both read it
  (the chain-state torque path, `constrained-mc.md` D5). Called directly, or in
  `run_pt`, each evaluates `total_torque` on its own — the same values, bitwise.
  The frame depends on `n̂` only and is built once, in `cmc_observables`:
  `ê_φ = normalize(ẑ × n̂)`, `ê_θ = ê_φ × n̂` — for `n̂ = (sinθ cosφ, sinθ sinφ,
  cosθ)` exactly the `ê_φ`, `ê_θ` above (`ẑ × n̂ = sinθ (−sinφ, cosφ, 0)`;
  `(ê_θ, ê_φ, n̂)` is right-handed). Averaged: `(−∂f/∂θ, (1/sinθ) ∂f/∂φ, 0)`,
  `f = F/n_cells`. At a pole (`|ẑ × n̂| < 1e-12`) `ê_φ = ŷ` (made orthogonal to
  `n̂`) and `ê_θ = ê_φ × n̂`: `(ŷ, x̂)` at `+ẑ`, `(ŷ, −x̂)` at `−ẑ` — the `φ = 0`
  limit of the frame at both poles, whatever `φ` the direction was built from.
  **Why it is binned directly**: the error bar of a projection `u·⟨T⟩` is
  `√(uᵀ Σ u)` with `Σ` the covariance of the three Cartesian means, which the
  per-component errors of `:torque` do not carry (the binner keeps no
  cross-component sums); propagating the per-component errors as if independent
  is wrong whenever the components are correlated (in general they are: a tangent
  torque has components along two or three Cartesian axes at once). Binning
  the projected sample gets the variance and the autocorrelation of exactly the
  quantity reported.
- `:torque_zeeman` (only with a field) — `MU_B_EV_T (M_μ × B) / n_cells`,
  `M_μ = Σ_{s active} m_{a(s)} e_s` (the moments, not the constraint weights): the
  part of `:torque` the external field contributes, written analytically
  (`T_Z = +MU_B_EV_T M_μ × B` from `E_Z = −MU_B_EV_T Σ m e·B` and the sign above).

The measuring functions are bound to the `H` they were built for (a run on another
`TiledHamiltonian` object throws) and hold no mutable state (the per-measurement
scratch of the torque belongs to the chain, not to the set), so one set may be
shared by concurrent lanes. `:M_par`, `:constraint_residual`, and `:torque_frame`
measure along the constraint they were BUILT for; their functions carry that
constraint, and `run_cmc` (single chain and every lane) and `resume` refuse,
before anything runs, a torque observable bound to another `H`, and a set built
for another `H` or for another constraint — direction more than `1e-6` (chord)
away, per-site weights more than `1e-6 max|w|` apart, or (near a pole) the
`:torque_frame` axes more than `1e-6` apart (`constrained-mc.md` D5).
Observables of the user's own are not inspected.
Derived: `cmc_evaluables()` is the
specific heat of B3 only. `Var(E)/kT²` stays exact in the constrained ensemble —
its reference measure does not depend on the temperature — while the
susceptibility and the Binder cumulant change meaning under a fixed direction of
`M` and are not offered.

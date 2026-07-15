# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Full classical spin Monte Carlo for fitted SCE models from
[`SCEFitting.jl`](../SCEFitting.jl) — the from-scratch successor of the frozen
`SpinClusterMC.jl` (Magesty-XML + Carlo.jl), with no API-compatibility constraint.
Core capabilities: tile the fitted training-cell Hamiltonian onto an `N₁×N₂×N₃`
supercell (`TiledHamiltonian`, via `MultipoleTerm.shifts`) — optionally after a
*verified* re-expression in a user-chosen smaller cell (`reduce_cell`, so `dims` is
not locked to training-cell multiples) — single-spin Metropolis
with an adaptive step, overrelaxation, annealing sweeps (`run_mc`), replica exchange
over threads (`run_pt`), numerical ground-state search (`minimize_energy` /
`find_ground_state`), composable observables with log-binning errors + jackknife
evaluables, and bit-reproducible JLD2 checkpoint/restart (reproducibility scope —
same package + Julia version, thread-count-independent, trajectory not
observable-ULPs: `docs/specs/pt-threads-determinism.md` P6). Self-contained core —
**no Carlo.jl dependency** (a Carlo adapter could later be a weakdep extension).

Relation to siblings: `SCETools.jl` keeps the single-training-cell *configuration
samplers* (MFA + light Metropolis); this package is for thermodynamics-grade runs
(observables, error bars, T sweeps). `SpinClusterMC.jl` is a read-only design
reference — its pain points (God-struct, module-level global caches, split
temperature-unit conventions, hard-coded observables, per-instance payload
duplication, positional hand-rolled serialization) are what this design avoids.

This package reads a fitted model **only** through `SCEFitting`'s public surface:
`multipole_terms`, `n_atoms(model)`, `intercept`, `SCEFitting.load(SCEPredictor, …)`,
`Lattice`/`Crystal`/`cartesian_positions`, and `SCEFitting.Harmonics` (`Zlm_unsafe`,
`lm_index`, `num_lm`, `grad_Zlm_unsafe`) — never SALC-basis internals and never
`model.basis.crystal` (not public tier; geometry helpers take an explicit `Crystal`).
During development the dependency is a path-dev: `Pkg.develop(path="../SCEFitting.jl")`.

## Numerical / physics conventions

- **Spin directions are unit vectors.** Internal state is `SpinConfig =
  Vector{SVector{3,Float64}}` (one entry per supercell site); the 3×n matrix layout
  of the siblings appears only at the I/O boundary (`to_matrix`/`from_matrix`).
- **Real (tesseral) spherical harmonics `Zₗₘ`** from `SCEFitting.Harmonics`
  (`lm_index(l, m) = l² + l + m + 1` ordering). `multipole_terms` returns the **raw**
  fitted `jϕ`; the `(4π)^(body/2)` scale is applied **exactly once**, in the
  `TiledHamiltonian` constructor (`ScaledTerm.coef`). Never re-apply downstream.
- **Energies** are in the model's energy units (eV for DFT-fitted models), `j0`
  (intercept) excluded everywhere — MC only needs differences; the reconstruction
  gate is `total_energy(H₁ₓ₁ₓ₁, cfg) == predict_energy(model, cfg) − intercept(model)`.
- **Supercell tiling**: `MultipoleTerm.shifts` are per-site integer training-cell
  lattice translations (`shifts[1] = 0` anchored). One instance per template term and
  supercell cell `t`, member `i` at `site_index(atom_i, mod.(t + shifts[i], dims))`.
  Each directed member is one plain summand — no ½ or 1/N factors.
- **Temperature**: absolute only, exactly one of `temperature` [K] XOR `kT`
  [model energy units]; `KB_EV` is the exact CODATA ratio. β enters only in accept
  steps; coefficients and energies stay in model units.
- **ΔE locality**: every instance's member *sites* are distinct after the toroidal
  wrap (asserted per term in the ctor — minimum-image models have distinct atoms
  outright; `AllImages` models may reuse an atom across images and need `dims` large
  enough), so the leave-one-out coefficient vector `c_s` is independent of `e_s` and
  `ΔE = c_s·(Z(e′) − Z(e))` is exact for any body order.

## Coupled ("linked") code sites — change one, check all

- **`hamiltonian.jl` ↔ the core's introspection contract** (`SCEFitting`'s
  `sce/introspect.jl`): field meanings of `MultipoleTerm` (coef/body/atoms/shifts/
  ls/folded), the raw-coef scale rule, and the shifts anchoring. Gates:
  `test_hamiltonian.jl` (dims=(1,1,1) ≡ `predict_energy − intercept`; 2×2×2
  periodic-replication = 8× cell energy; scale-once).
- **`energy.jl` 4-function contract ↔ `updates.jl` ↔ SCETools' `mc/metropolis.jl`**:
  `site_coeffs!`/`delta_energy` are the site-generalized siblings of SCETools'
  `_accumulate_site_term!` kernel (same `μ = idx − l − 1` mapping, rank-specialized
  barrier). Gates: `test_energy.jl` ΔE ≡ total-energy difference (1e-12).
- **`lm_index` ordering ↔ `zlm_row!` ↔ the overrelaxation l=1 axis extraction**
  (`updates.jl`): the tesseral l=1 components map to Cartesian axes; a reorder
  upstream breaks the OR axis. Gate: pure-l=1 OR proposals have `ΔE ≡ 0` and
  acceptance 1 (`test_overrelaxation.jl`).
- **`reduce.jl` ↔ `hamiltonian.jl` tiling ↔ `geometry.jl` ordering**: `reduce_cell`
  emits raw-coefficient `MultipoleTerm`s (the `(4π)^(body/2)` scale still happens
  once, in the `TiledHamiltonian` ctor), anchored `shifts[1] = 0`, and a reduced
  `Crystal` whose atom order matches `site_index` so `supercell_crystal(red.crystal,
  dims)` pairs with `TiledHamiltonian(red; dims)`. The anchored-form invariance and
  verification contract live in `docs/specs/cell-reduction.md`. Gates:
  `test_reduce.jl` (exact term recovery, energy identity via site permutation).
- **`minimize.jl` `_gradient!` ↔ `energy.jl` `site_gradient`**: the fast all-site
  gradient must stay arithmetically identical to the public per-site one (same
  `(l, m)` loop over `lm_index` order, same `ck == 0` skip). Gate: the bitwise `==`
  consistency test in `test_minimize.jl`; an `lm_index` reorder upstream breaks
  both together with the OR axis (previous bullet).
- **Checkpoint writer ↔ reader ↔ schema doc** (`checkpoint.jl`,
  `docs/specs/checkpoint-schema.md`): plain-data JLD2 schema v1, Xoshiro capture via
  `fieldnames`, accumulator state. Gate: bit-identical resume (`test_checkpoint.jl`).
- **Observable conventions** (C/χ/U definitions) live in ONE place:
  `docs/specs/binning-observables.md`; `observables.jl` and the guide pages follow it.
- **Coloring ↔ sweeps ↔ stationarity spec** (`hamiltonian.jl` `_color_sites` /
  `color_ptr`/`color_sites`, `updates.jl`, `docs/specs/updates-stationarity.md`
  U1): the sweeps assume every color class is instance-disjoint (exactly
  independent single-site kernels) and bit-determinism for any `sweep_tasks` rests
  on per-site RNG streams (`ChainState.site_rngs`, checkpoint schema v2) + the
  fixed-order ΔE reduction (`_reduce_dE`). Touch the coloring, the sweep loops, or
  the reduction and re-run `test/unit/test_parallel.jl` (serial ≡ parallel `==`).
- **Device tesseral row ↔ host `_zlm_row!` ↔ upstream recursions**
  (`src/gpu/zlm_device.jl`): `_zlm_row_device!` is a deliberate, operation-order-
  faithful reimplementation of `_zlm_row!` → `Harmonics.Zlm_unsafe` →
  `LegendrePolynomials.dnPl` (+ `Base.power_by_squaring` as `_zlm_cpow`), because
  the upstream path cannot compile in a GPU kernel. Any upstream change to those
  functions (a normalization, a recursion reorder, an SCEFitting `Harmonics`
  edit) breaks the dense bitwise gate in `test/unit/test_gpu.jl` — update the
  device file together with it.
- **GPU kernel ↔ keyed reference ↔ slot map ↔ workgroup-size pin**
  (`src/gpu/gpu_sweep.jl`, `src/gpu/philox.jl`, `docs/specs/gpu-prototype.md`
  G2–G4): `_metro_kernel!` and `_metropolis_sweep_keyed_ref!` implement ONE
  arithmetic contract (proposal slots, `_entry_walk_partial` dispatch + zero
  skips, lane-ordered fold, accept rule). Touch any of them — or the Philox slot
  layout, or the pinned default ws — and the other side plus the G-record move
  together; gate: the full-sweep bitwise section of `test/unit/test_gpu.jl`.
- **Inactive-site convention** (`site_active`/`n_active` — sites with no adjacent
  instance): update sweeps **skip**, standard observables **exclude**, per-site
  normalizations use `n_active`, and sweeps/renormalization/descent keep the spins
  **bitwise frozen**. These move together — skipping without excluding turns a
  frozen random direction into a constant observable bias. Touch `updates.jl`,
  `observables.jl`, `state.jl` `_renormalize!`, or `minimize.jl` `_gradient!`/
  `_minimize!` and re-check `test/unit/test_inactive.jl`.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `cd docs && make build` | docs (checkdocs = :exports) |

Statistical gates use fixed seeds with tolerances proven in SCETools' MC suite.
Manual smoke (not CI): Nd₂Fe₁₄B l02 model (`~/jijs/magesty/2-14-1/nd2fe14b/1x1x1/
magesty/l02/test`, rebuild via its `fit_mfa.jl` recipe), dims=(4,4,4), short PT
across the ordering temperature. Last run (2026-07-11, v0 completion): 1×1×1 and
64× counting gates at ~1e-13; construction 0.01 s / 7.8 MB index; 8 rungs ×
900 sweeps × 4352 sites in 38 s on 8 threads; ferrimagnetic projections
Nd ≈ −0.50 / Fe ≈ +0.69 at 250 K. Note: 8 rungs over 250–1300 K give *zero*
swaps at this size (rung count must scale like √(n_sites·C) — documented in the
PT guide), so use denser ladders for production.

## References

- `SPEC.md` — architecture, primary types, public API.
- `docs/specs/*.md` — decision records (tiling, update stationarity,
  binning/observable conventions, PT determinism, checkpoint schema).
- `references/` — supporting literature (notes tracked, PDFs local-only).

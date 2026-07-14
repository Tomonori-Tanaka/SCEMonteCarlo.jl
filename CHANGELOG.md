# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `run_mc` / `run_pt` default `seed` is now drawn fresh per call
  (`rand(UInt64)`) instead of the fixed `0`, so repeated default runs are
  independent samples rather than silent duplicates. Reproducibility is opt-in
  (pass an explicit `seed`) and never lost: the seed actually used is recorded
  in the result and in checkpoints.

### Added

- Benchmark suite (`bench/`, own environment): bottleneck-oriented scripts —
  `bench_kernels` (the single-spin attempt decomposed: `_zlm_row!` /
  `site_coeffs!` / `delta_energy`, plus `_total_energy` and the diagnostics
  paths), `bench_sweeps` (ns/attempt and allocs/sweep for Metropolis and
  overrelaxation), `bench_tiling` (`TiledHamiltonian` construction + retained
  memory), `bench_run` (`run_mc` with measurement-overhead isolation; `run_pt`
  thread scaling), `bench_minimize` (gradient pass, BB descent, multi-start
  search), and `bench_profile` (line-level `Profile` tree/flat reports per
  target). Two fixtures span the kernel regimes: a 2-atom bcc Fe `l = 1` model
  (light kernel, large lattice) and a synthetic-coefficient Nd₂Fe₁₄B model
  (`bench/assets/nd2fe14b.toml`, ~9400 terms, site adjacency ~276 — the real
  l02 production regime). Baselines and first findings (per-instance
  dynamic-dispatch allocations in the energy kernels) in `.claude/bench_log.md`.
- Ground-state search: `minimize_energy` (deterministic Riemannian
  Barzilai–Borwein projected-gradient descent on the sphere product, nonmonotone
  Armijo safeguard, no optimizer dependency, no RNG in the descent) and
  `find_ground_state` (multi-start simulated annealing with optional thermal
  cycling — `cycles`/`reheat` — polished by the same descent; threads-parallel and
  bit-identical for a fixed seed regardless of `ntasks`), both returning
  `GroundStateResult` with the per-start energy table as a degeneracy diagnostic.
  Includes the PT-polish recipe (`inits = pt.final_configs, anneal_sweeps = 0`),
  an executed docs guide with figures, and the decision record
  `docs/specs/ground-state-search.md`.
- Docs: a parallelism guide — how the Threads lane pool works, the explicit
  limits (no MPI/GPU; one PT ladder is bounded by one node), and multi-node
  recipes (`Threads.@threads` over temperatures, SLURM job arrays with blind
  `resume` retries — backed by a new idempotent-resume gate).
- Docs: executed figures in the parallel-tempering guide — four annealed chains vs
  four PT runs on a random-anisotropy model (one chain freezes into a basin 100×
  its own error bar away; PT rescues every seed), and the ladder diagnostics
  (swap acceptance collapsing with system size, recovering with rung count). The
  docs build now runs with `-t 4` so PT examples sweep lanes in parallel.
- Docs: an executed tutorial (`tutorials/cubic_heisenberg.md`) — the ferromagnetic
  transition of a simple-cubic classical Heisenberg model, with figures (energy,
  specific heat, magnetization, susceptibility, Binder-cumulant crossing on the
  literature ``k_BT_c/|J| = 1.443``) computed at docs-build time, plus a
  user-defined staggered-magnetization observable on the antiferromagnetic
  counterpart. CairoMakie + Spglib added to the docs environment.
- Cell reduction: `reduce_cell` / `ReducedCell` — re-express a model fitted on a
  supercell in a user-chosen smaller (or re-based, non-diagonal `M` included) cell,
  after verifying the lattice relation, the atomic mapping, and that every fitted
  term has its full set of translation copies; `TiledHamiltonian(red; dims)` then
  counts `dims` in reduced-cell units, decoupling finite-size checks from the
  training-cell granularity (`docs/specs/cell-reduction.md`).
- Geometry/I-O helpers: `supercell_crystal` (site ordering matched to
  `TiledHamiltonian`), `to_matrix` / `from_matrix`.
- Checkpoint/restart: `checkpoint`/`checkpoint_interval` on `run_mc`/`run_pt` +
  `resume` — versioned plain-data JLD2 schema (model fingerprint, Xoshiro words,
  accumulator cascades), atomic writes, and bit-identical resumed runs
  (`docs/specs/checkpoint-schema.md`).
- `run_pt`: replica exchange (parallel tempering) over threads — one lane per
  ladder rung, payload-swap exchanges every `exchange_interval` sweeps
  (thermalization and measurement alike), per-lane adaptive steps and
  accumulators, and bit-identical results for a fixed seed regardless of
  `ntasks`/thread count (`docs/specs/pt-threads-determinism.md`).
- Overrelaxation sweeps (`or_per_metropolis`): deterministic involutive reflection
  about the local `l=1` field axis + Metropolis correction — exactly microcanonical
  on pure-`l=1` channels, exact for any body order via the accept step
  (stationarity: `docs/specs/updates-stationarity.md`).
- `run_mc`: single-temperature and warm-started ladder (annealing) runs of
  single-spin Metropolis with the exact `ΔE = c_s·ΔZ` kernel, symmetric
  flip+Rodrigues proposal, thermalization-only adaptive step (frozen during
  measurement), periodic renormalize + energy re-anchoring with drift tracking,
  and bit-reproducible seeding. Results as `MCResult` / `TempResult` with a
  summary-table printer.
- Composable measurement layer: `Observable` / `Evaluable` with the standard set
  (`E`, `E²`, `m`, `|m|`, `m²`, `m⁴`, per-sublattice magnetization) and derived
  `C/k_B`, |m|-connected `χ`, Binder `U = ⟨m⁴⟩/⟨m²⟩²` (conventions:
  `docs/specs/binning-observables.md`).
- Error analysis: streaming `LogBinner` (log-binning plateau errors + `τ_int`,
  O(levels) memory), `BinStore` + leave-one-bin-out `jackknife` for derived
  quantities.
- `TiledHamiltonian`: the fitted SCE unfolded onto an `N₁×N₂×N₃` supercell from the
  public `multipole_terms` introspection (per-site integer `shifts`, toroidal wrap),
  with template-once + CSR-instance memory layout and the `(4π)^(body/2)` scale
  applied exactly once. Supports self-image (`AllImages`) clusters when `dims` keeps
  the images distinct sites.
- The 4-function energy contract: `total_energy`, `site_coeffs!` (leave-one-out
  coefficients — exact single-spin `delta_energy` for any body order), and
  `site_gradient` (on-sphere, via `Harmonics.grad_Zlm_unsafe`).
- Package scaffold: module skeleton, temperature control (`KB_EV`, `resolve_kt` —
  kelvin XOR model-energy-unit keywords), test harness (`TEST_MODE`
  default/all/unit/aqua/jet with Aqua + JET), Documenter docs skeleton.

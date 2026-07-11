# SCEMonteCarlo.jl — specification

Full classical spin Monte Carlo for fitted SCE models from `SCEFitting.jl`.
Self-contained core (no Carlo.jl), Threads parallelism, own binning analysis.
This file tracks the architecture and public API; the decision records live in
`docs/specs/*.md`. Validated end-to-end on the Nd₂Fe₁₄B l02 refit (68 atoms,
4692 terms): 4×4×4 tiling in 0.01 s / 7.8 MB of index arrays, machine-precision
training-cell and periodic-replication gates, and an 8-rung PT run recovering the
ferrimagnetic Nd-vs-Fe order at 250 K.

## Module layout

| File | Contents |
|---|---|
| `src/units.jl` | `KB_EV`, `resolve_kt` (kelvin XOR model-energy-unit control) |
| `src/hamiltonian.jl` | `ScaledTerm`, `TiledHamiltonian` (supercell tiling, CSR instance/site adjacency), `site_index` |
| `src/energy.jl` | the 4-function energy contract: `total_energy`, `site_coeffs!`, `delta_energy`, `site_gradient` |
| `src/binning.jl` | `LogBinner`, `BinStore`, `jackknife` |
| `src/observables.jl` | `Observable`, `Evaluable`, standard sets |
| `src/state.jl` | `SpinConfig`, `ChainState`, `SweepScratch` |
| `src/updates.jl` | Metropolis (adaptive step), overrelaxation, compound sweeps |
| `src/run.jl` | `run_mc` (single T + annealing), `TempResult`, `MCResult` |
| `src/pt.jl` | `run_pt` (replica exchange over threads), `PTResult` |
| `src/checkpoint.jl` | JLD2 schema v1, `resume` |
| `src/geometry.jl` | `supercell_crystal`, `to_matrix`/`from_matrix` |

## Dependency boundary

Reads a fitted model only through `SCEFitting`'s public surface
(`multipole_terms`, `n_atoms`, `intercept`, `SCEFitting.load`, `Lattice`/`Crystal`,
`SCEFitting.Harmonics`). The `(4π)^(body/2)` per-term scale is applied exactly once,
in the `TiledHamiltonian` constructor. The MC core is geometry-free (integer site
topology only); geometry helpers take an explicit `Crystal`.

## Public API

Exported: `KB_EV`, `TiledHamiltonian`, `n_sites`, `total_energy`, `Observable`,
`Evaluable`, `ObservableStat`, `standard_observables`, `standard_evaluables`,
`run_mc`, `MCResult`, `TempResult`, `run_pt`, `PTResult`, `resume`,
`supercell_crystal`.

Public, unexported (`SCEMonteCarlo.<name>`): `resolve_kt`, `ScaledTerm`,
`SpinConfig`, `site_index`, `site_atom`, `site_coeffs!`, `delta_energy`,
`site_gradient`, `LogBinner`, `std_error`, `tau_int`, `BinStore`, `bin_means`,
`jackknife`, `ChainState`, `SweepScratch`, `metropolis_sweep!`, `overrelaxation_sweep!`,
`to_matrix`, `from_matrix`.

## Design record index

- `docs/specs/hamiltonian-tiling.md` — supercell unfolding, CSR memory layout
- `docs/specs/updates-stationarity.md` — Metropolis/OR stationarity, adaptive-step freeze
- `docs/specs/binning-observables.md` — C/χ/U conventions (authoritative), log-binning, jackknife
- `docs/specs/pt-threads-determinism.md` — lane/RNG discipline, bit-reproducibility
- `docs/specs/checkpoint-schema.md` — JLD2 schema v1

# SCEMonteCarlo.jl — specification

Full classical spin Monte Carlo for fitted SCE models from `SCEFitting.jl`.
Self-contained core (no Carlo.jl), Threads parallelism, own binning analysis.
This file tracks the architecture and public API as milestones land; the
decision records live in `docs/specs/*.md`.

## Module layout

| File | Contents |
|---|---|
| `src/units.jl` | `KB_EV`, `resolve_kt` (kelvin XOR model-energy-unit control) |
| `src/hamiltonian.jl` | `ScaledTerm`, `TiledHamiltonian` (supercell tiling, CSR instance/site adjacency), `site_index` *(M1)* |
| `src/energy.jl` | the 4-function energy contract: `total_energy`, `site_coeffs!`, `delta_energy`, `site_gradient` *(M1)* |
| `src/binning.jl` | `LogBinner`, `BinStore`, `jackknife` *(M2)* |
| `src/observables.jl` | `Observable`, `Evaluable`, standard sets *(M2)* |
| `src/state.jl` | `SpinConfig`, `ChainState`, `SweepScratch` *(M3)* |
| `src/updates.jl` | Metropolis (adaptive step), overrelaxation, compound sweeps *(M3–M4)* |
| `src/run.jl` | `run_mc` (single T + annealing), `TempResult`, `MCResult` *(M3)* |
| `src/pt.jl` | `run_pt` (replica exchange over threads), `PTResult` *(M5)* |
| `src/checkpoint.jl` | JLD2 schema v1, `resume` *(M6)* |
| `src/geometry.jl` | `supercell_crystal`, `to_matrix`/`from_matrix` *(M7)* |

## Dependency boundary

Reads a fitted model only through `SCEFitting`'s public surface
(`multipole_terms`, `n_atoms`, `intercept`, `SCEFitting.load`, `Lattice`/`Crystal`,
`SCEFitting.Harmonics`). The `(4π)^(body/2)` per-term scale is applied exactly once,
in the `TiledHamiltonian` constructor. The MC core is geometry-free (integer site
topology only); geometry helpers take an explicit `Crystal`.

## Public API

Exported: `KB_EV`, `TiledHamiltonian`, `n_sites`, `total_energy`, `Observable`,
`Evaluable`, `ObservableStat`, `standard_observables`, `standard_evaluables`,
`run_mc`, `MCResult`, `TempResult`, `run_pt`, `PTResult`, `resume`
*(more added per milestone)*.

Public, unexported (`SCEMonteCarlo.<name>`): `resolve_kt`, `ScaledTerm`,
`SpinConfig`, `site_index`, `site_atom`, `site_coeffs!`, `delta_energy`,
`site_gradient`, `LogBinner`, `std_error`, `tau_int`, `BinStore`, `bin_means`,
`jackknife`, `ChainState`, `SweepScratch`, `metropolis_sweep!`, `overrelaxation_sweep!`.

## Design record index

- `docs/specs/hamiltonian-tiling.md` — supercell unfolding, CSR memory layout *(M1)*
- `docs/specs/updates-stationarity.md` — Metropolis/OR stationarity, adaptive-step freeze *(M4)*
- `docs/specs/binning-observables.md` — C/χ/U conventions (authoritative), log-binning, jackknife *(M2)*
- `docs/specs/pt-threads-determinism.md` — lane/RNG discipline, bit-reproducibility *(M5)*
- `docs/specs/checkpoint-schema.md` — JLD2 schema v1 *(M6)*

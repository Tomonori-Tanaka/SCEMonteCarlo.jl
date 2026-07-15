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
| `src/hamiltonian.jl` | `ScaledTerm`, `TiledHamiltonian` (supercell tiling, CSR instance/site adjacency, `site_active`/`n_active` — non-magnetic sites are frozen and excluded, precompiled sparse contraction programs, conflict-graph coloring for parallel sweeps), `site_index` |
| `src/energy.jl` | the 4-function energy contract: `total_energy`, `site_coeffs!`, `delta_energy`, `site_gradient` (program kernels + bitwise-gated rank-generic reference kernels) |
| `src/binning.jl` | `LogBinner`, `BinStore`, `jackknife` |
| `src/observables.jl` | `Observable`, `Evaluable`, standard sets |
| `src/state.jl` | `SpinConfig`, `ChainState` (chain + per-site RNG streams), `SweepScratch` |
| `src/updates.jl` | Metropolis (adaptive step), overrelaxation, compound sweeps — color-ordered, serial or `sweep_tasks`-parallel with bit-identical results |
| `src/gpu/*.jl` | GPU Metropolis prototype (KernelAbstractions, backend supplied by the caller): `philox.jl` keyed Philox4x32-10 stream, `zlm_device.jl` bitwise device tesseral row, `gpu_hamiltonian.jl`/`gpu_state.jl` device tables + chain state, `gpu_sweep.jl` fused kernel + drivers + keyed serial reference |
| `src/minimize.jl` | `minimize_energy` (on-sphere BB descent), `find_ground_state` (multi-start anneal + polish), `GroundStateResult` |
| `src/run.jl` | `run_mc` (single T + annealing), `TempResult`, `MCResult` |
| `src/pt.jl` | `run_pt` (replica exchange over threads), `PTResult` |
| `src/checkpoint.jl` | JLD2 schema v1, `resume` |
| `src/geometry.jl` | `supercell_crystal`, `to_matrix`/`from_matrix` |
| `src/reduce.jl` | `reduce_cell`/`ReducedCell` — verified re-expression of a supercell-fitted model in a user-chosen smaller cell |

## Dependency boundary

Reads a fitted model only through `SCEFitting`'s public surface
(`multipole_terms`, `n_atoms`, `intercept`, `SCEFitting.load`, `Lattice`/`Crystal`,
`SCEFitting.Harmonics`). The `(4π)^(body/2)` per-term scale is applied exactly once,
in the `TiledHamiltonian` constructor. The MC core is geometry-free (integer site
topology only); geometry helpers take an explicit `Crystal`.

## Public API

Exported: `KB_EV`, `TiledHamiltonian`, `n_sites`, `total_energy`, `Observable`,
`Evaluable`, `ObservableStat`, `standard_observables`, `standard_evaluables`,
`run_mc`, `MCResult`, `TempResult`, `run_pt`, `PTResult`, `minimize_energy`,
`find_ground_state`, `GroundStateResult`, `resume`, `supercell_crystal`,
`ReducedCell`, `reduce_cell`.

Public, unexported (`SCEMonteCarlo.<name>`): `resolve_kt`, `ScaledTerm`,
`SpinConfig`, `site_index`, `site_atom`, `site_coeffs!`, `delta_energy`,
`site_gradient`, `LogBinner`, `std_error`, `tau_int`, `BinStore`, `bin_means`,
`jackknife`, `ChainState`, `SweepScratch`, `metropolis_sweep!`, `overrelaxation_sweep!`,
`to_matrix`, `from_matrix`, `GPUTiledHamiltonian`, `GPUChainState`,
`gpu_metropolis_sweep!`, `gpu_run_sweeps!`, `to_host!`.

## Design record index

- `docs/specs/hamiltonian-tiling.md` — supercell unfolding, CSR memory layout
- `docs/specs/updates-stationarity.md` — Metropolis/OR stationarity, adaptive-step freeze
- `docs/specs/binning-observables.md` — C/χ/U conventions (authoritative), log-binning, jackknife
- `docs/specs/pt-threads-determinism.md` — lane/RNG discipline, bit-reproducibility
- `docs/specs/checkpoint-schema.md` — JLD2 schema v1
- `docs/specs/cell-reduction.md` — verified reduction to a user-chosen smaller cell
- `docs/specs/ground-state-search.md` — on-sphere descent, thermal cycling, multi-start determinism
- `docs/specs/gpu-feasibility.md` — GPU-port assessment: strategy, measured baseline, go/no-go
- `docs/specs/gpu-prototype.md` — GPU Metropolis prototype: keyed RNG layout, determinism contract, kernel shape, A100 readout

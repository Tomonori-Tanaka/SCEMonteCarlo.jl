# API reference

```@meta
CurrentModule = SCEMonteCarlo
```

## Module

```@docs
SCEMonteCarlo
```

## Hamiltonian

```@docs
TiledHamiltonian
n_sites
site_index
site_atom
ScaledTerm
SpinConfig
```

## External field

The `magmoms` / `field` keywords of [`TiledHamiltonian`](@ref) (see the
[external field](guide/field.md) guide).

```@docs
has_field
zeeman_energy
```

## Cell reduction

```@docs
reduce_cell
ReducedCell
```

## Energy contract

```@docs
total_energy
site_coeffs!
delta_energy
site_gradient
energy_gradient!
```

## Running

```@docs
run_mc
MCResult
TempResult
run_pt
PTResult
resume
```

## Ground-state search

```@docs
minimize_energy
find_ground_state
GroundStateResult
```

## Chain internals

```@docs
ChainState
SweepScratch
metropolis_sweep!
overrelaxation_sweep!
```

## GPU

The chain-level device sweep (see the [GPU guide](guide/gpu.md)). The gradient
tier (`SCEMonteCarlo.gpu_energy_gradient!`, `SCEMonteCarlo.GPUGradientScratch`,
`SCEMonteCarlo.gpu_zlm_rows!`) is public but unexported — the inter-package seam
for dependent packages' GPU dynamics.

```@docs
GPUTiledHamiltonian
GPUChainState
gpu_metropolis_sweep!
gpu_run_sweeps!
gpu_run_pt
to_host!
```

## Observables

```@docs
Observable
Evaluable
ObservableStat
standard_observables
standard_evaluables
```

## Binning

```@docs
LogBinner
std_error
tau_int
BinStore
bin_means
jackknife
```

## Geometry / I/O

```@docs
supercell_crystal
to_matrix
from_matrix
```

## Units

```@docs
KB_EV
MU_B_EV_T
resolve_kt
```

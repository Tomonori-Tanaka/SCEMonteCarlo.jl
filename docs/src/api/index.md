# API reference

The constrained Monte Carlo API is on [its own page](constrained_mc.md); the chain
internals and the GPU sweep API are on the [internals page](internals.md).

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
[external field](../guide/field.md) guide).

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

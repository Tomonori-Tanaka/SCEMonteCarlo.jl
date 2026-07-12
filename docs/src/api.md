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
resolve_kt
```

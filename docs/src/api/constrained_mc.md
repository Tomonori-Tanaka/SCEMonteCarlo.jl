# API: constrained Monte Carlo

```@meta
CurrentModule = SCEMonteCarlo
```

Sampling under a fixed direction of the (weighted) magnetization, for the
finite-temperature anisotropy. `total_torque` is public but unexported.

```@docs
run_cmc
CMCResult
MagnetizationConstraint
cmc_observables
cmc_evaluables
torque_observable
total_torque
```


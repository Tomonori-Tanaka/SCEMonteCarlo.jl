# API: constrained Monte Carlo

```@meta
CurrentModule = SCEMonteCarlo
```

Sampling under a fixed direction of the (weighted) magnetization, for the
finite-temperature anisotropy. `total_torque` is public but unexported. Usage:
the [constrained-MC guide](../guide/constrained_mc.md); the algorithm: the
[theory page](../theory/constrained_mc.md).

```@docs
run_cmc
CMCResult
MagnetizationConstraint
cmc_observables
cmc_evaluables
torque_observable
total_torque
```


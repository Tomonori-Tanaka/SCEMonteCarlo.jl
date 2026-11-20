# API: chain internals and GPU

```@meta
CurrentModule = SCEMonteCarlo
```

## Chain internals

```@docs
ChainState
SweepScratch
metropolis_sweep!
overrelaxation_sweep!
```

## GPU

The chain-level device sweep (see the [GPU guide](../guide/gpu.md)). The gradient
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


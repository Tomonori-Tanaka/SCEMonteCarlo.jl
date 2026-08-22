# Bench log

Tracked historical record (append-only). Append one entry per measurement
campaign. The old repo's log (#1–#10) stayed
with SLCEMonteCarlo.jl; this file restarts at #1 for the revived spin-only
package.

## #1 — 2026-08-13: revival GPU re-validation on kugui (i1accs 880108/880109)

- Code: SCEMonteCarlo `373a4b0`, SCEFitting `f9db40a` (git-clone deploy, clean
  trees). Env: fresh `bench/gpu` instantiate — CUDA.jl v6.2.1, KA 0.9.42,
  CUDA_Runtime 12.6 via the machine-global pin. A100-SXM4-40GB.
- `bench_gpu.jl 16 100`: device correctness (repeat-identity + drift) pass,
  acceptance matches CPU everywhere; GR9 gradient bitwise OK on CUDA.
- Ratios (same-node cpu-4T / device): bcc-16³ 3.1×, 4³ 8.0×, 8³ **17.1× (GO)**,
  16³ 22.3×. Device 8³ 3.23 ms/sweep, 16³ 21.95 ms/sweep (tables 0.85 GiB).
- Full record: `docs/specs/gpu-prototype.md` G6 "Revival re-validation".

## #2 — 2026-08-14: G8 gpu_run_pt device validation on kugui (i1accs 880608)

- Code: SCEMonteCarlo `559ffd5` (G8 = `e039bfa` + bench script), SCEFitting
  `f9db40a`. Same bench/gpu env as #1 (CUDA 6.2.1, runtime 12.6 pin). A100.
- `bench/bench_gpu_pt.jl 8 500`: repeat-run bitwise, exchange-free composition
  bitwise, rung-energy ordering, swap acceptances 0.03/0.12/0.09 — ALL PASS.
- Perf @ 8³: single chain 3.08 ms/sweep; ladder 4.17 ms/sweep/rung → ratio
  1.36 at 200 sweeps (per-rung host setup + boundary renorm, amortizes away).
- Full record: `docs/specs/gpu-prototype.md` G8 device-validation bullet.

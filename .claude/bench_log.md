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

## #3 — 2026-08-22: external field — sweep cost of the body-1 Zeeman templates (local CPU)

- Code: working tree of the zeeman-field spec (after `934db67`), SCEFitting
  `c213ccd`. Local Apple-silicon Mac, `julia -t 4`, `bench/bench_sweeps.jl 50 8 2`
  (the "+ field" reports added in this spec: same fixtures, same `n_active`,
  `magmoms` on every magnetic atom, boron 0, `B = 2 T ẑ`).
- bcc Fe 8³ (light kernel, adjacency 8 → 9): metropolis 95.6 → 102.3 ns/attempt
  (+7 %), OR 82.7 → 90.4 (+9 %) — proportional to the extra CSR entry per visit.
- Nd₂Fe₁₄B 2³ (heavy kernel, adjacency 146.6 → 147.6): metropolis 915 → 885
  ns/attempt, OR 886 → 862 (within run-to-run noise; the one extra general-branch
  entry is < 1 % of a visit).
- allocs/sweep = 0 in every case (the pass/fail gate). No kernel changed; the
  cost is data-only (one body-1 instance per moment-carrying site).
- CUDA re-validation: entry #4 below.

## #4 — 2026-08-22: external field — device validation on kugui (i1accs 887569)

- Code: SCEMonteCarlo `855c17d` + `bench/bench_gpu_zeeman.jl` (committed as
  `b607622`), SCEFitting `67c3946`; both kugui checkouts fast-forwarded by
  `git pull`. Env: the existing `bench/gpu` env (CUDA.jl 6.2.1, KA 0.9.42,
  runtime 12.6 via the machine-global pin, driver 560.35.3). A100-SXM4-40GB.
- `bench_gpu_zeeman.jl 8 100`: **ALL GATES PASS on CUDA** — per case (dimer with
  a Zeeman-only site + frozen zero-moment site, biquadratic + field, all-body-1,
  bcc Fe 8³ + field, Nd₂Fe₁₄B nbody=2 2³ + field with boron 0): repeat-run
  identity, drift ≤ 6.8e-13 on |E| = 258 eV (0 on bcc 8³), device acceptance
  within 0.023 of the CPU path, frozen sites bitwise, gradient **bitwise vs the
  lane reference** (GR9 holds with the body-1 templates; fallback unused) and
  ≤ 4.6e-16 scaled vs host `energy_gradient!` with tangency; Langevin law
  ⟨e₃z⟩ = 0.6060 vs L = 0.6136 (1.0σ, σ = 0.0078 — same numbers as the local
  KA-CPU run, i.e. the device RNG stream reproduces the host's); GPU-PT with a
  field: repeat / resume bitwise, `:M_B` present, field-free twin refused, swap
  acceptances 0.281 / 0.548.
- Informational: device sweeps vs the host keyed reference are NOT bitwise on
  CUDA (expected — libm), but the accepted-move counts agree exactly on every
  case (30/30, 35/35, 40/40, 3914/3914, 1500/1500): the divergence is last-ulp
  energy arithmetic, never an accept decision, over 5 sweeps.
- Sweep cost with vs without a field (device ms/sweep, ws = 32, 100 sweeps):
  bcc Fe 8³ 0.092 → 0.094 (+2.9 %), Nd₂Fe₁₄B nbody=2 4³ 1.555 → 1.602 (+3.0 %);
  acceptance unchanged (0.50 / 0.30). Job dir `~/mc/scemc_zeeman_val/` on kugui.

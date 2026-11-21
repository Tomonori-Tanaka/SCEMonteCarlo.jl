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

## #5 — 2026-11-21: constrained MC — first cost of `cmc_sweep!` (local CPU)

- Code: working tree of the constrained-mc spec, M4 (after `f6ef95c`), SCEFitting
  `4d2f86e`. Local Apple-silicon Mac, `julia -t 4`, `bench/bench_sweeps.jl 50 8 2`
  with the new `cmc_sweep! (serial)` row (uniform weights, n̂ = (θ, φ) = (0.6, 0.35), start
  from the constrained default pattern, frozen step 0.6, kT = 0.025 eV).
- Existing sweeps: no kernel and no sweep loop changed in this diff (the clamp of
  `_adapt_step!` / `_gpu_adapt_step!` only got the name `_STEP_MIN`, same value).
  bcc Fe 8³ metropolis 101–104 ns/attempt, OR 83–94; Nd₂Fe₁₄B 2³ metropolis
  1009–1026, OR 1017–1047. Any difference from #3 is run-to-run noise or the
  environment (machine state, SCEFitting / Julia versions), not this diff.
- `cmc_sweep!` per attempt (two runs): bcc Fe 8³ 258–263 ns (2.5× metropolis,
  acc 0.47); bcc Fe + field 258–309 ns; Nd₂Fe₁₄B 2³ 1923–1992 ns (1.9×, acc
  0.27); nbody=3 triplet 6261–6961 ns (2.0×, acc 0.17). The ~2× is the pair
  attempt itself — two leave-one-out coefficient builds and two tesseral rows per
  attempt; serial only by design (D3).
- allocs/sweep = 0 in every case, including `cmc_sweep!` (G16).

## #6 — 2026-11-21: constrained MC M5 — mover refactor of the run driver (local CPU)

- Code: before = HEAD `6cc6771` (clean tree); after = working tree of the
  constrained-mc spec, M5 (mover methods of `_compound_sweep!` /
  `_run_temperature!` / `_mc_loop!` in `run.jl`, `run_cmc`; `cmc_sweep!` renamed
  `_cmc_sweep!`). SCEFitting `4d2f86e`. Local Apple-silicon Mac, Julia 1.12.7,
  `julia -t 4`, `bench/bench_sweeps.jl 50 8 2` and `bench/bench_run.jl` (500
  measure sweeps, 8³ / 2³). An unrelated Julia process of another session was
  running during both measurements, so differences of 10–25 % are machine state.
- Sweeps (ns/attempt, before → after): bcc Fe 8³ metropolis 106.0 → 96.0, OR
  94.5 → 92.1, `_cmc_sweep!` 269.6 → 251.4; Nd₂Fe₁₄B 2³ metropolis 1274 → 977,
  OR 1209 → 1036, CMC 2386 → 1979; nbody=3 triplet metropolis 4496 → 3613, CMC
  8439 → 6592; bcc Fe + field metropolis 143.7 → 103.0, CMC 358.6 → 267.9;
  Nd₂Fe₁₄B + field metropolis 1286 → 1152, CMC 2412 → 2075. No sweep kernel and
  no sweep loop changed in this diff (the "after" being uniformly faster is the
  machine, not the code). allocs/sweep = 0 in every row.
- End to end (ns/attempt, before → after): `run_mc` bcc Fe 114.0 / 105.8 →
  111.2 / 106.3 (measure_interval 1 / 10); Nd₂Fe₁₄B 1073.5 / 1084.1 → 1053.1 /
  987.9; `run_pt` 8 rungs bcc Fe ntasks = 1 138.5 → 121.5, ntasks = 4 48.2 → 45.9.
  The mover dispatch is one static method call per sweep (`_SweepMover` is a
  singleton; the driver methods are specialized on it), invisible at this
  resolution. The existing driver paths are bitwise unchanged (G15 pins,
  `test/unit/test_regression_pins.jl`).

## #7 — 2026-11-21: constrained MC `:torque_frame` — one gradient shared with `:torque` (local CPU)

- What changed: `cmc_observables` gained `:torque_frame`, which reads the torque
  from a memo shared with `:torque` (reused only for a bitwise-identical
  configuration) instead of evaluating a second gradient. The sweep kernels are
  untouched; the measurement path adds an O(n_sites) compare + copy and a lock.
- Measured (M4 Mac, Julia 1.12.7, `-t 4`; F2g 2-atom model, dims (8,8,8) = 1024
  sites; 50 random configurations, best of 5; script in the session scratchpad,
  not kept): all default observables per measurement, HEAD `93257b7` (5
  observables) 255.9 us → working tree (6 observables, `:torque_frame` added)
  256.1 us; one `total_torque` 261.1 / 252.1 us. The frame projection is free;
  without the memo it would have doubled the measurement cost.

## #8 — 2026-11-21: constrained MC — cost of the torque measurement; chain-state torque path (local CPU)

- Question (review finding O5): each torque measurement went through the public
  `total_torque(H, config)`, which rebuilds the tesseral rows and allocates the
  gradient vector. Measure first, then decide on an internal path.
- Setup: Apple M4 (16 GB), macOS 15.7, Julia 1.12.7, `julia -t 4`, SCEFitting at
  its working tree of the day. Before = HEAD `377cfdf`; after = working tree
  (chain-state path). Production models of the Nd₂Fe₁₄B 68-atom cell (pruned,
  θ0): **l02** (574 pair templates, nlm 4) and **l044_c4.4** (990 pair + 3484
  three-body templates, nlm 9), both isotropic (their torque vanishes, the cost
  does not). Supercell 4×4×4 = 4352 sites (4096 active, B inactive). Constraint
  n̂(0.6, 0.35), `:auto` weights, 300 K. Partial torques: two subsets covering the
  model (l044: body 2 / body 3; l02: pairs touching Nd / the rest). Scripts in
  the session scratchpad, not kept. Other sessions' processes (2–3 busy cores at
  times) loaded the machine during several runs, so absolute times move by up to
  50 % between runs; same-process ratios are the reliable numbers.
- Per call (best of 5–9): one CMC sweep l02 1.40 ms / l044 28–42 ms;
  `total_torque` l02 0.63 ms (288 KiB) / l044 14.6–21.2 ms (448 KiB), of which the
  row rebuild `_zrows` 0.12 ms (160 KiB) / 0.34–0.48 ms (320 KiB). One gradient
  pass = 0.45–0.52 of a constrained sweep on both models.
- End to end, HEAD (`run_cmc`, l044 200 / l02 2000 measure sweeps, measurement
  share = (t − t_baseline)/t with an `:energy`-only baseline, min of 2–3):

  | set | k = 1 | k = 5 | k = 10 |
  |---|---|---|---|
  | default (`:torque` + `:torque_frame`, one gradient), l044 | 32–34 % | 10 % | 5 % |
  | default, l02 | 33 % | 9 % | 4 % |
  | default + 2 partials, l044 | 53 % | 17 % | 9 % |
  | default + 2 partials, l02 | 55 % | 20 % | 11 % |

  Allocation 290 KiB (l02) / 450 KiB (l044) per measurement, ×3 with two
  partials (1.7 GiB per 2000-sweep l02 run); GC time ≤ 0.05 s per run (≤ 1 %).
  Lanes (k = 1, 1/2/4 lanes): the measurement share stays ~30 % per lane; two
  duplicate lanes sharing ONE `cmc_observables` set ran +20 % slower than with
  a set each on l02 (2.73 vs 2.28 s: the memo misses when the lanes'
  measurements interleave), +3 % on l044.
- Decision: implement the internal path. The bulk of the cost is the gradient
  itself (no caching removes it — the `measure_interval` advice is the lever),
  but the rebuild + allocation is 21–25 % of a measurement on the light pair
  model, the 1.7 GiB/run of garbage is avoidable, and a per-chain computation
  removes the memo's lock and its shared-set miss.
- After, same process A/B of one measurement's torque work (HEAD path = one
  `total_torque` per Hamiltonian vs the chain-state `_observe`):
  l02 default 0.633 → 0.500 ms (−21 %), +2 partials 1.623 → 1.216 ms (−25 %);
  l044 default 14.66 → 14.24 ms (−3 %), +2 partials 30.57 → 29.20 ms (−4.5 %).
  Allocation per measurement 295 KiB → 736 B (default) and 886 KiB → 1.5 KiB
  (+2 partials) — the remainder is the boxing of the observables' return values,
  not the torque. End to end l02, k = 1: measurement share 33.1 → 28.7 %
  (default), 55.1 → 48.3 % (+2 partials); total allocation of the 2000-sweep run
  570 MiB → 7.7 MiB and 1.7 GiB → 9.4 MiB; GC time 0.019 / 0.053 s → 0. l044 at
  k = 1: share ~34 % → ~34 % within noise, allocation 91 MiB → 3.4 MiB per run.
  Two duplicate lanes on one shared set: 2.96 vs 2.92 s with own sets (l02).
- Results bitwise unchanged: digests (every statistic's mean/err/τ/count,
  acceptance, step, drift, final configuration) of the measured runs on both
  models (default and +2 partials at k = 1/5/10, 1/2/4 lanes, the shared set)
  identical before/after, and a separate 12-run digest on l02 2×2×2 (ladder;
  partials; field + `:magmoms` + partials; signed weights + `carryover = false`
  + `init`; hybrid Nd weight 0 at the pole; 4 lanes `ntasks = 2`; duplicate lanes
  on one shared set; checkpointed run and its resume; `run_mc` standard and with
  the CMC set; `run_pt` with the CMC set) identical between `git archive HEAD`
  (+ Manifest copy) and the working tree.
- `bench/bench_sweeps.jl 50 8 4` (sweep kernels untouched; run under load):
  bcc Fe 8³ metropolis 97.7 ns/attempt, OR 92.0, `_cmc_sweep!` 239.7;
  Nd₂Fe₁₄B fixture 4³ metropolis 925.1, OR 902.6, CMC 1775.1; nbody=3 triplet
  metropolis 3231.9, CMC 6178.9; bcc Fe + field CMC 253.8; Nd₂Fe₁₄B + field
  metropolis 991.1, CMC 1933.5. allocs/sweep = 0 in every row.

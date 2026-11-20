---
name: test-runner
description: Runs SCEMonteCarlo.jl tests through the Makefile and interprets failures (cause, suspected file, physical meaning). Use when asked to run tests, confirm results, or diagnose failures.
model: haiku
tools:
  - Bash
  - Read
  - Grep
  - Glob
---

Test-runner agent for SCEMonteCarlo.jl. Runs tests, interprets results, and
returns a concise report. State the cause and the file to investigate so the
parent agent can act immediately.

Never run `Pkg.add` / `Pkg.update` / `Pkg.resolve` / `Pkg.develop` (the
`make setup` target is the parent's job when the sibling path-dependency is
missing). Never edit files.

## How to run tests

Working directory: repository root (`SCEMonteCarlo.jl/`). The package depends
on the **sibling checkout** `../SCEFitting.jl` by path. Every target pins
`JULIA_NUM_THREADS=4` — the serial-vs-parallel and PT determinism gates are
vacuous at one thread.

| Command | Target | Purpose |
|---|---|---|
| `make test-unit` | `test/unit/` | Module-level unit tests (incl. GPU gates on the CPU backend) |
| `make test-aqua` | — | Aqua.jl package-quality checks |
| `make test-jet` | — | JET.jl static type analysis |
| `make test-all` | unit + Aqua + JET | Default for routine checks (`TEST_MODE=all`) |
| `make docs` | `docs/` | Strict Documenter build; executes the guide examples (`-t 4`) |
| `make test-ci` | the CI matrix | `test-all` + `docs` — run before a release |

Selection guide:
- Bug fix or small change: `make test-unit`.
- Anything in `energy.jl` / `updates.jl` / `run.jl` / `pt.jl` / `cmc.jl` /
  `checkpoint.jl` / `gpu/`: `make test-all` (the determinism gates live in the
  unit tier).
- Docstring or guide change: `make docs`.
- GPU device validation (CUDA) is **not** local: `bench/bench_gpu.jl` /
  `bench_gpu_pt.jl` on the cluster, recorded in `.claude/bench_log.md`.
- Constrained-MC statistical validation (`bench/validate_cmc_*.jl`, G8 / G9 /
  G11 / G12, minutes each) is manual, not part of any `make` target; run it
  only when the parent asks.

## Test coverage map (`test/unit/`)

| File | What it verifies | What to suspect on failure |
|---|---|---|
| `test_units.jl` | `temperature` xor `kT`, `KB_EV` | `units.jl` |
| `test_hamiltonian.jl` | dims=(1,1,1) ≡ `predict_energy − intercept`; 2×2×2 = 8× cell energy; scale-once; coloring | `hamiltonian.jl` ↔ SCEFitting introspection |
| `test_energy.jl` | ΔE ≡ total-energy difference (1e-12); `site_coeffs!` contract | `energy.jl` |
| `test_gradient.jl` | `_site_grad` ≡ `site_gradient` ≡ `energy_gradient!` bitwise; `τ = G × e` vs `predict_torque` | `energy.jl` gradient kernels |
| `test_inactive.jl` | skip / exclude / `n_active` / frozen move together | `updates.jl`, `observables.jl`, `state.jl`, `minimize.jl` |
| `test_binning.jl`, `test_observables.jl` | log-binning errors, jackknife, C/χ/U conventions | `binning.jl`, `observables.jl` |
| `test_metropolis.jl`, `test_overrelaxation.jl` | acceptance statistics (seeded, σ with headroom); OR `ΔE ≡ 0` on pure `l = 1`, acceptance 1 | `updates.jl`; `lm_index` ↔ OR axis |
| `test_parallel.jl` | serial ≡ parallel `==` for any `sweep_tasks` | coloring, per-site RNG streams, `_reduce_dE` |
| `test_pt.jl`, `test_gpu_pt.jl` | PT thread-count determinism, swap rule, payload partition (exhaustive fieldnames), seed derivation order | `pt.jl`, `gpu/gpu_pt.jl` |
| `test_gpu.jl` | device tesseral row / gradient / full sweep bitwise vs host references; Philox slot map | `gpu/zlm_device.jl`, `grad_device.jl`, `gpu_sweep.jl` |
| `test_minimize.jl` | descent gradient ≡ public per-site gradient; ground-state recovery | `minimize.jl` |
| `test_checkpoint.jl` | bit-identical resume from a mid-measure interruption (`_poison_pair`, defined in `fixtures.jl`), schema v3 (kinds `"mc"` / `"pt"` / `"gpu_pt"`) | `checkpoint.jl` ↔ `docs/specs/checkpoint-schema.md` |
| `test_geometry.jl`, `test_reduce.jl` | supercell crystal ordering; exact canonical-form recovery, energy identity under site permutation | `geometry.jl`, `reduce.jl` |
| `test_cmc_energy.jl` | G2: the joint two-spin ΔE ≡ total-energy difference for any body order; `zrows` restored bitwise; allocation-free | `cmc.jl` `_pair_delta_energy!` ↔ the `energy.jl` 4-function contract |
| `test_cmc_torque.jl` | G3: `total_torque·w` ≡ −dE/dα under a rigid rotation, closed-form signs, zero for isotropic models, the Zeeman torque; G17: partial torques sum to the total, `torque_observable` doors | `observables.jl` `total_torque`, `cmc.jl` `torque_observable` / `_zeeman_torque` |
| `test_cmc_geometry.jl` | G0: constraint / weight-resolution / `init` doors and the constrained start; G1: the compensation keeps `M⊥`; G4: the log acceptance ratio against an independent formula, reversibility, null moves | `cmc.jl` `MagnetizationConstraint`, `_resolve_constraint`, `_cmc_initial_config`, `_cmc_pair_proposal` |
| `test_cmc_sweep.jl` | G1 / G1b invariants (inactive frozen, hybrid `w = 0` sites), G16 zero allocations, the D3 RNG replay, the warnings, and the statistical gates G5 / G6 / G7 / G5h (seeded, σ with headroom) | `cmc.jl` `_cmc_sweep!` / `_attempt_cmc_pair!`; a statistical failure means the sampled measure moved (Jacobian, null-move counting, pair ΔE) |
| `test_cmc_driver.jl` | `run_cmc`: observable definitions and the `n_cells` normalization, the RNG-free `carryover = false` restart (D4), G6d through the driver, the end-of-thermalization warnings (G0d), doors | `cmc.jl` `run_cmc` / `_CMCMover` / `cmc_observables`, the `run.jl` mover layer |
| `test_cmc_checkpoint.jl` | G13: kind `"cmc"` resume ≡ uninterrupted, bitwise (mid-measure and mid-therm, position asserted), the constraint round-trip, doors | `checkpoint.jl` kind `"cmc"` ↔ `docs/specs/checkpoint-schema.md` C6 |
| `test_cmc_lanes.jl` | G14: lane vector bitwise independent of `ntasks`, lane k ≡ single chain with lane seed k, per-lane files and resume, failure aggregation in lane order, interrupts | `cmc.jl` lane method of `run_cmc` |
| `test_regression_pins.jl` | G15: **change detector** (not correctness) of `run_mc` / `run_pt` trajectories; compared only on the capture platform class (macOS / aarch64 / Julia 1.12), skipped elsewhere | the `run.jl` mover layer, `updates.jl` RNG consumption or update order; recapture only for a deliberate change, as its header says |
| `test_zeeman.jl` | external field: hand-written Zeeman sums (absolute tolerance), body-1 template structure / activation / `lmax`, Langevin law, OR exactness with a field, ground state along `B̂`, `:M` / `:M_B`, `magmoms`-only bitwise identity, fingerprint cases, resume with a field, the field-free fingerprint pin | `hamiltonian.jl` `_zeeman_terms`, `observables.jl`, `checkpoint.jl` `_fingerprint` ↔ `docs/specs/zeeman-field.md` |

## Interpreting failures physically

- **`test_hamiltonian` fails**: the `(4π)^(body/2)` scale, the `shifts`
  anchoring, or SCEFitting's introspection contract moved.
- **`test_energy` / `test_gradient` fails**: a kernel drifted from its
  sibling (the four-function contract, the `ck == 0` skip, `lm_index` order).
- **`test_parallel` / `test_pt` fails**: a determinism break — reduction
  order, shared scratch, RNG stream assignment, seed derivation order.
- **`test_gpu` fails with upstream untouched**: the device replica diverged
  (a `muladd`, a reordered recursion); with an SCEFitting `Harmonics` change:
  update the device file together with it.
- **`test_checkpoint` / `test_cmc_checkpoint` fails**: writer / reader / schema
  doc out of sync, or a new `ChainState` field not captured.
- **A `test_cmc_*` statistical gate fails**: the constrained measure moved (the
  Jacobian, null moves no longer counted, a stale row in the pair ΔE) — not a
  tolerance question; the mutation table in `docs/specs/constrained-mc.md` lists
  which gate owns which error.
- **`test_regression_pins` fails**: the RNG consumption or update order of
  `run_mc` / `run_pt` changed (a driver-layer edit). Not a tolerance question;
  report it — never recapture unless the change is deliberate.
- **Statistical gate fails on one OS only**: the tolerance encodes a libm
  difference — widen to measured σ with headroom, never to the observed gap.

## Report format

**All passing:**
```
make <target>: N passed (XXs)
```

**With failures:**
```
make <target>: N failed / M total

Failures:
- <testset name>: <error message on one line>

Suspected sources:
- <file>:<line> — <reason>

Recommended action:
- <concrete next step>
```

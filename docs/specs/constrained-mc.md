# Decision record — constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: in progress (spec `261121-constrained-mc`, agreed 2026-11-21). This
record is the standing contract for the CMC sampler; sections marked
*(pending)* are filled when the owning milestone lands. Owner: `src/cmc.jl`
(sampler, driver, observables), `src/observables.jl` (`total_torque`),
`src/checkpoint.jl` (kind `"cmc"`).

Primary source: P. Asselin et al., PRB 82, 054415 (2010), arXiv:1006.3507
(acceptance rule p. 7 step 7 and Eq. (A11); torque identity App. C, Eq. (C15)).
The weighted compensation follows Y. Toga et al., arXiv:1606.00333; the hybrid
(partly constrained) variant follows the VAMPIRE implementation used by
M. O. A. Ellis et al., arXiv:1706.05188.

## D1 — the constrained ensemble

Constrained sites `𝒞 = { s : site_active(s) and w_{a(s)} ≠ 0 }` with signed
per-cell-atom weights `w_a`; `M = Σ_{s∈𝒞} w_{a(s)} e_s`, `M_∥ = M·n̂`,
`M_⊥ = M − M_∥ n̂`. The sampled measure, exact at finite N, is

```
π(dξ) ∝ Π_s dΩ_s · exp(−βE(ξ)) · δ_{S²}(M̂(ξ), n̂),   δ_{S²}(M̂, n̂) = |M|² δ²(M_⊥) Θ(M_∥)
```

so that `F(n̂) = −kT ln P(n̂) + const`, `P` being the canonical density of `M̂`
per solid angle (Asselin's F). Inactive sites are excluded from `M`, from the
move set, and from every observable, and stay bitwise frozen. Signed weights
need no special case: the gauge map `ẽ_s = sign(w_s) e_s` sends the signed
problem exactly onto the `|w|` problem (compensation, sign preservation,
Jacobian, and proposal symmetry all commute with it).

**Theorem (I)** (the oracle of the geometry gate G5): when `E` is rotation
invariant (a constant, or isotropic couplings), `exp(−βE) Π dΩ_s` is invariant
under global rotations, so the constrained measure at `n̂` is the rotated image
of the one at any other direction; since `∫ dΩ_n̂ δ_{S²}(M̂, n̂) = 1`, averaging
over `n̂` recovers the unconstrained measure, and so the distribution of every
rotation-invariant quantity (`|M|`, `e_s·M̂`, `e_s·e_t`) under the constrained
measure equals its unconstrained distribution — derived without the Jacobian of
the pair move, which is what makes it an independent oracle for it.

Weights: `:auto` resolves to `H.magmoms` when present, else all ones;
`:magmoms` requires moments; an explicit vector is indexed by the cell atoms of
`H` (the reduced cell after `reduce_cell`). `magmoms ≥ 0` is enforced
(`zeeman-field.md` Z1), so signed weights only come from an explicit vector.

**Constraint, resolution, and initialization (landed M3, `src/cmc.jl`).**
`MagnetizationConstraint(direction; weights)` / `(θ, φ; weights)` normalizes the
direction (`n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`; zero, non-finite, or non-finite
angles are refused) and validates the weights (`:auto | :uniform | :magmoms` or
a vector of finite reals, copied); signed zeros are folded to `+0.0` so that
`==`, `isequal`, `hash` (field-wise) agree. Resolution re-checks finiteness.
Resolution against `H` (`_resolve_constraint`) refuses `:magmoms` without
moments, a weight vector of the wrong length, and `|𝒞| < 2`; it stores the
per-site weights with `site_w[s] = 0` off `𝒞` (inactive sites included).
The start of a constrained chain is deterministic and consumes no RNG:
the default pattern is `e_s = sign(w_s) n̂` on `𝒞` and `n̂` on every other site;
a given `init` is rotated rigidly (every site) by the minimal rotation taking its
`M̂` to `n̂`, refused when `|M| < 1e-8 Σ_{s∈𝒞}|w_s|`. When `M̂` is within 1e-8 of
`−n̂` the minimal rotation is ill-conditioned (its plane is rounding noise), so
the rotation is the half turn about `normalize(n̂ × x̂)` (`ŷ` when `n̂ ∥ x̂`)
followed by the residual minimal rotation — the half turn alone (to rounding)
when `M̂ = −n̂`. Rotated spins are re-normalized and clamped into `[-1, 1]`.

## D2 — the pair move *(proposal landed M3; the attempt M4)*

For a source `i ∈ 𝒞` and partner `j`: rotate `e_i` by the symmetric Rodrigues
proposal (no flips in CMC); compensate
`e_j⊥′ = e_j⊥ − (w_i/w_j)(e_i⊥′ − e_i⊥)`; null move unless `|e_j⊥′|² < 1`;
`e_j∥′ = σ_j √(1 − |e_j⊥′|²)` with `σ_j = (e_j∥ ≥ 0 ? +1 : −1)`; null move
unless `M_∥′ > 0`; accept with

```
ℓ = −βΔE + 2 ln(M_∥′/M_∥) + ln|e_j∥| − ln|e_j∥′|
```

`ΔE` is the exact joint difference by sequential leave-one-out (the single-site
contract of `hamiltonian-tiling.md` applied twice; any body order).

The proposal is the pure function `_cmc_pair_proposal` (M3): it returns
`(e_j′, ΔE, ℓ, M′)`, or `(e_j, NaN, −Inf, M)` for a null move, without touching
the state. `M′ = M + w_i(e_i′ − e_i) + w_j(e_j′ − e_j)` is formed once, as a
vector (M4): the same value decides `M_∥′ > 0`, enters the Jacobian, and is what
an accepted attempt stores, so the three agree bitwise. Both null tests are
written NaN-safe (`!(s2 < 1)`, `!(M_∥′ > 0)`), so a NaN anywhere in the inputs
is a null move. A current state with `M_∥ ≤ 0` (or
NaN) is a caller error and throws, rather than producing a `log` of a negative
ratio. A partner with `e_j·n̂ = 0` exactly gives `ℓ = −Inf` with a finite `ΔE`
(always rejected; the reverse move has zero density).

`e_j′` is re-normalized and clamped into `[-1, 1]`. The constructed
`e_j⊥′ + e_j∥′ n̂` is unit only to rounding, and its components overshoot 1 in a
measurable fraction of near-axis proposals (≈ 13 % of the UN-normalized
constructions in the M3 test that aims at `e_j′ = x̂`). The division by the
computed norm alone already brings every component into `[-1, 1]` (with
correctly rounded arithmetic the computed norm is ≥ every |component| and the
rounded quotient is monotone); the clamp is kept as a free guard. The bound
matters twice: the tesseral harmonics throw a `DomainError` for `|e·ẑ| > 1`
(the Legendre recursion `dnPl` reads only `u[3]`), and a checkpoint restore
refuses any component outside `[-1, 1]` (`_validate_unit_column`). The change is
deterministic and rounding-sized, so it cannot bias the measure at any
statistical resolution; `M_∥′` and the Jacobian use `e_j′·n̂` of the
re-normalized vector, so `ℓ` is the log ratio of the move actually made. A sign
flip of `e_j′·n̂` against `σ_j` is a null move (a guard: not reachable while
`s2 < 1`, since `|e_j∥′| ≥ ~1e-8` dwarfs any rounding).

**Null moves count as attempts.** Re-drawing after a geometric failure samples
`π(x)(1 − r(x))` instead of `π` (measured 8–13σ bias at N = 3 in the spec's
investigation).

## D3 — partner rule, sweep, RNG *(landed M4, `src/cmc.jl`)*

Partner uniform over `𝒞 \ {i}` (state-independent; a same-weight-class rule is
rejected because it conserves each class's transverse sum and breaks
ergodicity). Unlike U1 of `updates-stationarity.md`, the CMC sweep consumes RNG
for partner selection and is serial only — the global `M_∥` couples every pair,
so color classes do not commute.

**One pair attempt** (`_attempt_cmc_pair!`, source `i ∈ 𝒞`), all randomness from
`site_rngs[i]` in this fixed order:

1. partner: one call `k = rand(rng, 1:|𝒞|-1)`, incremented when `k ≥ pos(i)`
   (the position of `i` in the ascending `cset`), `j = cset[k]` — no rejection
   loop of our own (Random's range sampler may redraw internally, which is a
   deterministic function of the stream state);
2. rotation axis (`_random_unit`), then angle `step · randn` — the rotation of
   `_attempt_metro!` without its antipodal flip;
3. the proposal of D2; a null move (`ΔE` NaN) ends the attempt as a rejected
   attempt and draws nothing more;
4. accept when `ℓ ≥ 0`; otherwise draw `u` and accept when `ln u < ℓ` (a
   zero-density partner, `ℓ = −Inf` with finite `ΔE`, draws `u` and is
   rejected).

On accept: `config[i]`, `config[j]`, the `zrows` columns `i` and `j` (copied from
the rows `_pair_delta_energy!` left in the scratch), `dE[i] = ΔE` (the joint
change; each source is visited once per sweep), and `M ← M′`.

**One CMC sweep** (`_cmc_sweep!(st, H, β, sc, rc, scratch) -> (nacc, M)`;
internal — renamed from `cmc_sweep!` in M5, STYLE_GUIDE §1.6):
`M` is recomputed from scratch in `cset` order (nothing is carried across
sweeps, so `ChainState` is unchanged); `dE` is zeroed; the active sites are
visited in site order — a pair attempt for `s ∈ 𝒞` (the running position counter
is `pos(s)`, since `cset` is ascending), one `_attempt_metro!` (flip included) for
an active `w = 0` site; inactive sites are skipped and stay bitwise frozen. Then
`st.energy += _reduce_dE(H, dE)` (the fixed-order sum of the other sweeps),
`acc_metro += nacc`, `att_metro += n_active` — null moves are rejected attempts,
so `_adapt_step!` applies unchanged and `acceptance_metropolis` is the
acceptance over all attempts. The sweep returns the incrementally maintained `M`
(tests compare it with a from-scratch sum). Allocation-free (G16). The
`_CMCScratch` sizes are asserted at every sweep entry (O(1)); the constrained set
and its order live in `_ResolvedConstraint.cset`, so the scratch needs no index
array of its own.

**Checks.** At every sweep start, `|M_⊥| > 1e-10 · Σ_{s∈𝒞}|w_s|` (absolute; a
relative bound misfires on starts near the `|M_init|` door) warns once — the
compensation never reads `M_⊥`, so rounding leaves it as a random walk and a
value above the bound means the state was modified outside the sweep (this one
keeps `maxlog = 1`). Two end-of-thermalization checks are separate functions that
`run_cmc` calls once per temperature, after the final renormalization of the
thermalization phase (M5): `_cmc_check_step_floor(st, kT, n̂)` warns when the
adapted step sits at the clamp `_STEP_MIN = 1e-3` (a high null fraction from large
weight ratios), and `_cmc_check_weak_constraint(rc, config, kT)` warns when
`M_∥ / Σ|w| < 0.05` (above the ordering temperature, or nearly cancelling
weights). Both carry `kT` and `n̂` in the message and as log keys and are NOT
rate-limited: `maxlog` counts per call site and per logger, so a rate limit would
report only the first affected temperature of a ladder, or of a series of runs
(direction scans).

## D4 — the driver and determinism *(single chain landed M5; lanes landed M7)*

**Mover.** `run_cmc(H, con; …)` runs the single-chain temperature loop of
`run_mc` (`_mc_loop!` → `_run_temperature!` → `_compound_sweep!`, `run.jl`)
through a *mover* argument. The mover owns the three things that differ: one
sweep, the `carryover = false` restart, and the result type; plus a hook at the
end of every temperature's thermalization (after the final renormalization;
read-only, `_mover_after_thermalization`).
`_SweepMover` is the existing behavior (Metropolis + `or_per_metropolis`
overrelaxation sweeps; restart = a uniform-random configuration drawn from the
chain stream `st.rng`; `MCResult`; no hook). The methods without a mover argument
construct it and delegate, so `run_mc` and an `"mc"` `resume` keep their calling
paths, arithmetic, and RNG consumption, and the 5-argument
`_compound_sweep!(st, H, β, scs, plan)` that the PT lanes call is untouched.
Gate: the G15 regression pins (`test/unit/test_regression_pins.jl`, a change
detector — `run_mc` with and without overrelaxation, `carryover` true and false
over three temperatures, `run_pt`), captured at `6cc6771` and reproduced bitwise
by the refactored code on the capture machine (2026-11-21). `run_cmc` refuses
`or_per_metropolis` and `sweep_tasks` with an explanatory `ArgumentError`.

`_CMCMover` holds the resolved constraint, the lane's `_CMCScratch` (its size and
the constraint's site count are asserted against `H` at construction), and a
private copy of the constrained start (copied in the constructor, so no chain can
alias it — an aliased start makes the restart a silent no-op; gate D4d). Its sweep is `_cmc_sweep!` on `scs[1]`; the `M` the sweep
returns is dropped (every sweep recomputes `M` from the configuration, so the
chain is a function of the configuration and the RNG streams alone — the
precondition of resume ≡ uninterrupted in M6). Its hook runs the two
end-of-thermalization checks of D3. Its result is a `CMCResult` whose
`constraint` carries the run's direction **bitwise** (an internal constructor
skips the normalization, which is not idempotent: about a third of normalized
vectors move by an ulp when normalized again) and the resolved per-cell-atom
weights, signed zeros folded to `+0.0`. Both movers subtype the internal
`_Mover`.

**Single chain.** Serial only (no `sweep_tasks`, no `or_per_metropolis`). The
chain stream is `Xoshiro(seed)`; it is consumed only to seed the per-site streams
when the `ChainState` is built, and every sweep draws from the per-site streams in
the order of D3. The run is bitwise reproducible for a given seed within the scope
of `pt-threads-determinism.md` P6 (same package and Julia versions, any thread
count). The start is deterministic and RNG-free (D1: the default pattern or the
rotated `init`); with `carryover = false` every temperature after the first
returns to that same start, with the step reset to `step`, and **no RNG is
consumed** — unlike `run_mc`, whose restart draws from `st.rng`. Gates: the D4
testset of `test/unit/test_cmc_driver.jl` (the restored configuration against the
default pattern written from the weights, every stream compared with a copy) and
D4d (through `run_cmc`: the first measured configuration of the second temperature
lies within one sweep of the default pattern). The G15 pins are compared only on
their capture platform class (macOS / aarch64 / Julia 1.12): a trajectory is not
portable — one rounding-level ΔE difference flips an accept decision and changes
the RNG consumption.
Checkpointing landed in M6 (D6): the file stores the constrained start, which a
`carryover = false` resume needs and which for a user `init` is not derivable
from the file's other contents.

**Direction lanes** *(landed M7, `src/cmc.jl`)*. `run_cmc(H,
constraints::AbstractVector{MagnetizationConstraint}; <the single-chain keywords>,
observables = c -> cmc_observables(H, c), ntasks = nothing, seed,
checkpoint = nothing)::Vector{CMCResult}` runs one independent single-chain
`run_cmc` per constraint (a *lane*). The single-chain keywords are listed
explicitly (as in `run_pt`), so a typo is a `MethodError` at the call; each
means the same for every lane (the same `init` is rotated onto each lane's
direction).

- **Seeds.** `master = Xoshiro(UInt64(seed))` (as `run_pt`); the lane seeds are
  `rand(master, UInt64)` drawn in lane order `k = 1, 2, …`, **all before any
  lane runs** (`_cmc_lane_seeds`). Lane `k` is the single-chain driver with its
  lane seed, so `CMCResult.seed` (and the lane's checkpoint) holds the lane seed
  and one lane is reproducible alone: `rs[k]` is bitwise `run_cmc(H,
  constraints[k]; seed = rs[k].seed, observables = observables(constraints[k]),
  …)`.
- **Temperatures.** `resolve_kt(temperature, kT)` runs once and every lane gets
  `kT = kts` — the same Float64 ladder bitwise (a kelvin ladder is converted
  once), and a one-shot iterator is consumed once.
- **Scheduling.** `ntasks` defaults to `min(K, Threads.nthreads())` (as
  `run_pt`) and bounds the number of tasks: the lanes are split into contiguous
  blocks of `cld(K, ntasks)` lanes (`Iterators.partition`), one
  `Threads.@spawn` per block, each running its lanes in order; `ntasks = 1`
  runs them serially on the calling task. Lane `k` writes only slot `k` of the
  result vector, so the results come back in input order and are bitwise
  independent of `ntasks` and of thread scheduling: lanes share no mutable state
  (each builds its own `ChainState`, `_CMCScratch`, mover, binners,
  checkpointer), and `H` is only read. Contiguous blocks (rather than a shared
  work counter) also make a scheduling-dependent seed derivation
  deterministically visible to the gate (the lazy-seed mutant below).
- **Observables are a function of the constraint.** `observables` is
  `constraint -> Vector{Observable}`, called once per lane, in lane order, on
  the calling task, before any lane runs. A `Vector{Observable}` is refused
  with an `ArgumentError` that says why: `:M_par` and `:constraint_residual`
  (and any user observable built for a direction) measure along the constraint
  they were built for, so one vector shared by every lane would silently
  measure every lane along one direction. A non-callable value (a tuple) gets
  its own `ArgumentError`. The door cannot see inside the function: `c -> obs`
  returning one fixed vector bypasses it (documented). Lanes with `ntasks ≥ 2`
  measure and finalize concurrently, so user observables and evaluables must
  not share mutable state across lanes (the defaults capture none). The
  single-chain method keeps `observables::Vector{Observable}` (agreed API; its
  docstring states the same-constraint requirement). **No `extra_observables`
  keyword in v1**: `c -> [cmc_observables(H, c); extra]` covers it with one
  mechanism.
- **Evaluables are one shared vector.** They read only the binned statistics,
  never the constraint, and `Evaluable` holds no mutable state.
- **Duplicate constraints are allowed**: each is an independent replica with
  its own lane seed (a cheap way to replicate a direction).
- **Every door before any lane runs**, the errors unwrapped: an empty vector,
  `ntasks < 1`, `resolve_kt`, one `UpdatePlan` built up front (the plan-level
  doors: sweep counts, step, nbins, seed, …), `or_per_metropolis` /
  `sweep_tasks`, a negative `checkpoint_interval` and the checkpoint path (see
  below), and per lane — whatever the observables function does — the
  constraint's resolution against `H` and its constrained start from `init`
  (the `|M_init|` door depends on each lane's weights), then the observables
  function's result type and the observable / evaluable checks. No lane is
  wasted on an argument error and none fails `K` times over.
- **Failures.** A lane that throws does not stop the others: the failure is
  logged at once (`@error "run_cmc: lane k (n̂ = …) failed; the other lanes go
  on"`, with the exception and backtrace), every lane runs (a finished lane's
  checkpoint file is complete), and then the failures are thrown together as a
  `CompositeException` of `CapturedException`s in **lane order** — the same
  shape for every `ntasks`. No public exception type is added. Recovery: pass
  `checkpoint` for long scans — the completed lanes' files return their stored
  results through `resume(file, H)`.
- **Interrupts.** A `Threads.Atomic{Bool}` stop flag is checked by each block
  before it starts a lane. An `InterruptException` raised in a lane sets it; one
  delivered to the caller while it waits for the tasks (Ctrl-C) sets it too; the
  first interrupt of a call then logs one `@warn` (no further lane starts, the
  running lanes finish first — `_cmc_interrupt!`, after the flag is up),
  and the caller keeps waiting until **every** task has finished
  (`_cmc_wait_lanes`), so no task outlives the call and no lane goes on writing
  its checkpoint file after `run_cmc` has thrown (two writers on one path).
  Lanes already running are **not** stopped mid-run: they finish and write
  their files. `run_cmc` then throws a bare `InterruptException()` (it takes
  precedence over lane failures, which were already logged). The shape is the
  same for `ntasks = 1`, where the interrupt arrives inside the running lane.
- **Checkpoints.** With `checkpoint = path`, lane `k` writes
  `"<stem>.lane<k>.jld2"`, `stem` = `path` without a trailing `".jld2"` in any
  letter case (stripped once), else `path` itself (`_lane_checkpoint_stem`). A
  path that names a directory (empty, ending in `/`, or an existing directory)
  or leaves an empty stem (`".jld2"`) is refused up front. Each lane file is an
  ordinary `"cmc"` file (D6) with the lane's constraint and lane seed,
  resumable alone with `resume(file, H)`. There is no resume of the whole set
  (D7): resume the unfinished lanes one by one. Files of an earlier call with
  the same stem are not removed — a lane that fails before its first write, or
  a re-run with fewer lanes, leaves stale files — so check a file's constraint
  and seed (or use a fresh stem) before reusing it.

Gates (`test/unit/test_cmc_lanes.jl`, G14), on Fmix dims (2,1,1) with six
lanes (four directions with the Fmix weights, a duplicate of lane 2, and a
`:uniform` lane), two temperatures. **G14**: the result vector at `ntasks` 2,
4, and 20 (blocks 3+3, 2+2+2, one lane per task) is bitwise the `ntasks = 1`
vector (every result-shaped field, final configurations, constraints, seeds);
lane `k` is bitwise the single-chain run with lane seed `k`, the seeds derived
**in the test** from the rule above; every lane's `:constraint_residual` is
below 1e-13 and its `:M_par` positive at every temperature, and its final
configuration's weighted magnetization — summed in the test with that lane's
weights — is parallel to that lane's `n̂` (and not to any other lane's); the
duplicate lane differs from its original. **CK**: the file names (with and
without the `.jld2` suffix, `.JLD2`, a doubled suffix; no stray names), kind
`"cmc"`, the stored direction and weights per lane, `resume(file_k, H)` of
every completed lane ≡ the lane's result; a lane interrupted mid-measure
(`_poison_pair` in lane 3 only, file at temperature 2, measure sweep 40)
resumes ≡ its uninterrupted lane, at `ntasks` 1 and 4, while every other lane —
including lane 4, which shares lane 3's task at `ntasks = 4` — completes.
**AGG**: lane 1 failing at its last measurement and lane 6 at its first, at
`ntasks` 1, 4, 20 (at 4 and 20 lane 1 is held until lane 6 has failed, so the
failures complete in reverse lane order): one `CompositeException`, all
entries `CapturedException`s, messages in lane order, one `@error` record per
failure naming the lane. **INT**: an `InterruptException` raised in lane 2 at
`ntasks = 1` and in lane 1 at `ntasks = 4`, and one delivered to the waiting
caller task at `ntasks = 4` (`schedule(caller, InterruptException(); error =
true)` from lane 3): a bare `InterruptException`; the running lanes are held
until the interrupt has fired and then sleep, so a block that went on would
visibly start its next lane — none does (exactly the running lanes' files
exist), and those files are complete at the moment `run_cmc` throws. **Doors**:
as listed above, plus a one-shot `Iterators.Stateful` ladder giving every lane
the full ladder; no checkpoint file appears.

## D5 — torque conventions

`total_torque(H, config) = Σ_{s active} G_s × e_s = −Σ e_s × ∂E/∂e_s` (full
system; `G` from `energy_gradient!`, Zeeman included; never weighted). Identity,
exact at finite N: `⟨T⟩ = −n̂ × ∂F/∂n̂`, i.e. `⟨T⟩·ê_φ = −∂F/∂θ`,
`⟨T⟩·ê_θ = (1/sinθ) ∂F/∂φ`, `⟨T⟩·n̂ = 0`, with
`ê_φ = (−sinφ, cosφ, 0)`, `ê_θ = (cosθ cosφ, cosθ sinφ, −sinθ)`. Observables
report per training cell (`÷ n_cells`, as `:M`). Partial torques of
term-subset Hamiltonians sum to the total (plus the analytic Zeeman part);
an individual partial torque is a free-energy gradient only to O(K).

The analytic Zeeman torque is `T_Z = +MU_B_EV_T (M_μ × B)`, `M_μ = Σ_{s active}
m_{a(s)} e_s` (from `E_Z = −MU_B_EV_T Σ m (e·B)` and `T·w = −dE/dα`: a moment is
turned toward `+B`). `torque_observable(name, Hpart, H)` admits a partial
Hamiltonian only if it matches `H` in `n_cell_atoms`, `dims`, and `n_sites`, carries
no field, and every template equals (atoms, shifts, ls, folded, scaled coef) a
fitted template of `H`; the observable is bound to that `H` object and refuses a
run on any other.

**Observables (landed M5).** `cmc_observables(H, con)` = `:energy`, `:energy2`,
`:torque` (`total_torque / n_cells`, Cartesian), `:M_par` (`M·n̂ / n_cells`, weight
units per cell), `:constraint_residual` (`|M_⊥|/M_∥`, relative, radians), and with
a field `:torque_zeeman` (`MU_B_EV_T (M_μ × B) / n_cells`); `cmc_evaluables()` =
the specific heat only. Definitions, units, and why χ and U are left out:
`binning-observables.md` B5 (the one place the observable conventions live). Like
`torque_observable`, the measuring functions are pure and bound to their `H`.
`CMCResult.points[k].acceptance_metropolis` is the accepted fraction of all
attempts of the measurement phase (pair attempts with null moves counted as
rejections, and the hybrid single-spin attempts, pooled); `acceptance_or` is
`NaN`.

## D6 — checkpoint *(landed M6, `src/checkpoint.jl`)*

Kind `"cmc"`, additive under schema v3 (the `gpu_pt` precedent; layout and the
older-version failure mode: `checkpoint-schema.md` C6). `run_cmc(…;
checkpoint, checkpoint_interval)` writes with the cadence of `run_mc`
(periodic every `checkpoint_interval` sweeps, the clock reset by every write,
plus the unconditional end-of-temperature boundary write of the shared
`_mc_loop!`); writing consumes no RNG.

- **What is stored beyond `"mc"`.** `_Checkpointer` carries the constraint
  record (`_constraint_record(rc)`: the direction bitwise, the resolved
  per-cell-atom weights with signed zeros folded — the same record as
  `CMCResult.constraint`) and the constrained start. `_write_header` — and
  nothing else — writes `constraint/{direction, weights, partner_rule =
  "uniform"}` and `start/config` on every write; `plan/or_per_metropolis = 0`
  and `plan/sweep_tasks = 1` are the plan's fixed values for a constrained run
  (asserted when the checkpointer is built, refused on read otherwise).
- **Why the start is stored.** The `carryover = false` restart is
  deterministic and RNG-free (D4): it returns to the start rather than drawing
  from a restored stream (`checkpoint-schema.md` C4, per-kind statement). For a user
  `init` the start is the rotated `init`, derivable from nothing else in the
  file. It is stored for every `"cmc"` file (the default pattern too), so the
  reader has one path.
- **The fingerprint is unchanged**: the constraint is run configuration, not
  model identity, and `model_fingerprint` is pinned by dependent packages'
  checkpoint formats.
- **`resume(path, H; observables = nothing, evaluables = nothing, …)`.** The
  kind and, for `"cmc"`, the constraint (refusing a `partner_rule` other than
  `"uniform"`) are read before the defaults are resolved — `cmc_observables(H,
  constraint)` / `cmc_evaluables()` for `"cmc"`, the standard lists otherwise —
  and the existing checks then run unchanged. The mover is rebuilt from the
  stored constraint and start; the chain is restored verbatim (the mover holds
  no chain state). The completed-file early return builds a `CMCResult` for
  `"cmc"` (an `MCResult` for `"mc"`); the return type gains `CMCResult`.
  Explicitly passed lists behave as before: the standard observables on a
  `"cmc"` file are a name mismatch. A `"gpu_pt"` or unknown kind is refused by
  name right after the kind is read, before any default is resolved.
- **Consistency door (read side, non-projecting).** With the stored `n̂` and
  weights, `M = Σ_{s∈𝒞} w_s e_s` of both `start/config` and `chain/config` must
  have `M·n̂ > 0` and `|M⊥| ≤ 1e-8 Σ|w|`, or the file is refused as corrupted
  (also on the completed-file path). A rewritten direction or start that is
  still a set of unit vectors passes every other door and would resume into
  another ensemble with at most a drift warning.

Gates (`test/unit/test_cmc_checkpoint.jl`): G13 resume ≡ uninterrupted,
bitwise — results, final configuration, the result's constraint, and the chain
block of the final checkpoint file (energy, step, counters, chain and per-site
RNG words) — each interrupting the writer mid-run (`_poison_pair`, or a
poisoned evaluable for the default-argument case) with the file position
asserted: Fmix dims (2,1,1) over three temperatures with `measure_interval = 3`
and `renorm_interval = 70`, `carryover = true` (file at temperature 2, measure
sweep 160) and `carryover = false` with a user `init` (same position; the
resumed tail crosses the restart of temperature 3), the boundary-only cadence
with `carryover = false` (file at the temperature-2/3 boundary), FZ with
weights (1, −2) and a field (`measure_interval = 2`, `renorm_interval = 30`;
file at temperature 2, measure sweep 130), and `resume(path, H)` with default
arguments (file at temperature 1, measure sweep 290) and on completed files.
A mid-thermalization gate (`carryover = false` + `init`, file at
temperature 2, therm sweep 70 — inside an adaptation window) and a double
resume (that resumed run interrupted again, file at temperature 2, measure
sweep 110, written by the resumed writer; the second resume crosses the
temperature-3 restart). The final-file comparison includes `start/config` and
`constraint/{direction, weights}` as rewritten by the resumed writer. The
default-argument gate also runs on a constraint whose stored direction is not
idempotent under normalization.
RT: the stored constraint round-trips in `==`, `isequal`, `hash`, a `−0.0`
moment stored folded, the direction bitwise over 12 random directions (at
least one of which moves under a second normalization). Doors: fingerprint,
the GPU method, `partner_rule`, explicit standard observables / evaluables,
corrupted direction / weights / plan / start, a relabelled kind (`"foo"`,
`"gpu_pt"`), and the consistency door (direction rewritten to another unit
vector, start or chain rewritten to all `+x̂`; on a completed and a mid-run
file). Kinds: an `"mc"` file resumed
with default arguments still resolves the standard lists (equal to the
uninterrupted default run).

## D7 — out of scope for v1

Constrained PT, block constraints, per-site weights, |M| constraints,
overrelaxation in CMC, colour-parallel CMC sweeps, GPU, constrained
`minimize_energy`, post-processing fits (K(T), κ_lm), a resume of a whole
lane set (one file per lane, each resumed alone; D4), an `extra_observables`
keyword (D4).

## Mutation record *(filled per milestone)*

| Gate | Mutation | Observed shift | σ | Result |
|---|---|---|---|---|
| G2 | naive ΔE (`c_j` from the unmodified rows) | per sharing case, max \|naive − exact\| > 1e-5 (asserted lower bound); smallest measured 0.059 (F4 end pair (1,4)) | — (deterministic) | killed: `ΔE ≈ exact` and `worst < 1e-13` on all sharing cases; invisible on non-sharing pairs, as designed |
| G2 | column `i` not restored | — | — | killed: bitwise `zrows` restore on every case |
| G2 | both columns written tentatively before `ΔE_i` | O(ΔE) | — | killed: `ΔE ≈ exact` on every sharing case |
| G3 | torque sign (`e × G`) | O(T) | — | killed: rotated-energy difference, Zeeman share, G17 sum |
| G3 | wrong axis (cyclic components) | O(T) | — | killed: same assertions |
| G3 | Zeeman double-counted in `total_torque` | O(T_Z) ≈ 1e-3 eV | — | killed: rotated-energy difference on FZ, Zeeman share |
| G3/G17 | analytic Zeeman torque zero / sign flipped | O(T_Z) | — | killed: hand-written `μ_B Σ m e × B`, G17 sum on FZ |
| G17 | partial torque on a site-shifted configuration | O(T_part) | — | killed: G17 sum and per-part rotated-energy differences |
| G17 | door without the template match | — | — | killed: relabelled-atom, perturbed-coefficient, and foreign-term `Hpart` rejections |
| G17 | partial torque not divided by `n_cells` | factor 2 (dims = (2,1,1)) | — | killed: G17 sum and per-part differences |
| G2 | partner row tabulated inside the tentative-write window | — | — | killed: `zrows` bitwise intact after a harmonics `DomainError` (only that assertion; values unchanged) |
| G2 | an allocation inside the window | — | — | killed: `@allocated == 0` on F3a, F4, FZ |
| G3 | torque sign / wrong axis (closed forms) | O(T) | — | also killed by the uniaxial `T·ŷ = −N K sin 2θ` and in-plane `T·ê_θ = N K sin 2φ` checks |
| G17 | template key without `shifts` | — | — | killed: shifts-only `Hpart` rejection |
| G17 | no field door | — | — | killed: field-on-`Hpart` rejection with the field-door message |
| G17 | observable not bound to its `H` | — | — | killed: measuring on an equal but distinct `H` throws |
| G1 | compensation ratio `\|w_i/w_j\|` | O(1) partner error on mixed-sign pairs | — (deterministic) | killed: G1 `\|M⊥\| ≤ 1e-12 Σ\|w\|` on Fmix and F2w− (F2w+ is blind, as designed); G4 tallies `ref`, `null_s2`, `finite` |
| G1 | lab-frame compensation (`ẑ` in place of `n̂`) | O(1) | — | killed: G1 `\|M⊥\|` on all three fixtures; G4 tallies `ref`, `sign`, `rev_*`, null classification; the aimed-at-`x̂`/`ẑ` targets |
| G1/G4 | `j == i` admitted | — | — | killed: `AssertionError` for `i = j` on null-geometry inputs (the non-null case is caught by `_pair_delta_energy!` regardless) |
| G1/G4 | zero-weight partner assert removed | — | — | killed: the Fmix zero-weight partner must raise `AssertionError` |
| G4 | Jacobian exponent 2 → 1 | `ln(M_∥′/M_∥)` per move (> 1e-3 on ≥ 500 triples) | — | killed: tally `ℓ` (independent formula) on all five fixtures; reversibility passes, as expected |
| G4 | `\|z\|` ratio missing / inverted | `ln\|e_j∥/e_j∥′\|` per move | — | killed: tally `ℓ` on all five fixtures, and the `e_j∥ = 0 ⇒ ℓ = −Inf` case |
| G4 | `σ_j` forced to +1 | O(1) partner error when `e_j∥ < 0` | — | killed: tallies `ref`, `sign`, `rev_e`, `rev_ℓ`, `rev_ΔE`, `finite`, `null_M` |
| G4 | unweighted `M_∥′` update in the proposal | O(1) on weighted fixtures | — | killed: tallies `ℓ`, `null_M`, `rev_*` on F2w± and Fmix; invisible on uniform weights, as designed |
| G4 | `M_∥′ > 0` null test removed | — | — | killed: `DomainError` (`log` of a negative ratio) and the null-move counts |
| G4 | NaN-unsafe `s2 ≥ 1` alone / `M_∥′ ≤ 0` alone | none | — | **survives — equivalent mutants**: a NaN that reaches one test reaches the other (`s2` NaN ⇒ `e_j′` NaN ⇒ `M_∥′` NaN; `M_∥′` NaN only via a NaN `s2`), so each guard alone already makes every NaN input a null move. Both kept as defense in depth |
| G4 | NaN-unsafe, both tests | — | — | killed: a NaN proposal must return `(e_j, NaN, −Inf)` |
| G4 | sign guard `(e_j′·n̂ ≥ 0) == (σ_j > 0)` removed | none | — | **survives — guard on an unreachable region**: with `s2 < 1`, `\|e_j∥′\| ≥ ~1e-8` dwarfs any rounding, so no input flips the sign |
| G4 | `e_j′` not re-normalized | `e_j′·ẑ` one ulp above 1 | — | killed: the proposal aimed at `e_j′ = ẑ` throws `DomainError` in the harmonics; tally `domain` on the aimed-at-`x̂` proposals |
| G4 | the clamp in `_unit_clamped` removed (division only) | none | — | **survives — equivalent**: with correctly rounded arithmetic the division by the computed norm already keeps every component in `[-1, 1]`; the clamp is a guard |
| G0 | `sign(w)` dropped from the default pattern | `M̂` unchanged (Σw n̂ still ∥ n̂) | — | killed: the per-site default pattern and `M_∥ = Σ\|w\|` on Fmix |
| G0 | unweighted `M` in `_cmc_magnetization` | O(1) misrotation | — | killed: tallies `Mhat`, `axis_fixed`, `plane`; the weighted `\|M_init\|` door |
| G0 | inactive site admitted to `𝒞` | — | — | killed: Fmix `cset`/`site_w`, the `\|𝒞\| < 2` door with the only partner inactive, the antiparallel starts, tally `Mhat` |
| G0 | `\|M_init\|` door removed | — | — | killed: `[v, −v]`, the 1e-9-tilted pair, and the F3 (2, 1, 1) case at `\|M\| = 3.5e-8` are refused |
| G0 | door scaled by `\|𝒞\|` instead of `Σ\|w\|` | threshold 3e-8 instead of 4e-8 | — | killed: F3 weights (2, 1, 1) with `\|M\| = 3.5e-8` must throw |
| G0 | door scaled by `Σ w²` | threshold 6e-8 instead of 4e-8 | — | killed: the same fixture with `\|M\| = 4.5e-8` must pass |
| G0 | half turn for every obtuse start (non-minimal) | O(1) | — | killed: tallies `axis_fixed`, `plane` (the axis `M̂₀ × n̂` is fixed) |
| G0 | near-antiparallel branch removed | `M̂ = −n̂` exactly is left unrotated | — | killed: the exact antiparallel starts (`n̂ = ẑ`, `n̂ = x̂`); tilts ≥ 1e-12 rad are handled by the plane formula alone |
| G0 | signed-zero fold of the direction / of the weights removed (inner constructor; file-based mutation, restored and `cmp`-checked) | — | — | killed: `MagnetizationConstraint(0.0, 2.0; weights = [-0.0, 1])` vs `([0, 0, 1]; weights = [0.0, 1])` must agree in `==`, `isequal`, `hash` |
| G0 | resolve-time finiteness re-check removed | — | — | killed: a weight vector mutated to NaN after construction must be refused |
| G5 | null move re-drawn instead of counted (M4) | G5: +0.022 on N=2 `⟨M_∥⟩`, +0.17 on weighted `⟨M_∥²⟩`; seen also by G6 (+5.5e-4, F2u θ=0.3) and G7 (−0.035 on `T·ê_φ/K`) | 1.3e-3 / 8.9e-3; 7.2e-5; 3.2e-3 | killed by G5, which owns it: 15.8–21.8σ across seeds (3.8× atol at the pinned seed). G6 sees it at 7.6σ with seed 5601 but only 4.3σ with another seed — not a reliable owner; G7 11σ; D3 (1227 offenders) |
| G5/G6/G7 | Jacobian exponent 2 → 1 (in the sweep) | G5 −0.33 (N=2 `⟨M_∥⟩`); G6 −8.3e-3; G7 +0.145 | 1.5e-3; 8.6e-5; 3.4e-3 | killed: 220σ / 97σ / 42σ |
| G5/G6/G7 | `\|z\|` factor missing | G5 +0.17 (N=2); G6 +4.5e-3; G7 −0.29 | 1.3e-3; 5.9e-5; 3.0e-3 | killed: 126σ / 77σ / 96σ |
| G5 | exponent 1 **and** `\|z\|` missing | N=2 equal weights: none (as designed); N=3 `P(e·n̂ < 0)` +0.061 | 8.8e-4 | killed: N=3 70σ, weighted N=2 37σ; G6 (F2w−) 18σ, G7 49σ |
| G5/G6/G7 | `σ_j` forced to +1 | G5 −0.076 (N=3 `P`); G6 +4.6e-3 (F2w−); G7 −0.050 | 3.2e-4; 5.0e-4; 3.4e-3 | killed: 240σ / 9.3σ / 15σ |
| D3 | partner by a rejection loop over `𝒞` (variable RNG consumption) | statistics unchanged (G5–G7 all < 3σ) | — | **statistically invisible, as expected** (the partner law is still uniform on `𝒞 \ {i}`); killed by the D3 RNG-contract replay (843 offenders) |
| D3 | partner drawn from all of `𝒞`, `j = i` a no-op | statistics unchanged (< 2σ) — a mixture with the identity kernel keeps `π` | — | **statistically invisible, as expected**; killed by D3 (948 offenders) |
| G1/D3 | partner drawn from all of `𝒞`, `j = i` passed to the proposal | — | — | killed: `AssertionError` in `_cmc_pair_proposal` (every sweep gate errors) |
| D3 | accept uniform drawn also when `ℓ ≥ 0` | statistics unchanged (< 2.2σ) | — | **statistically invisible, as expected** (an RNG-contract change only); killed by D3 (352 offenders) |
| G1 | `M` not updated after an accept | stale `M_∥` within the sweep | — | killed: `ArgumentError` (the stale `M_∥` turns non-positive), G1b `\|M_inc − M\| = 2.2` |
| G1/G6 | stale `zrows` column `j` after an accept | G6 +8.9e-4 (F2u θ=0.3) | 7.4e-5 | killed: G1 `rows_ok`, `worst_E` on all three fixtures; G6 12σ |
| G1b | hybrid `w = 0` sites skipped | site 7 never moves | — | killed: G1b (0 of 2000 sweeps); the measure gates are blind (no `w = 0` site there) |
| G5h | hybrid `w = 0` site attempted at `0.5β` (call-site slip in `_cmc_sweep!`) | −0.146 on `⟨e₁·e₃⟩` (exact `L(1.2) = 0.36620`) | 1.6e-3 | killed: G5h −92σ (18× atol); D3 sweep replay (48 offending sweeps). Survived all 85 tests of the first M4 version — G5h was added for it |
| G1/D3 | source position `pos = s` instead of the running count | wrong partner exclusion once `𝒞` is not a prefix of the sites | — | killed: G1 Fmix on dims (2,1,1) (`j = i`, AssertionError) and the D3 sweep replay; invisible on Fmix (1,1,1), where `cset = 1:6` |
| D3 | hybrid site drawing from the chain stream `st.rng` | statistics unchanged | — | **statistically invisible, as expected**; killed by the D3 sweep replay (300 offending sweeps, `st.rng` consumed) |
| G1/G6 | naive pair ΔE (`c_j` from the unmodified rows) | G6 −6.7e-3 (F2u θ=0.6 `T·ê_φ`) | 1.1e-4 | killed: G1 `worst_E` (incremental vs `total_energy`) on all three fixtures; G6 62σ |
| G0 | drift threshold relative (`1e-10·M_∥`) instead of absolute | — | — | killed: the no-warning case `M_∥ = 2e-3`, `\|M_⊥\| = 1e-10` (relative 5e-8) |
| G6d/D4-driver/G0d | `run_cmc` sweeping with the unconstrained `_SweepMover` (M5) | G6d (through `run_cmc`) T_x +9.83e-3, T_y −2.70e-2 — the free dimer's mean torque vanishes | 1.18e-4 / 1.15e-4 (the mutant run's own σ, which the gate divides by; the correct run has 6.0e-5 / 1.21e-4) | killed: G6d 83σ / 235σ in the gate's σ (164σ / 222σ in the correct run's σ); the ladder testset (residual, `M_∥ > 0`; 3 failures); G0d (4 failures — the unconstrained chains end with `M_∥ < 0`) |
| D4 | CMC restart draws `rand(st.rng)` (M5) | none in any result — the chain stream is never read again after the site streams are seeded | — | killed: D4 `st.rng` against its copy (the only gate that can see it, by construction) |
| D4 | CMC restart keeps the adapted step (M5) | — | — | killed: D4 `st.step == step` |
| OBS/G6d | `n_sites` in place of `n_cells` in `cmc_observables` (M5) | factor `n_cell_atoms` | — | killed: 13 OBS assertions on Fmix dims (2,1,1) (`n_cells = 2`, `n_sites = 16`); G6d T_x +4.98e-3 (166σ) |
| G0d | end-of-thermalization hook empty (M5) | — | — | killed: 5 failures + 1 error (warning counts, kT / n̂ keys) |
| G0d | weak-constraint check not called / step-floor check not called (M5) | — | — | killed: 3 / 2 failures |
| G0d | `maxlog = 1` put back on the weak-constraint warning (M5) | only the first temperature of a ladder, and the first run, report | — | killed: 2 failures (two records per two-temperature ladder; a second run reports again) |
| G15 | `_SweepMover` restart drawn from a COPY of `st.rng` (stream not advanced) (M5) | the second and third restarts repeat the first | — | killed: G15 `mc_or` (14 assertions). **Survived the first version of the pin** (two temperatures — one restart — make it an equivalent mutant); `mc_or` was extended to three temperatures and recaptured from a `git archive` of `6cc6771` |
| G15 | overrelaxation sweeps before the Metropolis sweep in the 5-argument `_compound_sweep!` (M5) | O(1) | — | killed: G15 `mc_or` 40 and `pt` 40 assertions (`mc_plain`, without overrelaxation, is blind, as designed) |
| D4d | the chain aliasing the stored start (both copies removed — `_CMCMover` and `run_cmc`), so the `carryover = false` restart is a no-op (M5 review) | first measured configuration of the 2nd temperature 7.5e-3 from the default pattern instead of 1.0e-4 (the 2nd-temperature `⟨E⟩` of the reviewer's case −0.8594 → −0.7286) | — | killed: D4d (bound 1e-3; constant energy, step 1e-4, one sweep). Survived every test of the first M5 version. Removing only the `run_cmc` copy is now an equivalent mutant: the mover copies its start in its constructor |
| G0d | end-of-thermalization hook moved after the measurement phase (M5 review) | the hook sees the measured end state (nearly cancelling `M`) | — | killed: the `sweeps_therm = 0` no-warning case (the start has `M_∥/Σ\|w\| = 1`) |
| OBS | `:torque_zeeman` from the constraint weights instead of the moments (M5 review) | invisible under `:auto` (weights = moments) | — | killed: 2 assertions with explicit weights (1, −3) and `:uniform` on FZ |
| result | signed-zero fold of the resolved weights removed in `_mover_result` (M5 review) | a `−0.0` moment reported as `−0.0` | — | killed: 3 assertions (`isequal`, `hash`, `signbit` against a constraint built from `[2, 0, 1]`) |
| doors | `or_per_metropolis` accepted silently (M5 review) | — | — | killed: 2 assertions (the explained `ArgumentError` for any value, 0 included) |
| result | `CMCResult.constraint` built with the public, normalizing constructor (M5) | the direction moves by an ulp for about a third of directions | — | killed: 4 assertions over 12 random directions. **Survived the first version** (only n̂(0.6, 0.35), where the normalization happens to be idempotent); the random-direction loop asserts that at least one direction moves |
| G13 | reader ignores the stored direction (`ẑ`) (M6) | O(1) — another constraint | — | killed: every G13 gate (251 failures) and RT |
| RT | reader normalizes the stored direction again (public constructor) (M6) | an ulp on about a third of directions | — | killed: RT (12 failures — bitwise direction, `isequal`, `hash`, the resumed result's direction). Every G13 gate passes: n̂(0.6, 0.35) is idempotent under normalization, as the M5 record found |
| G13 | weights not stored (the reader resolves `:auto`) (M6) | O(1) on Fmix and FZ (1, −2) | — | killed: every G13 gate (197 failures + 1 error: the boundary resume starts from a state that violates the `:auto` constraint) and RT |
| RT | weights stored unfolded (`−0.0` written) (M6) | `isequal` / `hash` only — the resumed result folds again | — | killed: RT (4 failures: `isequal`, `hash`, the stored `signbit`) |
| G13 | start not stored — the reader rebuilds the default pattern (M6) | the restart of temperature 3 goes elsewhere | — | killed: the `carryover = false` + `init` gate (22) and the boundary-only gate (20), plus the start-size door (never read); the `carryover = true` gates are blind, as designed |
| G13 | the writer stores the default pattern instead of the rotated `init` (M6) | same | — | killed: the stored-start assertion and the same two gates (43) |
| G13 | defaults resolved before reading the kind (standard observables) (M6) | — | — | killed: the default-argument gate and RT `resume(p, H)` (name-mismatch errors), and two doors that now hit the name check first |
| G13 | default evaluables resolved before reading the kind (M6) | — | — | killed: every G13 gate errors at the `_check_evaluables` door (`:susceptibility` needs `:m2`) |
| G13 | completed-file early return always builds an `MCResult` (M6) | — | — | killed: the completed-file gate (`isa CMCResult`, `.constraint`) and RT |
| G13 | `_write_header` writes the constraint group on a checkpointer's first write only (M6) | — | — | killed: every mid-run gate (the later file has no `constraint/`) and 6 doors |
| G13 | `"cmc"` resumed through the plain `"mc"` path (`_SweepMover`) (M6) | — | — | killed: every G13 gate and RT (the early return builds an `MCResult`; the unconstrained mid-run resume errors) |
| G13 | therm-phase resume restarts the sweep count at 0 (M6 review) | the adaptation window and the phase length shift | — | killed: the mid-thermalization gate (80 failures). Survived the first M6 version (no file stopped in thermalization) |
| G13 | the resumed writer stores the chain configuration as `start/config` (M6 review) | the next resume's restart goes elsewhere | — | killed: the `start/config` comparison of the final file (every mid-run gate) and the double resume (25 failures). Survived the first M6 version (the resumed writer's groups were never compared) |
| doors | consistency door removed / chain side only removed / start side only removed (M6 review) | a rewritten direction or start resumes into another ensemble | — | killed: 4 / 1 / 2 door assertions |
| G13 | default observables rebuilt through the normalizing public constructor (M6 review) | an ulp in `n̂` for `:M_par` / `:constraint_residual` | — | killed: the default-argument gate on the non-idempotent constraint (5 failures). Survived the first M6 version (n̂(0.6, 0.35) is idempotent); the first fix also survived — its helper rebuilt the constraint from the direction, normalizing it again into an idempotent one — so the gate now asserts that the run's own stored direction moves under normalization |
| doors | `"gpu_pt"` / unknown kind refused only after the defaults are resolved (M6 review) | a relabelled `"cmc"` file fails as an observable-name mismatch | — | killed: the two relabelled-kind doors |
| G13 | `chain/frozen` not restored (always `false`) (M6 review) | none | — | **survives — equivalent mutant**: a thermalization-phase entry sets `frozen = false` itself, and nothing in the single-chain loop reads it during the measurement phase (also passes `test_checkpoint.jl`). Kept in the file for the schema's completeness |
| G14 | lane seeds drawn lazily inside each lane task from the shared master stream (order follows the scheduling) (M7) | lanes run on other seeds | — | killed: the ntasks gate, lane k ≡ single chain, the checkpoint and interrupted-lane gates (1056 failures). With contiguous blocks the draw order can never be the lane order at `ntasks ≥ 2`, so the kill does not depend on a lucky schedule |
| G14 | one observable vector (`observables(constraints[1])`) shared by every lane (M7) | `:constraint_residual` O(1) on every other lane | — | killed: the own-constraint gate (residual, `M_par`), lane k ≡ single chain, the interrupted-lane gate (73 failures). The ntasks gate is blind, as designed (the mutant is deterministic) |
| G14 | results collected in completion order (`push!` under a lock) (M7) | lanes permuted at `ntasks ≥ 2` | — | killed: the ntasks gate, lane k ≡ single chain, the checkpoint gates (921 failures) |
| G14 | lane seeds from a master stream advanced by one draw (M7) | every lane on the next seed | — | killed: lane k ≡ single chain, the interrupted-lane gate's `seed == seeds[3]` (199). The ntasks gate is blind, as designed |
| CK | lane file index off by one (`k - 1`) (M7) | `scan.lane0.jld2 … scan.lane5.jld2` | — | killed: the file-name gate and every per-lane resume (351 failures, 9 errors) |
| CK | stem rule ignoring `.jld2` (`stem = path`) (M7) | `scan.jld2.lane1.jld2` | — | killed: the file-name gates, and every resume by the documented name errors |
| CK | stem rule case-sensitive (M7 review) | `up.JLD2.lane1.jld2` | — | killed: the suffix-case file-name gate and the `.JLD2` empty-stem door (3) |
| doors | directory door removed (M7 review) | an existing directory `d` writes `d.lane<k>.jld2` beside it | — | killed: the directory-path door (1; the `/`-terminated and `.jld2` paths still meet the empty-stem door) |
| CK/AGG | a failing lane aborts its block (no per-lane capture) (M7) | lane 4 (same task as lane 3 at `ntasks = 4`) never runs | — | killed: the interrupted-lane gate (missing lane-4 file, a bare `ErrorException` at `ntasks = 1`) and AGG (13) |
| doors | vector `observables` accepted (door removed, the vector used for every lane) (M7) | — | — | killed: the vector-observables door (3) |
| AGG | failures collected in completion order (M7 review) | `["fail lane 6", "fail lane 1"]` at `ntasks` 4 and 20 | — | killed: AGG (2). **Survived the first M7 version** (no test with two failing lanes); the first AGG version also let it survive — six tasks on four threads started lane 6 only after lane 1 had ended, so lane 1 is now held until lane 6 has failed |
| AGG | failures reversed (M7 review) | — | — | killed: AGG (3; every `ntasks`). Survived the first M7 version |
| AGG | failures stored bare (no `CapturedException`) (M7 review) | — | — | killed: the `all(isa CapturedException)` assertions of AGG and the interrupted-lane gate (5). Survived the first M7 version |
| AGG | no `@error` record per failure (M7 review) | — | — | killed: the log assertions of AGG and the interrupted-lane gate (5) |
| INT | stop flag not checked by the blocks (M7 review) | lanes 2, 4, 6 start after the interrupt | — | killed: INT (b) and (c) file lists (3) |
| INT | the caller rethrows a delivered interrupt at once (stops waiting) (M7 review) | the running lanes outlive the call | — | killed: INT (c) (1 failure, 3 errors — the lane files are incomplete or still being written when `run_cmc` throws) |
| INT | a lane's `InterruptException` treated as a failure (M7 review) | a `CompositeException`, the blocks go on | — | killed: INT (a), (b) (6) |
| INT | the bare `InterruptException` not thrown after the lanes (M7 review) | the call returns or throws the failures instead | — | killed: INT (a)–(c) (3). The removed rethrow of the first M7 version survived every test then |
| doors | `temperature` / `kT` passed through to every lane instead of resolved once (M7 review) | a one-shot iterator is consumed by lane 1; lanes 2, 3 get an empty ladder | — | killed: the `Iterators.Stateful` door (1 error) |
| doors | the per-lane resolution and `init` check skipped (left to the observables function / the lanes) (M7 review) | the errors arrive from the lanes as a `CompositeException` | — | killed: the custom-observables resolution door and the one-lane `init` door (4) |
| doors | the up-front `UpdatePlan` removed (M7 review) | plan errors arrive from the lanes, `K` times | — | killed: the `sweeps_measure = 0`, `nbins = 1`, and negative-seed doors (3) |

All other M3 mutants were applied in memory (function definitions re-evaluated
into the loaded module, source untouched, `shasum` identical before and after).
The M4 mutants were applied the same way, one Julia process per mutant, each
running the whole `test/unit/test_cmc_sweep.jl`; shifts are the worst row of each
statistical gate, σ the binning error of that row, atol = 5σ. The M5 mutants were
applied the same way (function definitions re-evaluated, sources untouched and
`shasum`-identical), one Julia process per mutant, each running
`test/unit/test_cmc_driver.jl` and `test/unit/test_regression_pins.jl` (the
review-round mutants: `test_cmc_driver.jl` only). The M6 mutants were applied
the same way at the source-text level: the owning file (`src/checkpoint.jl`
or `src/cmc.jl`) was read, one site replaced, and the text re-evaluated into the loaded module
(`include_string`), one Julia process per mutant running
`test/unit/test_cmc_checkpoint.jl` (the `frozen` mutant also
`test_checkpoint.jl`); an unmodified re-evaluation passes every assertion, and
`shasum` of `src/` was identical before and after. The M7 mutants (with the
review round's) were applied the same way — the lane section of `src/cmc.jl`
re-evaluated from the edited source text — one Julia process per mutant with
`-t 4`, each running `test/unit/test_cmc_lanes.jl`; an unmodified
re-evaluation passes all 3526 assertions, and `shasum` of `src/` was identical
before and after.

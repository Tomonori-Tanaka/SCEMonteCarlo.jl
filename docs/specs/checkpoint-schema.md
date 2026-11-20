# Decision record — checkpoint schema (v3) and bit-identical resume

Status: landed (M6). Owner: `src/checkpoint.jl`; gates in
`test/unit/test_checkpoint.jl`.

## C1 — format: JLD2, plain data only

A checkpoint is a JLD2 file of **plain data** — `Bool`/`Int`/`Float64`/`UInt64`/
`String` scalars and arrays in named groups. No Julia struct is ever serialized
for reconstruction, so a package refactor cannot silently break the format (the
SpinClusterMC hand-rolled positional-serialization failure mode). Writes are
atomic: PID-suffixed temp file + `mv` (one writer per checkpoint path assumed —
concurrent runs must use distinct paths). Checkpoint writing consumes no RNG
(gated: a checkpointed run equals an uncheckpointed one bit-for-bit).

Rejected: `Serialization` stdlib (positional, Julia-version-fragile), TOML/JSON
(no bit-exact `Float64` round-trip without hex-float contortions, huge configs).

## C2 — schema v2

v2 (2026-07-15, colored sweeps): adds `plan/sweep_tasks` and the per-site RNG
streams `chain/site_rngs` (a `words × n_sites` UInt64 matrix — one Xoshiro per
site). v1 files are rejected by the version check (pre-release breaking change).

## C3 — schema v3: the fingerprint mixer

v3 (2026-08-22): `model_fingerprint` changes for every model. The v2 mixer was
plain FNV-1a, `h = (h ⊻ x) · p` per 64-bit word. An odd multiplier carries bit
63 of its input only into bit 63 of the product, so the hash was *linear* in
the top bit of every word: a `Float64` sign flip toggled one parity bit, and an
even number of sign flips anywhere in the payload — a reversed field, two
negated `folded` entries, `B → −B` on an odd number of moment-carrying atoms —
cancelled exactly. A resume door blind to a reversed field is not an identity.
v3 folds the top half into the bottom half after every multiply
(`h ⊻= h >> 32`), which moves the sign bit to where the next multiply spreads
it, and finishes with the MurmurHash3 `fmix64` avalanche. The Zeeman work first
papered over the field case by mixing the field words twice (once bit-rotated);
v3 removes that special case. Gates: the mixer property test in
`test_checkpoint.jl` (every single-bit flip and every pair of top-bit flips
changes the output; the v2 mixer is shown to collide), the `folded` two-flip
test in `test_zeeman.jl` (was `@test_broken`), and the portable pins.
v2 files are rejected by the version check (pre-release breaking change);
dependent packages that store `model_fingerprint` in their own files
(SCESpinDynamics) inherit the change.

```
schema_version    Int     == 3, hard-checked on load
kind              String  "mc" | "pt" | "gpu_pt" | "cmc"
julia_version, package_version   String (informational)
model_fingerprint UInt64  FNV-1a with a per-word 32-bit fold and a murmur
                          finalizer (C3) over (n_cell_atoms, dims, every
                          term's coef/atoms/shifts/ls/folded, and — only when
                          the Hamiltonian carries `magmoms` — a tag, the
                          moments, and the field; zeeman-field.md Z4) — NOT
                          Base.hash
                          (which is Julia-version-dependent); mismatch on
                          resume ⇒ error. Stable across Julia versions, NOT
                          across machines for SALC-basis models: the `folded`
                          tensors come out of LAPACK and differ in their last
                          bits between BLAS builds (seen macOS vs ubuntu CI,
                          2026-08-22), so a checkpoint resumes on the machine
                          family that wrote it — hand-built terms are portable
zeeman/{magmoms, field}   present iff the Hamiltonian carries `magmoms`
                          (additive, 2026-08-22): Vector{Float64} μ_B per cell
                          atom and the 3-vector in tesla — informational, the
                          reader verifies through the fingerprint only
checkpoint_interval, exchange_interval   Int
plan/*            every UpdatePlan field (kts, sweeps, intervals, step0,
                  adapt_*, renorm_interval, nbins, carryover, sweep_tasks, seed)
plan/observable_names, plan/observable_ncomps   resume-compatibility check
-- kind == "mc":
progress/{temp_index, phase ("therm"|"measure"), sweep}
npoints; points/<i>/{kT, acceptance_*, final_step, max_drift, stat_names,
                     stats/<name>/{mean, err, tau_int, count}}
chain/{config (3×n), energy, rng (UInt64 words), site_rngs (words × n_sites),
       step, frozen, counters, max_drift}
has_accs; accs/<obs>/{binner/{count, sums, sums2, pending, pending_full, n},
                      store/{bin_size, means, nfull, acc, nacc}}
-- kind == "pt":
progress/{phase, done, parity}
exchange_rng; swap_att; swap_acc; nlanes
lane/<r>/{chain fields...}; lane/<r>/accs/<obs>/... (measure phase only)
-- kind == "gpu_pt"  (additive, 2026-08-14 — G8b; older packages reject it as
                      an unknown kind, loudly):
workgroupsize     Int     part of the determinism scope — resume USES the
                          stored value (it is not a resume kwarg)
progress/{phase, done, parity}
exchange_rng; swap_att; swap_acc; nlanes
lane/<r>/{config (3×n), energy, dev_seed (UInt64), sweep_index, step,
          counters [acc_metro, att_metro], max_drift}
lane/<r>/accs/<obs>/... (measure phase only)
-- kind == "cmc"  (additive, 2026-11-21 — constrained MC, run_cmc; C6):
plan/*            as above, with or_per_metropolis = 0 and sweep_tasks = 1
                  (fixed: the constrained sweep is serial and has no
                  overrelaxation; a reader refuses other values)
constraint/{direction (3, Float64 — the run's n̂ bitwise), weights
            (n_cell_atoms, Float64 — resolved against the Hamiltonian, signed
            zeros folded to +0.0), partner_rule ("uniform")}
start/config      3 × n_sites Float64 — the constrained start (the default
                  pattern or the rotated `init`) that a carryover = false
                  ladder returns to
progress/*, npoints, points/<i>/*, chain/*, has_accs, accs/<obs>/*   as "mc"
```

The `gpu_pt` lane block is deliberately tiny next to a CPU chain: the keyed
device Philox has **no stream state**, so `(dev_seed, sweep_index)` — two
integers — replace the CPU chain's `rng` + `site_rngs` words entirely, and the
host mirror's RNG streams are not stored at all (the GPU driver never consumes
them; a resumed mirror carries empty `site_rngs`, loud on any future *indexed*
misuse). Two C4 statements read differently for this kind: (1) no per-lane
phase counter is stored — `phase_sweeps` is `progress/done`, valid because
every lane advances in lockstep; (2) the rows-rebuild bit-identity does not
follow from purity alone — the run's device rows were written by
`_zlm_row_device!` while the restore recomputes them with the host `_zlm_row!`,
so it rests on the **G4 bitwise device-row ≡ host-row gate**
(`test/unit/test_gpu.jl`), a load-bearing coupled site of this format.
Resume: `resume(path, gH::GPUTiledHamiltonian)` on a matching backend — the
bitwise scope is the G3/G8 device contract (seed, backend, workgroupsize,
package + Julia version), the exchange uniforms being host-Xoshiro as in `pt`.
Write points mirror `pt` (segment boundaries + the thermalization→measurement
boundary, no end-of-run write); each lane write downloads the device state
fresh (`to_host!` — copies only, no RNG, no device mutation).

## C4 — what makes resume bit-identical

- `config` stored exactly; `zrows` are **rebuilt** from it (`Zlm_unsafe` is a pure
  function — same bits), while `energy` is restored **verbatim** (recomputing
  would erase the incremental value the trajectory depends on).
- Xoshiro state is captured generically over `fieldnames(Xoshiro)` (5 words on
  Julia 1.12) and rebuilt with `Xoshiro(words...)`; a word-count mismatch (another
  Julia's layout) errors instead of silently reseeding. Draw-stream equality is
  gated over 100 draws.
- All counters (acceptance windows, `since`-last-write, phase sweep counts, PT
  parity and swap tallies) are stored; every schedule (adapt/renorm/measure/
  checkpoint/exchange) is deterministic in them.
- LogBinner cascades and BinStore partial bins are stored in full, so error bars
  continue exactly (restore-path inner constructors on both types).
- Write points: MC — every `checkpoint_interval` sweeps + every temperature
  boundary; PT — at segment boundaries once `checkpoint_interval` sweeps have
  accumulated + the thermalization→measurement boundary. `interval = 0` ⇒
  boundaries only.
- Resume boundary semantics match the uninterrupted control flow exactly: a
  temperature-boundary checkpoint re-runs the `carryover = false` restart draw on
  the restored RNG (as the uninterrupted run would); a mid-phase checkpoint skips
  straight into the loop at `sweep + 1`.
- Per kind, `"cmc"`: the `carryover = false` restart of a constrained run is
  **deterministic and consumes no RNG** (`constrained-mc.md` D4) — it copies the
  stored `start/config` back into the chain and resets the step. So the restart
  replays from the file instead of from a restored RNG draw, and the start must
  be stored: for a user `init` it is not derivable from anything else in the
  file. The mover holds no chain state (every sweep recomputes `M` from the
  configuration), so the stored chain block, the constraint, and the start
  determine the resumed trajectory completely.

## C5 — resume contract

`resume(path, H; observables = nothing, evaluables = nothing, checkpoint = path)`
— the caller re-supplies the Hamiltonian and the observable/evaluable
*functions* (closures are not serializable); the file's `model_fingerprint` and
observable names/ncomps are checked and mismatches error. `nothing` resolves to
the defaults of the kind, AFTER the kind (and, for `"cmc"`, the `constraint/`
group) is read: `standard_observables(H)` / `standard_evaluables()` for `"mc"` /
`"pt"`, `cmc_observables(H, constraint)` / `cmc_evaluables()` for `"cmc"`.
Order of the reads and checks: schema version → fingerprint → kind (a
`"gpu_pt"` or unknown kind is refused here, by name, before any default is
resolved) → for `"cmc"` the constraint (refusing a `partner_rule` other than
`"uniform"`) → defaults → `_check_observables` → name/ncomp comparison →
`_check_evaluables` → the kind's body (for `"cmc"`: the plan's fixed values,
the start, the chain, and the consistency door of C6). Explicitly passed lists behave as before. The returned result covers the
**whole** run (completed `TempResult`s are stored in the file as plain data and
re-emitted). By default the resumed run keeps checkpointing to the same path with
the stored cadence.

## C6 — kind `"cmc"` (constrained MC, additive under v3)

Added 2026-11-21 with `run_cmc` checkpointing (`constrained-mc.md` D6),
following the `gpu_pt` precedent: a new kind under the same schema version,
no change for existing kinds or files. The constraint is run configuration,
not model identity, so `model_fingerprint` is unchanged and the constraint
lives in its own group, written by `_write_header` on every write (each write
is a fresh file). Write points and the end-of-temperature boundary write are
those of `"mc"` (the shared `_mc_loop!`). `resume(path, H)` rebuilds the
constrained mover from `constraint/` and `start/config`; the completed-file
early return builds a `CMCResult` (for `"mc"`, an `MCResult`). The stored
direction is taken bitwise (no second normalization — it is not idempotent at
the ulp level) and the weights as the resolved vector.

**Direction lanes** (`run_cmc` over a vector of constraints, `constrained-mc.md`
D4) add no layout: lane `k` writes an ordinary `"cmc"` file named
`<stem>.lane<k>.jld2` (`stem` = the given path without a trailing `.jld2` in any
letter case, else the path itself; a path naming a directory or leaving an empty
stem is refused) holding its own constraint and lane seed (`plan/seed`), and
resumes alone. No file describes the lane set, and files of an earlier call with
the same stem are not removed.

**Consistency door (read side, non-projecting).** With the stored `n̂` and
weights, `M = Σ_{s∈𝒞} w_s e_s` is computed for both `start/config` and
`chain/config`; the file is refused as corrupted unless `M·n̂ > 0` and
`|M⊥| ≤ 1e-8 Σ|w|`. A legal file is far inside both bounds (the start points
along `n̂` to rounding; the pair moves keep `|M⊥|` at the rounding level), while
a rewritten direction, weight vector, or configuration would otherwise resume
into another ensemble with at most a drift warning. It runs before the
completed-file early return as well (that path returns the stored chain).

Gates: `test/unit/test_cmc_checkpoint.jl` — G13 resumes with the file position
asserted (mid-measure with `carryover` true and false with a user `init`,
mid-thermalization inside an adaptation window, a double resume whose second
writer is itself a resumed run, the boundary-only cadence, a field model with
signed weights, default arguments on an idempotent and a non-idempotent
direction), the comparison of the final file's chain, `start/config`, and
`constraint/` with the uninterrupted run's, the constraint round-trip, and the
doors (including the consistency door on rewritten direction, start, and chain).

**Opened by an older package version.** Two cases, by the schema version the
older version reads:

- **Schema-v3 versions without kind `"cmc"`** (`855c17d` through `ea4f40a`)
  do not reach their "unknown checkpoint kind" error with default arguments:
  `resume(path, H)` defaults to `standard_observables(H)`, the schema version
  and the fingerprint match, and the observable-name comparison fails first —
  `ErrorException("the resumed observables (names/ncomps) do not match the
  checkpoint; stored: [\"energy\", \"energy2\", \"torque\", \"M_par\",
  \"constraint_residual\"] with [1, 1, 3, 1, 1]")` (with `"torque_zeeman"` /
  `3` appended for a field model; the run's own names if it used custom
  observables). Passing the CMC observables with the default
  `standard_evaluables()` fails next, in the resume-time `_check_evaluables`
  (present in all of these versions — it landed in `4a2a8d6`): an
  `ArgumentError`, `:susceptibility` needs `:m2`, which is not measured. Only
  observables with the stored names and component counts together with
  evaluables they satisfy reach `"unknown checkpoint kind cmc"`.
- **Earlier versions** (schema v2 and before) fail at the schema-version
  check: `"checkpoint schema v3 ≠ v2 of this package version"` (`≠ v1` for
  the first checkpointing versions).

So the failure differs from `gpu_pt`'s (rejected by kind, by name), but it is
equally loud: no version continues a `"cmc"` file as another kind.

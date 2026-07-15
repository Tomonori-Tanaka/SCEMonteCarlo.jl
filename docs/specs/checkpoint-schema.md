# Decision record — checkpoint schema (v1) and bit-identical resume

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

```
schema_version    Int     == 2, hard-checked on load
kind              String  "mc" | "pt"
julia_version, package_version   String (informational)
model_fingerprint UInt64  stable FNV-1a over (n_cell_atoms, dims, every term's
                          coef/atoms/shifts/ls/folded) — NOT Base.hash (which is
                          Julia-version-dependent); mismatch on resume ⇒ error
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
```

## C3 — what makes resume bit-identical

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

## C4 — resume contract

`resume(path, H; observables, evaluables, checkpoint = path)` — the caller
re-supplies the Hamiltonian and the observable/evaluable *functions* (closures
are not serializable); the file's `model_fingerprint` and observable
names/ncomps are checked and mismatches error. The returned result covers the
**whole** run (completed `TempResult`s are stored in the file as plain data and
re-emitted). By default the resumed run keeps checkpointing to the same path with
the stored cadence.

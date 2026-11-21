# Checkpointing and restart

```@meta
CurrentModule = SCEMonteCarlo
```

Long runs checkpoint to a JLD2 file and restart **bit-identically** — the resumed
trajectory, error bars, and final configurations equal the uninterrupted run's
exactly (this is tested with `==`, not `≈`).

```julia
run_mc(H; temperature = [900, 600, 300], sweeps_measure = 10^6,
       checkpoint = "run.jld2", checkpoint_interval = 50_000, seed = 1)

# …after a crash / walltime kill:
result = resume("run.jld2", H)      # returns the FULL run's MCResult
```

The same keywords work on [`run_pt`](@ref) (writes at segment boundaries) and on
[`run_cmc`](@ref) (as `run_mc`; `resume` returns the full `CMCResult`, and the
file also stores the constraint and the constrained start, so the resumed run
needs only `H` — with the default observables; custom lists must be passed
again, built from the same constraint). `checkpoint_interval = 0` writes only at natural boundaries
(MC and CMC: each temperature; PT: the thermalization→measurement boundary).
`run_cmc` over a vector of constraints writes one file per lane,
`"<stem>.lane<k>.jld2"` (`stem` is the path without a trailing `.jld2`); each is
an ordinary constrained-run file — resume the unfinished lanes one by one with
`resume(file, H)` (there is no resume of the whole set), and a completed lane's
file returns its stored result, so pass `checkpoint` for long scans. Files of an
earlier call with the same stem are not removed (a lane that failed before its
first write, a re-run with fewer lanes): check their constraint and seed, or use a
fresh stem.

## The resume contract

- The caller re-supplies `H` and any custom `observables` / `evaluables` —
  function objects are not serialized. Left out, they default to the default
  lists of the driver that wrote the file (the standard lists for `run_mc` /
  `run_pt`, `cmc_observables(H, constraint)` / `cmc_evaluables()` with the
  stored constraint for `run_cmc`). Custom lists for a `run_cmc` file must be
  built from that same constraint: a set from `cmc_observables` built for
  another one passes the name check but is refused (`ArgumentError`) before the
  run continues. The file stores the **model fingerprint**
  and the observable names/component counts, and errors on any mismatch (a resume
  against different physics never silently continues). The fingerprint covers the
  model's coefficient tensors bit for bit, and for an SCEFitting model those come
  out of LAPACK — they can differ in their last bits between BLAS builds, so
  resume a checkpoint on the same machine family that wrote it (macOS ↔ Linux
  resumes are rejected as a mismatch).
- The returned result covers the whole run: completed temperatures are stored in
  the file as plain data and re-emitted.
- By default the resumed run keeps checkpointing to the same path with the stored
  cadence (`checkpoint = nothing` disables, `checkpoint_interval` overrides).
- Resuming a run that already **finished** simply returns its result unchanged —
  `resume` is idempotent, so a blind "resubmit and `resume` if the file exists"
  retry loop (see [parallelism](parallelism.md)) is safe.

## Why it is bit-identical

The file captures configs, the incremental energy (restored verbatim), Xoshiro
RNG words, every schedule counter, and the full binning-accumulator state —
and every schedule in the package is deterministic in those counters. Writes are
atomic (temp file + `mv`) and consume no RNG, so checkpointing never perturbs the
run it protects. One writer per checkpoint path — two concurrent runs must not
share one. Schema and rationale: `docs/specs/checkpoint-schema.md`.

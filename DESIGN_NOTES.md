# Design notes index

Index of design discussions, investigations, and on-hold ideas. Follow the
links for detail. Active development units and the standing decision records
live under `docs/specs/`; historical benchmarks (incl. the GPU validations on
kugui) live in `.claude/bench_log.md` and `git log`.

Operating rules: [`docs/design-notes/README.md`](docs/design-notes/README.md).

## Design proposals

| Topic | Status | Last update |
|---|---|---|

Completed proposals are folded into their corresponding spec under
`docs/specs/` and removed from this index.

## Investigations and standing rationale

- The decision records under [`docs/specs/`](docs/specs/README.md) — tiling,
  stationarity, binning, PT determinism, checkpoint schema, cell reduction,
  ground-state search, the GPU feasibility study and prototype record.
- `CLAUDE.md` "Project goal" — the SpinClusterMC.jl pain points this design
  avoids (God-struct, module-level caches, split temperature units,
  hard-coded observables, payload duplication, positional serialization).

## Performance backlog

- `.claude/bench_log.md` #1–#2 — GPU revival re-validation and `gpu_run_pt`
  device validation on kugui (A100); the 16³ l044 tables do not fit on the
  device (known limit, `docs/specs/gpu-prototype.md`).

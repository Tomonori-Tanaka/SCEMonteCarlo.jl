# Requirements: <title>

Status: draft (YYYY-MM-DD)

## Goal

<!-- What we want to achieve. 1-3 sentences. -->

## Background

<!-- Why now. Constraints, requester, prior investigations, the decision
     record(s) this touches. -->

## Scope

Includes:

- <!-- e.g., a new update scheme in `updates.jl` -->

Excludes:

- <!-- e.g., the GPU mirror (separate spec) -->

## Invariants

<!-- Things that must NOT change. -->

- Spins stay unit vectors; inactive sites stay bitwise frozen.
- `(4π)^(body/2)` applied once in the `TiledHamiltonian` ctor; `j0` excluded.
- `temperature` xor `kT`; β only in accept steps.
- Detailed balance / stationarity of every update (U1 color classes).
- Bitwise contracts: serial ≡ parallel, PT thread-count independence,
  resume ≡ uninterrupted (mid-measure), CPU ≡ GPU.
- Checkpoint schema version; dependent-package-facing names
  (`energy_gradient!`, `model_fingerprint`, `_gradient_lane_ref!`).
- ...

## Completion criteria

- [ ] <!-- e.g., `make test-all` passes -->
- [ ] <!-- e.g., new gate with an implementation-independent oracle; σ with headroom -->
- [ ] <!-- e.g., decision record `docs/specs/<topic>.md` updated -->

## References

- Related issues / PRs:
- Related decision records / design notes / upstream SLCEMonteCarlo.jl commits:

# `docs/specs/` index

Two kinds of document live here:

- **Decision records** (flat `<topic>.md`): the standing contracts the code
  cites by path — update the record whenever the contract moves.
- **Spec folders** (`[YYMMDD]-[slug]/` with `requirements.md` /
  `design.md` / `tasklist.md`): mid-sized or larger development units.
  Operating rules in the "Managing development units" section of
  [CLAUDE.md](../../CLAUDE.md); start from [`_template/`](_template/).

## Decision records

| Record | Contract |
|---|---|
| [hamiltonian-tiling.md](hamiltonian-tiling.md) | Supercell tiling of `multipole_terms`, `shifts` anchoring, scale-once, ΔE locality |
| [updates-stationarity.md](updates-stationarity.md) | Metropolis / overrelaxation stationarity, color classes (U1), adaptive step |
| [binning-observables.md](binning-observables.md) | Log-binning errors, jackknife evaluables, the `C` / `χ` / `U` definitions |
| [pt-threads-determinism.md](pt-threads-determinism.md) | Replica exchange over threads; bit-reproducibility scope (P6) |
| [checkpoint-schema.md](checkpoint-schema.md) | JLD2 checkpoint schema (v3: fingerprint mixer fold), resume discipline |
| [cell-reduction.md](cell-reduction.md) | `reduce_cell`: verified re-expression in a smaller cell |
| [ground-state-search.md](ground-state-search.md) | On-sphere gradient descent, `find_ground_state` |
| [gpu-feasibility.md](gpu-feasibility.md), [gpu-prototype.md](gpu-prototype.md) | GPU study and the device implementation record (G1–G8) |
| [zeeman-field.md](zeeman-field.md) | External field: sign/units, body-1 Zeeman templates, append rule, fingerprint scope, `:M` / `:M_B` |

## Spec folders

| Spec | Status | One-line summary |
|---|---|---|
| [260822-zeeman-field/](260822-zeeman-field/) | landed (CPU); CUDA re-validation pending | Zeeman term: constant per-atom moments (μ_B) in a uniform field (T), represented as body-1 `ScaledTerm`s appended by the ctor — no kernel changes |

(Development before this index — the carve-out from SLCEMonteCarlo.jl and the
GPU prototype — is recorded in `CHANGELOG.md` and the decision records above;
it is not back-filled here.)

This table and the `Status:` line in each `tasklist.md` are duplicated
intentionally; update both when a spec lands.

## Completion criteria

Each `tasklist.md` ends with a shared exit checklist (see
[`_template/tasklist.md`](_template/tasklist.md)). In particular,
**`.claude/agents/` is easy to forget** — whenever module names or Makefile
targets change, sweep the agent files as well.

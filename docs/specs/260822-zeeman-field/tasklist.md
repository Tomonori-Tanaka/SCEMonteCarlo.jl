# Tasklist: Zeeman term — constant site moments in a uniform field

Status: draft v2 (2026-08-22) — body-1 cluster-term representation.

This file holds coarse-grained, commit-sized milestones. Day-to-day tracking
goes through `TaskCreate` in-session.

## Milestones

### M0 — pins before touching anything

- [x] Capture the field-free `model_fingerprint` of the dimer fixture at
      `c7a354a` and write the labelled regression pin (change detector) into
      `test/unit/test_zeeman.jl` — the one value this spec must not move.
      (dims (1,1,1) `0x84d69fe51471f311`, (2,2,2) `0x107b7c1f7b2cf03e`.)

### M1 — core (`units.jl`, `hamiltonian.jl`, `reduce.jl`)

- [x] `MU_B_EV_T` + docstring + export; `test_units.jl` value check against
      the CODATA ratio written out in the test.
- [x] Ctor keywords, door validation, `_zeeman_terms` generator (one term
      per `m_a ≠ 0` atom, only when `B ≠ 0`), append after scaling,
      `lmax = max(lmax, 1)`, `n_fitted_terms`, `magmoms` / `field` fields,
      `has_field`, `zeeman_energy`, `Base.show` split.
- [x] `ReducedCell` form forwards the keywords.
- [x] Gates (`test_zeeman.jl`): field identity (both dims, absolute
      tolerance; plus the `ReducedCell` form), `zeeman_energy`, structure
      gate (incl. the all-`l = 0` `lmax` case), `ΔE ≡` difference, own-spin
      independence, program ≡ reference bitwise with a field fixture,
      gradient closed form + FD, door errors, `magmoms`-only bitwise identity
      (terms / activation / energies), zero-moment sublattice frozen with a
      field.
- [x] Commit: `feat(hamiltonian): zeeman term as body-1 cluster terms`.

### M2 — dynamics-level consequences, observables, checkpoint identity

- [x] Langevin gate on the host (record σ, seed, mutation size in the test
      comment); OR exactness gate with a field; ground-state gate with
      default `gtol` / ladder.
- [x] `:M` / `:M_B` observables + hand-sum gate; `magmoms`-only seeded
      `run_mc` bitwise identity including `:m` / `:sublattice_m`;
      `binning-observables.md` definitions + the B3 sentence on field-induced
      activation; `energy_gradient!` docstring (field is in `G`, names the
      dependent-package hazard and `has_field`).
- [x] `_fingerprint` explicit conditional mixing; additive `zeeman/` group;
      resume gates (MC / PT bit-identical with a field; mismatch errors for
      the three collision pairs); M0 pin still passes.
- [x] `test_parallel.jl` / `test_pt.jl` field variants (placed in
      `test_zeeman.jl`: serial ≡ parallel with a field; PT run + resume with a
      field).
- [x] Commit: `feat(observables): magnetization in μ_B, field projection, fingerprint scope`.

### M3 — device coverage, records, docs, panel

- [ ] Body-1 fixture in `test_gpu.jl` G3 / G7 bitwise gates (Zeeman-only
      site + fitted-plus-Zeeman site; all-body-1 model for the zero-length
      table case); device Langevin gate; GPU-PT run + resume gate with a field.
- [ ] CUDA validation on a kugui node recorded in `.claude/bench_log.md`
      (bitwise gates + Langevin on the device, zero-length tables) — may
      trail the merge; flagged in `CHANGELOG.md` until done.
- [ ] `docs/specs/zeeman-field.md` (Z1 sign/units, Z2 body-1 folding and the
      `coef = 1.0` / no-`4π` rule, Z3 append rule, Z4
      fingerprint/checkpoint, Z5 gates); T4 / scale-once note in
      `hamiltonian-tiling.md`; `checkpoint-schema.md` fingerprint scope.
- [ ] `docs/src/guide/field.md` + `make.jl`; `observables.md` (drop
      "magnitudes are not part of the model"), `running.md`,
      `ground_states.md`, `gpu.md`; `api.md`; `SPEC.md` (exports, dependency
      boundary `Harmonics.N1`, `H.terms` split on the public `ScaledTerm`
      tier, module table); `CLAUDE.md` coupled-sites bullet; `.claude/agents/`
      sweep; `CHANGELOG.md`.
- [ ] Tier 2 panel (numerical / maintainability / performance / api); resolve.
- [ ] Commit: `docs(zeeman): decision record, guide, and scaffolding sweep`.

## Exit checklist

Run through every item once implementation lands. ~~Strike through~~ items
that do not apply.

- [ ] `make test-all` passes (4 threads).
- [ ] `make docs` builds (strict).
- [ ] If results or a determinism contract changed: test added with an
      implementation-independent oracle; bitwise gates still hold.
- [ ] If `src/gpu/` changed: device validation on a CUDA node recorded in
      `.claude/bench_log.md`. (No device *code* changes; the new fixture and
      zero-length-table case are still validated on CUDA.)
- [ ] If public API changed: `SPEC.md` and `docs/src/api.md` updated;
      dependent-package-facing names flagged.
- [ ] Decision record(s) under `docs/specs/` updated.
- [ ] If a hot path was touched: before / after recorded in
      `.claude/bench_log.md` (no kernel change; one extra body-1 instance per
      moment-carrying site in the adjacency — `bench-sweeps` before/after on
      the l02 fixture with a field).
- [ ] Tier 2 review panel run and findings resolved.
- [ ] If module names or Makefile targets changed: `.claude/agents/` swept.
- [ ] `CHANGELOG.md` `[Unreleased]` updated.
- [ ] `Status:` line here and the table in `docs/specs/README.md` updated.
- [ ] Implementation commit hash appended below.

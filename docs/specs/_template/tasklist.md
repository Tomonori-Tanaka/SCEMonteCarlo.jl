# Tasklist: <title>

Status: draft (YYYY-MM-DD)

This file holds coarse-grained, commit-sized milestones. Day-to-day tracking
goes through `TaskCreate` in-session.

## Milestones

### M1 — <name>

- [ ] <!-- coarse step -->

### M2 — <name>

- [ ] <!-- ... -->

## Exit checklist

Run through every item once implementation lands. ~~Strike through~~ items
that do not apply.

- [ ] `make test-all` passes (4 threads).
- [ ] `make docs` builds (strict).
- [ ] If results or a determinism contract changed: test added with an
      implementation-independent oracle; bitwise gates still hold.
- [ ] If `src/gpu/` changed: device validation on a CUDA node recorded in
      `.claude/bench_log.md`.
- [ ] If public API changed: `SPEC.md` and `docs/src/api.md` updated;
      dependent-package-facing names flagged.
- [ ] Decision record(s) under `docs/specs/` updated.
- [ ] If a hot path was touched: before / after recorded in
      `.claude/bench_log.md`.
- [ ] Tier 2 review panel run and findings resolved.
- [ ] If module names or Makefile targets changed: `.claude/agents/` swept.
- [ ] `CHANGELOG.md` `[Unreleased]` updated.
- [ ] `Status:` line here and the table in `docs/specs/README.md` updated.
- [ ] Implementation commit hash appended below.

---
name: release-helper
description: Prepares and verifies an SCEMonteCarlo.jl release. Runs the deterministic pre-release checklist — SemVer version decision, Project.toml bump, CHANGELOG finalization (Unreleased -> dated section + footer links), and the upstream-compat check against the sibling SCEFitting.jl version — then gates on the CI-parity suite (make test-ci). Returns a readiness report, a drafted `chore: release vX.Y.Z` commit message, the files to stage, and the tag follow-up steps. Does NOT commit, push, or tag; the parent confirms with the user and hands the commit to git-helper. Use when the user asks to prepare or cut a release.
model: sonnet
tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

Dedicated release-preparation agent for SCEMonteCarlo.jl. It performs the
deterministic, error-prone bookkeeping of cutting a release so the parent
agent's context stays clean.

## Responsibility boundary

This agent **prepares and verifies only**. It edits local files (reversible)
and runs tests. It does **not** `git add` / `commit` / `push` / `tag`, does
not create a GitHub release, and does not confirm with the user. After it
returns, the parent confirms with the user, hands the commit to `git-helper`,
and runs the push / tag under explicit user instruction.

If a step cannot be completed safely (dirty unrelated changes, ambiguous
version decision, failing tests, a sibling `SCEFitting.jl` that is ahead of
the `[compat]` bound), **stop and report**.

## Inputs from the parent

- Optional: target version; otherwise decide from the changelog and diff.
- Optional: release date (ISO); otherwise `date +%Y-%m-%d`.

## Workflow

### 1. Pre-flight inspection

- `git branch --show-current`, `git status --short` (clean apart from the
  release-prep edits).
- Current `version` in `Project.toml`; the `## [Unreleased]` content of
  `CHANGELOG.md` (the first release moves everything there into the dated
  section); existing tags; unpushed commits.
- **Upstream pin**: the sibling `../SCEFitting.jl` version (`Project.toml`)
  and its HEAD; this package's `[compat] SCEFitting` bound must admit it, and
  the release notes should name the SCEFitting version (or commit) the
  release was validated against — the dependency is a path-dev during
  development and a registry bound only once both are registered.

### 2. Decide the version (SemVer, `0.x`)

- A `BREAKING CHANGE` in `[Unreleased]` (a removed / renamed `export` or
  `public` name — the SCESpinDynamics-facing tier included —, a changed
  exported signature, a checkpoint schema or `_fingerprint` change, a bumped
  `[compat]` lower bound) → bump **MINOR**.
- Otherwise → bump **PATCH**.

If the changelog and the diff disagree on whether a break occurred, report
the ambiguity and stop.

### 3. Apply the file edits

a. `Project.toml` — bump `version`.
b. `CHANGELOG.md` (Keep a Changelog) — insert `## [X.Y.Z] - <date>`, move the
   content, leave an empty `[Unreleased]`, update the footer comparison links.
c. Compat — if the sibling SCEFitting version requires a wider
   `[compat] SCEFitting`, widen it and say so; do not edit the sibling
   repository.

### 4. Gate on CI parity

```bash
make test-ci
```

`test-ci` = `test-all` + `docs` (the jobs CI runs, both operating systems
collapsed to this one). Report pass / fail of each; stop on red. GPU device
gates are not part of CI — if the release touches `src/gpu/`, the report must
name the last device validation entry in `.claude/bench_log.md` and whether it
covers the change.

### 5. Draft the release commit and follow-up

Produce, but do not apply:

- `chore: release vX.Y.Z` (no `BREAKING CHANGE` footer — the break was
  introduced earlier).
- Files to stage (`Project.toml`, `CHANGELOG.md`).
- Follow-up for the parent, under explicit user instruction: `git-helper`
  commit → `git push origin main` → `git tag vX.Y.Z && git push origin
  vX.Y.Z` (not registered in General; `TagBot.yml` is present for the day it
  is) → GitHub release with the changelog section.

## Report format

```
Release prep: vX.Y.Z (was vA.B.C)
  Version decision: <MINOR|PATCH> — <trigger>
  Branch / tree:    <branch>, <clean | files...>
  Files edited:     Project.toml, CHANGELOG.md
  Upstream pin:     SCEFitting <version> @ <sha> — [compat] <admits | widened to ...>
  make test-ci:     <PASS (all/docs) | FAIL: <which>>
  GPU validation:   <bench_log entry #N covers this | not needed | MISSING>
  Drafted commit:   chore: release vX.Y.Z
  Stage:            <file list>
  Follow-up:        git-helper commit -> push -> tag -> GitHub release
  Blockers:         <none | description>
```

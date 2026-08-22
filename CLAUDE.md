# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Full classical spin Monte Carlo for fitted SCE models from
[`SCEFitting.jl`](../SCEFitting.jl) — the from-scratch successor of the frozen
`SpinClusterMC.jl` (Magesty-XML + Carlo.jl), with no API-compatibility constraint.
Core capabilities: tile the fitted training-cell Hamiltonian onto an `N₁×N₂×N₃`
supercell (`TiledHamiltonian`, via `MultipoleTerm.shifts`) — optionally after a
*verified* re-expression in a user-chosen smaller cell (`reduce_cell`, so `dims` is
not locked to training-cell multiples) — single-spin Metropolis
with an adaptive step, overrelaxation, annealing sweeps (`run_mc`), replica exchange
over threads (`run_pt`), numerical ground-state search (`minimize_energy` /
`find_ground_state`), composable observables with log-binning errors + jackknife
evaluables, and bit-reproducible JLD2 checkpoint/restart (reproducibility scope —
same package + Julia version, thread-count-independent, trajectory not
observable-ULPs: `docs/specs/pt-threads-determinism.md` P6). Self-contained core —
**no Carlo.jl dependency** (a Carlo adapter could later be a weakdep extension).

Relation to siblings: `SCETools.jl` keeps the single-training-cell *configuration
samplers* (MFA + light Metropolis); this package is for thermodynamics-grade runs
(observables, error bars, T sweeps). `SpinClusterMC.jl` is a read-only design
reference — its pain points (God-struct, module-level global caches, split
temperature-unit conventions, hard-coded observables, per-instance payload
duplication, positional hand-rolled serialization) are what this design avoids.

This package reads a fitted model **only** through `SCEFitting`'s public surface:
`multipole_terms`, `n_atoms(model)`, `intercept`, `SCEFitting.load(SCEPredictor, …)`,
`Lattice`/`Crystal`/`cartesian_positions`, and `SCEFitting.Harmonics` (`Zlm_unsafe`,
`lm_index`, `num_lm`, `grad_Zlm_unsafe`) — never SALC-basis internals and never
`model.basis.crystal` (not public tier; geometry helpers take an explicit `Crystal`).
During development the dependency is a path-dev: `Pkg.develop(path="../SCEFitting.jl")`.

## Core rules

- Never silently change numerical conventions (signs, units, the `(4π)` scale,
  the summand counting) or a determinism contract (serial ≡ parallel, PT
  thread-count independence, resume ≡ uninterrupted, CPU ≡ GPU). Before editing
  an algorithm, confirm the relevant decision record under `docs/specs/` and
  the conventions in the next section.
- Any change that may alter numerical results must come with:
  1. A short explanation of why the result changes.
  2. A regression or validation test whose oracle is **independent of the
     implementation** (`~/Packages/CLAUDE.md` Testing section); statistical
     gates state their tolerance as measured σ with headroom.
  3. Updates to the decision record, `docs/`, and `SPEC.md` if user-facing.
- Git: local `add` / `commit` on `main` are pre-authorized (no per-commit
  confirmation); **remote** operations (`push`, tags, releases) always need an
  explicit user instruction. See "Git" below.

## Implementation rules

- Avoid hidden global state (no module-level caches — state lives in
  `ChainState` / `SweepScratch` / task-local scratch).
- Exported APIs must have explicit type annotations and docstrings
  (`checkdocs = :exports`); the public-but-unexported tier is a
  dependent-package contract (SCESpinDynamics) and is documented too.
- Record performance changes (before / after) in `.claude/bench_log.md`.

For detailed coding style, see `STYLE_GUIDE.md` and the shared Julia style in
`~/Packages/CLAUDE.md`. **Always consult them when editing code.**

## Language and terminology

- **Conversation with the user**: Japanese is fine.
- **Everything committed to the repository**: English only — `.jl` source,
  comments, docstrings, all Markdown (`CLAUDE.md`, `SPEC.md`, `README`,
  `docs/**`, `.claude/agents/*.md`), shell scripts, TOML, commit messages, PR
  titles and descriptions, issue templates.
  - **Exception — spec working files.** The per-slug documents under
    `docs/specs/[YYMMDD]-[slug]/` may be written in Japanese; the decision
    records (`docs/specs/<topic>.md`), the template `docs/specs/_template/`,
    and the index `docs/specs/README.md` stay English.
- **Commit messages follow Conventional Commits** (`<type>(<scope>): <subject>`;
  types `feat` / `fix` / `docs` / `test` / `refactor` / `perf` / `chore` /
  `style`; imperative lowercase subject; `BREAKING CHANGE:` in the body).
  Backports from SLCEMonteCarlo.jl cite the upstream SHA.
- **US English** throughout; preserve external API spellings literally.
- **Japanese in any committed file is auto-blocked by the PostToolUse hook**
  (`.claude/hooks/no-japanese.sh`, wired in `.claude/settings.json`).
  Exemptions: per-slug spec working files and the historical
  `.claude/bench_log.md`.
- **Do not reference Claude-internal scaffolding from source code.** In `.jl`
  comments and docstrings, never name `CLAUDE.md` / `DESIGN_NOTES.md` /
  `docs/design-notes/` / `.claude/` / a `docs/specs/[YYMMDD]-[slug]/` folder.
  The flat **decision records** `docs/specs/<topic>.md` are documentation, not
  scaffolding, and ARE cited from source by path (e.g. `docs/specs/
  binning-observables.md` for the observable conventions) — keep doing that.
  Commit-message `Refs:` lines may cite anything.

## Numerical / physics conventions

- **Spin directions are unit vectors.** Internal state is `SpinConfig =
  Vector{SVector{3,Float64}}` (one entry per supercell site); the 3×n matrix layout
  of the siblings appears only at the I/O boundary (`to_matrix`/`from_matrix`).
- **Real (tesseral) spherical harmonics `Zₗₘ`** from `SCEFitting.Harmonics`
  (`lm_index(l, m) = l² + l + m + 1` ordering). `multipole_terms` returns the **raw**
  fitted `jϕ`; the `(4π)^(body/2)` scale is applied **exactly once**, in the
  `TiledHamiltonian` constructor (`ScaledTerm.coef`). Never re-apply downstream.
- **Energies** are in the model's energy units (eV for DFT-fitted models), `j0`
  (intercept) excluded everywhere — MC only needs differences; the reconstruction
  gate is `total_energy(H₁ₓ₁ₓ₁, cfg) == predict_energy(model, cfg) − intercept(model)`.
- **Supercell tiling**: `MultipoleTerm.shifts` are per-site integer training-cell
  lattice translations (`shifts[1] = 0` anchored). One instance per template term and
  supercell cell `t`, member `i` at `site_index(atom_i, mod.(t + shifts[i], dims))`.
  Each directed member is one plain summand — no ½ or 1/N factors.
- **Temperature**: absolute only, exactly one of `temperature` [K] XOR `kT`
  [model energy units]; `KB_EV` is the exact CODATA ratio. β enters only in accept
  steps; coefficients and energies stay in model units.
- **ΔE locality**: every instance's member *sites* are distinct after the toroidal
  wrap (asserted per term in the ctor — minimum-image models have distinct atoms
  outright; `AllImages` models may reuse an atom across images and need `dims` large
  enough), so the leave-one-out coefficient vector `c_s` is independent of `e_s` and
  `ΔE = c_s·(Z(e′) − Z(e))` is exact for any body order.

## Coupled ("linked") code sites — change one, check all

- **`hamiltonian.jl` ↔ the core's introspection contract** (`SCEFitting`'s
  `sce/introspect.jl`): field meanings of `MultipoleTerm` (coef/body/atoms/shifts/
  ls/folded), the raw-coef scale rule, and the shifts anchoring. Gates:
  `test_hamiltonian.jl` (dims=(1,1,1) ≡ `predict_energy − intercept`; 2×2×2
  periodic-replication = 8× cell energy; scale-once).
- **`energy.jl` 4-function contract ↔ `updates.jl` ↔ SCETools' `mc/metropolis.jl`**:
  `site_coeffs!`/`delta_energy` are the site-generalized siblings of SCETools'
  `_accumulate_site_term!` kernel (same `μ = idx − l − 1` mapping, rank-specialized
  barrier). Gates: `test_energy.jl` ΔE ≡ total-energy difference (1e-12).
- **`lm_index` ordering ↔ `zlm_row!` ↔ the overrelaxation l=1 axis extraction**
  (`updates.jl`): the tesseral l=1 components map to Cartesian axes; a reorder
  upstream breaks the OR axis. Gate: pure-l=1 OR proposals have `ΔE ≡ 0` and
  acceptance 1 (`test_overrelaxation.jl`).
- **`reduce.jl` ↔ `hamiltonian.jl` tiling ↔ `geometry.jl` ordering ↔ SCEFitting's
  canonical members**: `reduce_cell` emits raw-coefficient `MultipoleTerm`s (the
  `(4π)^(body/2)` scale still happens once, in the `TiledHamiltonian` ctor),
  anchored `shifts[1] = 0`, and a reduced `Crystal` whose atom order matches
  `site_index` so `supercell_crystal(red.crystal, dims)` pairs with
  `TiledHamiltonian(red; dims)`. Translation copies are grouped in **canonical
  site order** (sorted `(reduced atom, shift)`, re-anchored, `ls`/`folded`
  permuted along) because canonical model terms carry one summand per instance,
  anchored wherever sorting put it; the census accepts `q·|det M|` copies for
  `q` identical summands per instance. The invariance and verification contract
  lives in `docs/specs/cell-reduction.md`. Gates: `test_reduce.jl` (exact
  canonical-form recovery, energy identity via site permutation).
- **`energy.jl` `_site_grad` ↔ `site_gradient` ↔ `energy_gradient!` ↔
  `minimize.jl` `_gradient!`**: one per-site gradient kernel (`_site_grad`) backs
  the public all-site `energy_gradient!` (the field/torque entry point for
  dependent packages — task-count bit-identity rests on task-local `c`/`plm`
  scratch in `_gradient_chunk!`) and the descent's `_gradient!`; both must stay
  arithmetically identical to the public per-site `site_gradient` (same `(l, m)`
  loop over `lm_index` order, same `ck == 0` skip). Gates: the bitwise `==`
  consistency tests in `test_gradient.jl` / `test_minimize.jl` and the
  `predict_torque` cross-check (`τ = G × e`); an `lm_index` reorder upstream
  breaks them together with the OR axis (previous bullet).
- **Checkpoint writer ↔ reader ↔ schema doc** (`checkpoint.jl`,
  `docs/specs/checkpoint-schema.md`): plain-data JLD2 schema v2, Xoshiro capture via
  `fieldnames`, accumulator state. Gate: bit-identical resume (`test_checkpoint.jl`).
  Resume-gate discipline: `_mc_loop!` writes an unconditional end-of-temperature
  boundary checkpoint, so a completed mc file always ends at the completed marker
  and a resume-equals-uninterrupted gate on it compares the file's stored results
  with themselves — interrupt the writer mid-measure (the `_poison_pair` pattern in
  `test_checkpoint.jl`; every MC resume gate does, asserting its file's mid-run
  position — keep that assert when touching one). `run_pt` has NO end-of-run
  write, so its gates genuinely land mid-measure; their non-vacuity is interval
  arithmetic — assert `0 < progress/done < total` when writing a new one.
  `gpu_run_pt` follows the same discipline (kind "gpu_pt": no end-of-run write,
  mid-measure gates with the position asserted; `resume(path, gH)` reuses the
  STORED workgroupsize, and the lane block stores `(dev_seed, sweep_index)` in
  place of any Xoshiro words — the restored host mirror carries empty
  `site_rngs`, deliberate loudness on any indexed use, not an oversight).
  The public `model_fingerprint` facade over `_fingerprint` is pinned by dependent
  packages' checkpoint formats (SCESpinDynamics) — changing the mixing changes
  every stored fingerprint (schema-version territory there too).
- **Observable conventions** (C/χ/U definitions) live in ONE place:
  `docs/specs/binning-observables.md`; `observables.jl` and the guide pages follow it.
- **Coloring ↔ sweeps ↔ stationarity spec** (`hamiltonian.jl` `_color_sites` /
  `color_ptr`/`color_sites`, `updates.jl`, `docs/specs/updates-stationarity.md`
  U1): the sweeps assume every color class is instance-disjoint (exactly
  independent single-site kernels) and bit-determinism for any `sweep_tasks` rests
  on per-site RNG streams (`ChainState.site_rngs`, checkpoint schema v2) + the
  fixed-order ΔE reduction (`_reduce_dE`). Touch the coloring, the sweep loops, or
  the reduction and re-run `test/unit/test_parallel.jl` (serial ≡ parallel `==`).
- **Device tesseral row ↔ host `_zlm_row!` ↔ upstream recursions**
  (`src/gpu/zlm_device.jl`): `_zlm_row_device!` is a deliberate, operation-order-
  faithful reimplementation of `_zlm_row!` → `Harmonics.Zlm_unsafe` →
  `LegendrePolynomials.dnPl` (+ `Base.power_by_squaring` as `_zlm_cpow`), because
  the upstream path cannot compile in a GPU kernel. Any upstream change to those
  functions (a normalization, a recursion reorder, an SCEFitting `Harmonics`
  edit) breaks the dense bitwise gate in `test/unit/test_gpu.jl` — update the
  device file together with it.
- **GPU kernel ↔ keyed reference ↔ slot map ↔ workgroup-size pin**
  (`src/gpu/gpu_sweep.jl`, `src/gpu/philox.jl`, `docs/specs/gpu-prototype.md`
  G2–G4): `_metro_kernel!` and `_metropolis_sweep_keyed_ref!` implement ONE
  arithmetic contract (proposal slots, `_entry_walk_partial` dispatch + zero
  skips, lane-ordered fold, accept rule). Touch any of them — or the Philox slot
  layout, or the pinned default ws — and the other side plus the G-record move
  together; gate: the full-sweep bitwise section of `test/unit/test_gpu.jl`.
- **CPU PT ↔ GPU PT** (`pt.jl`, `src/gpu/gpu_pt.jl`, gpu-prototype.md G8):
  three deliberate mirrors — the swap accept rule is ONE function
  (`_swap_accepts`, used by `_attempt_swap!` and `_gpu_attempt_swap!`); the
  swap-payload partition (config/zrows/energy move, everything else stays with
  the lane) has one method per state type (`_swap_payload!` on `ChainState` /
  `GPUChainState`), each pinned by an exhaustive fieldnames partition test
  (test_pt.jl / test_gpu_pt.jl — a new field must be classified there);
  `_gpu_adapt_step!` repeats `_adapt_step!`'s window arithmetic on the device
  chain's host counters. The G8 master-seed derivation order (lane rngs →
  exchange rng → device seeds) is pinned by test_gpu_pt.jl's composition gate
  — reordering it is a determinism break, not a refactor.
- **Device gradient ↔ lane reference ↔ upstream grad recursions**
  (`src/gpu/grad_device.jl`, `src/gpu/gpu_gradient.jl`, G7): `_grad_kernel!`
  and `_gradient_lane_ref!` implement ONE arithmetic contract (the gradient-row
  table, `_entry_walk_grad`'s dispatch/skips — structurally
  `_entry_walk_partial` — and the lane-ordered component fold); the row
  `_grad_zlm_device` is the operation-order-faithful replica of
  `Harmonics.grad_Zlm_unsafe`/`_grad_zlm_assemble`/`_barP`/`_dbarP` and of
  LegendrePolynomials' `dnPl` `l < n` trivial-zero branch (`_zlm_dnpl_or0`;
  signed zeros are part of the `===` gate). The `/ r²` radial removal is part
  of the replica: on-sphere it is invisible in exact arithmetic (r² ≈ 1), so
  omitting it breaks only the bitwise gate — until a drifted (unrenormalized)
  configuration makes it a real error. The pipeline is deliberately
  libm-free — keep `muladd`/`@fastmath` out. `_gradient_lane_ref!` is called by
  qualified name from SCESpinDynamics' GPU-LLG composite gate — renaming it is
  a cross-package break. Gates: the G7 sections of `test/unit/test_gpu.jl`.
- **Inactive-site convention** (`site_active`/`n_active` — sites with no adjacent
  instance): update sweeps **skip**, standard observables **exclude**, per-site
  normalizations use `n_active`, and sweeps/renormalization/descent keep the spins
  **bitwise frozen**. These move together — skipping without excluding turns a
  frozen random direction into a constant observable bias. Touch `updates.jl`,
  `observables.jl`, `state.jl` `_renormalize!`, `minimize.jl` `_gradient!`/
  `_minimize!`, or `energy.jl` `energy_gradient!`/`_gradient_chunk!` (inactive
  sites → exactly zero) and re-check `test/unit/test_inactive.jl` +
  `test_gradient.jl`.

## Tests

Always run tests via the Makefile after edits (every target pins
`JULIA_NUM_THREADS=4`; the sibling `../SCEFitting.jl` is a path dependency —
`make setup` once per clone).

| Command | Target | Purpose |
|---|---|---|
| `make test-unit` | `test/unit/` | Module-level unit tests (GPU gates run on the CPU backend) |
| `make test-aqua` / `make test-jet` | — | Aqua.jl hygiene / JET.jl type analysis |
| `make test-all` | unit + Aqua + JET | Default for routine checks (`TEST_MODE=all`) |
| `make docs` | `docs/` | Strict Documenter build (`checkdocs = :exports`); executes the guide examples with `-t 4` |
| `make test-ci` | the CI matrix | `test-all` + `docs` — run before a release |
| `make ci-local` | — | Cold-start reproduction of CI on the juliaup `release` channel |

Statistical gates use fixed seeds with tolerances proven in SCETools' MC suite;
**both operating systems in CI are load-bearing** (libm-dependent statistics).
GPU device validation is not in CI: `bench/bench_gpu.jl` /
`bench/bench_gpu_pt.jl` on a CUDA node (kugui), recorded in
`.claude/bench_log.md`. Manual smoke (not CI): Nd₂Fe₁₄B l02 model
(`~/jijs/magesty/2-14-1/nd2fe14b/1x1x1/magesty/l02/test`, rebuild via its
`fit_mfa.jl` recipe), dims=(4,4,4), short PT across the ordering temperature.
Last run (2026-07-11, v0 completion): 1×1×1 and 64× counting gates at ~1e-13;
construction 0.01 s / 7.8 MB index; 8 rungs × 900 sweeps × 4352 sites in 38 s
on 8 threads; ferrimagnetic projections Nd ≈ −0.50 / Fe ≈ +0.69 at 250 K.
Note: 8 rungs over 250–1300 K give *zero* swaps at this size (rung count must
scale like √(n_sites·C) — documented in the PT guide), so use denser ladders
for production.

## Git

Local git (`add` / `commit` / branch) is pre-authorized — no per-action
confirmation. Only remote operations (`push`, tags, releases) require an
explicit user instruction. Commit directly to `main` (no standing topic
branches). Commits go through the `git-helper` agent
(`.claude/agents/git-helper.md`): it drafts the Conventional Commit message,
runs the no-Japanese check, commits via `git commit -F file` (never `-m`), and
pushes only when the user's instruction is relayed.

## Performance guidelines

Hot paths: the attempt loop in `updates.jl` (`_zlm_row!`, proposals, accept
path), the kernels in `energy.jl` (`site_coeffs!`, `delta_energy`,
`_site_grad`), the `TiledHamiltonian` constructor in `hamiltonian.jl`,
`minimize.jl`, the observables / binning measurement path, and the device
kernels in `src/gpu/`.

- **Zero allocations per sweep** is the standard; `bench/bench_sweeps.jl`
  reports allocs/sweep and nonzero is a red flag.
- `SVector{3,Float64}` spins; task-local `SweepScratch`; no shared mutable
  state across tasks; the fixed-order `_reduce_dE` — thread scaling must never
  buy a determinism break (serial ≡ parallel is bitwise).
- Device replicas are operation-order-faithful and libm-free: no `muladd`, no
  `@fastmath`, no reordered recursion — CPU ≡ GPU is a bitwise gate.
- **Bench bookkeeping**: when touching a hot path, run the matching
  `bench/bench_*.jl` before and after (`bench/README.md` explains how to
  localize a bottleneck) and append an entry to `.claude/bench_log.md` — even
  if numerical results are unchanged. GPU numbers come from the cluster.

## Managing development units

Mid-sized or larger work goes into **spec folders**. No cross-sprint progress
trackers.

- **Standing contracts**: the decision records `docs/specs/<topic>.md`
  (tiling, stationarity, binning, PT determinism, checkpoint schema, cell
  reduction, ground-state search, GPU). A change that moves a contract updates
  its record in the same commit.
- **Active development units**: `docs/specs/[YYMMDD]-[slug]/`, each with
  `requirements.md` / `design.md` / `tasklist.md`. Index at
  `docs/specs/README.md`; template at `docs/specs/_template/`.
- **Cross-cutting design notes, investigations, on-hold ideas**:
  `DESIGN_NOTES.md` (index) with bodies under `docs/design-notes/`.
- **Day-to-day TODOs**: `TaskCreate` (in-session only).
- **Historical benchmark records**: `.claude/bench_log.md` and `git log`.

### Spec-folder workflow

**Always create a spec folder and agree on it before starting mid-sized or
larger work.**

Entry criteria (any of these triggers a spec): multi-day effort; multiple
design choices (API, types, conventions, schema); a mid-sized or larger change
to existing behavior; future readers will ask "why was this done this way?".

Skip the spec for: bug fixes (covered by a regression test); documentation or
comment fixes; a small refactor within a single file; minor behavior tweaks
already covered by existing tests.

Procedure (Claude executes):
1. Create `docs/specs/[YYMMDD]-[slug]/` (`YYMMDD` = `date +%y%m%d`; `slug` is
   English kebab-case).
2. Copy the three files from `docs/specs/_template/` and fill them out with the
   user; run the `spec-reviewer` agent before presenting the draft.
3. Reach agreement on the spec before starting implementation.
4. Keep the folder after completion. Update the `Status:` line in
   `tasklist.md` and the table in `docs/specs/README.md` together.

## Working principles for Claude

### Free to proceed without asking

- Bug fixes (minimal change plus test), adding / fixing tests, documentation
  typos, notes in `DESIGN_NOTES.md` / `docs/design-notes/`, local commits of
  finished work units.

### Sub-agent usage

- After implementing, use the `test-runner` agent to run and diagnose tests.
- For performance investigation, use the `profiler` agent (CPU benches
  locally; GPU on the cluster via `hpc-runner`).
- For commits, hand off to `git-helper`. The main agent must not run
  `git commit -m` directly.
- To prepare a release, use `release-helper` (version decision, bump,
  CHANGELOG, the SCEFitting compat pin, `make test-ci` gate). It never
  commits, pushes, or tags.

### Code review: two tiers

**Tier 1 — `code-reviewer`.** A single generalist pass over the diff, for bug
fixes and small changes. Run it before committing.

**Tier 2 — the four-axis review panel.** After a spec-level feature lands:
`numerical-reviewer` (opus — physics, determinism contracts, coupled sites,
test oracles), `maintainability-reviewer`, `performance-reviewer`,
`api-reviewer` (incl. the dependent-package contract).

Panel procedure (the main agent orchestrates):
1. Launch all four reviewers **in one message** (parallel), same diff range.
2. Collect the reports (`blocker` / `major` / `minor`, optional
   `[contention: <axis>]`).
3. Apply every `numerical-reviewer` finding; apply the remaining blockers /
   majors unless contested.
4. Detect conflicts (same location, exclusive fixes; a contested tag the named
   axis actually contradicts).
5. Correctness always wins. A **material** performance vs maintainability
   tradeoff with no correctness angle goes to the user via `AskUserQuestion`.
6. Hand the user a single merged summary.

### Propose before implementing

- Algorithm changes (numerical results or acceptance statistics may change).
- Refactors that cross layer boundaries (tiling → kernels → algorithms →
  measurement → drivers / persistence) or the CPU ↔ GPU mirror.
- Performance improvements (present benchmark numbers first).
- Backports to / from SLCEMonteCarlo.jl.

### Always confirm — do not implement first

- Physics-convention changes (scale, units, summand counting, detailed
  balance, observable definitions).
- Loosening any bitwise determinism contract.
- New external dependencies (incl. GPU backends).
- Public-API signature changes (`export` / `public`, the
  SCESpinDynamics-facing names).
- Checkpoint schema or `_fingerprint` changes.
- `git push`, tags, releases, or any other remote operation.

## References

Consult as needed before working.

- `STYLE_GUIDE.md` — package-specific style deltas. **Always consult when
  editing code.**
- `SPEC.md` — architecture, primary types, public API.
- `docs/specs/README.md` — the decision records (standing contracts) and the
  spec folders; `DESIGN_NOTES.md` — design notes and investigations.
- `CHANGELOG.md` — what landed.
- `bench/README.md` — fixtures and the bottleneck procedure;
  `.claude/bench_log.md` — recorded campaigns (CPU and kugui GPU).
- `references/` — supporting literature (notes tracked, PDFs local-only).
- Published docs: <https://tomonori-tanaka.github.io/SCEMonteCarlo.jl/dev/>.
- `.claude/mcp-setup.md` — optional MCP servers (GitHub / Context7 / arXiv).

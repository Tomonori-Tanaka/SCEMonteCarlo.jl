# Decision record — ground-state search (`minimize_energy` / `find_ground_state`)

Status: landed (post-v0). Owner: `src/minimize.jl`; gates in
`test/unit/test_minimize.jl`.

## G1 — manifold formulation of the descent

The state space is the product of unit spheres `(S²)^n`. The descent works in
ambient coordinates with two manifold ingredients:

- **Gradient.** `_gradient!` computes the tangent-projected gradient of `+E` per
  site, `G_s = Σ_k c_k ∇Z_k(e_s)` with `grad_Zlm_unsafe` (already projected:
  `e_s·G_s = 0`), via the shared per-site kernel `_site_grad` in `energy.jl` —
  the same kernel behind the public all-site `energy_gradient!`. Per site it is
  the *identical arithmetic* of the public `site_gradient` — same `(l, m)` loop,
  same `ck == 0` skip — a coupled code site pinned by bitwise `==` gates
  (`test_minimize.jl`, `test_gradient.jl`), so the fast path can never drift
  from the public definition.
- **Retraction.** `R_x(−αG) = (x − αG)/|x − αG|`, per site. Division-safe by
  construction: `G ⊥ x` and `|x| = 1` give `|x − αG| = √(1 + α²|G|²) ≥ 1` — the
  denominator can never vanish, no special-casing needed (a zero-gradient site
  simply retracts to itself).

The Armijo model is *exact* on the manifold: the differential of `x ↦ x/|x|` at a
unit point is the tangent projector, and `G` is already tangent, so
`d/dα E(R(x − αG))|₀ = −⟨G, G⟩`. Hence `E_trial ≤ f_ref − σ·α·‖G‖²` is a genuine
sufficient-decrease test, not an ambient-space approximation.

Trial energies are computed by rebuilding the trial tesseral matrix and calling
`_total_energy` from scratch — never incrementally. Every step moves every site, so
per-site ΔE bookkeeping buys nothing, and the from-scratch evaluation keeps the
reported minimum drift-free by construction. On acceptance the current/trial
matrices are reference-swapped (no copy).

## G2 — BB1 step + GLL nonmonotone safeguard

- **Step seed.** Barzilai–Borwein BB1, `α = ⟨s,s⟩/⟨s,y⟩` with the ambient secant
  pair `s = x_k − x_{k−1}`, `y = G_k − G_{k−1}`. On the manifold this is a *scaling
  heuristic only* — correctness is owned entirely by the Armijo safeguard, so no
  vector transport is needed. Nonconvex curvature (`⟨s,y⟩ ≤ 0`) falls back to the
  largest admissible step.
- **Clamp.** `α ∈ [10⁻¹⁰, π] / gsup` where `gsup = max_s |G_s|`: the retraction
  angle is `atan(α|G_s|)`, so the cap bounds every per-site rotation by
  `atan(π) ≈ 1.26` rad — scale-free and unit-correct. First iteration (no secant
  history): `α₀ = 0.1/gsup` (max rotation `atan(0.1) ≈ 0.1` rad).
- **Safeguard.** Grippo–Lampariello–Lucidi nonmonotone Armijo: the reference is the
  max over the last `M = 10` *accepted* energies (`σ = 10⁻⁴`, step halving, ≤ 30
  backtracks). GLL induction gives `E_k ≤ E_0` for every accepted iterate — the
  window max never rises above the start — which is the monotone gate in the tests.
- **Stagnation.** Exhausted backtracking means the energy resolution floor (near
  machine precision the Armijo inequality can become unsatisfiable). The run
  returns the current iterate with `converged = false` — honest reporting, never a
  throw; same for exhausting `maxiter`.
- **Convergence.** `gsup ≤ gtol`, default `gtol = 10⁻⁸ × _site_energy_scale(H)` —
  measured on the suite fixtures the descent reaches this comfortably (the dimer
  converges in ~6 iterations to `|G| ~ 10⁻¹¹` against a scale-adjusted default of
  ~10⁻⁹). The scale drops the neighbor `|Z_lm|` and `|∇Z_lm|` factors (which exceed
  1 for `l ≥ 2`), so on high-`l` models the default flag can be optimistic — the
  docstring says to pass an explicit `gtol` where the stationarity certificate
  matters. The reported *energy* is exact either way.

## G3 — annealing defaults and the lean loop

- **Scale.** `_site_energy_scale(H)` = max over sites of
  `Σ_{adjacent instances} |coef|·Σ|folded|` — one deterministic CSR pass, a cheap
  overestimate of the per-site energy scale (the `(4π)^(body/2)` in `coef` roughly
  cancels the tesseral normalization). Overestimating only wastes a few hot sweeps
  (ladder) or loosens the default `gtol` proportionally; both are benign defaults
  that production work should override anyway.
- **Default ladder.** Geometric, 20 rungs over three decades below the scale
  (`scale → 10⁻³·scale`) — geometric because linear ladders collapse at the cold
  end (measured in the PT guide). An explicit ladder must be strictly decreasing
  (hot → cold); anything else is an `ArgumentError` — verified, never assumed. A
  single-rung ladder is allowed (fixed-temperature basin hopping-ish stirring).
- **Lean loop, not `UpdatePlan`.** The anneal worker calls `metropolis_sweep!` /
  `overrelaxation_sweep!` / `_adapt_step!` / `_renormalize!` directly: no
  observables, binning, or checkpointing are needed, and a dummy `UpdatePlan` would
  drag in irrelevant validations (`sweeps_measure ≥ 1`, `nbins ≥ 2`). Adaptation is
  never frozen — the whole anneal is thermalization-flavored, and the stationarity
  concerns of measurement do not apply (nothing is measured; the anneal only has to
  *end somewhere good*). `_renormalize!` at every rung boundary keeps the tracked
  energy exact where cycling decisions read it.

## G4 — thermal cycling (`cycles`, `reheat`)

A single anneal that freezes into a false basin stays there (measured: the PT
guide's freeze demo). Independent restarts (`nstarts`) fix this globally but
discard all found structure. **Thermal cycling** (Möbius et al., PRL **79**, 4297
(1997)) is the established middle ground: after the first full descent, re-enter
the ladder at rung `clamp(ceil(reheat·n_rungs), 1, n_rungs)` — *partial* re-heating
— and cool again; repeat `cycles` times. Re-heating to the top would be a plain
restart (that is `nstarts`' job); partial re-heating keeps the basin's information
while allowing barrier crossings. The best cold-end configuration across cycles
(compared on the exact post-renormalize energy) is the one polished. Measured on
the rugged fixture: with a deliberately short cold ladder, `cycles = 3` moves a
frozen `cycles = 1` run from the −4.328 basin to the −4.884 ground state at fixed
seed (gated).

Cycling consumes the same per-start RNG stream sequentially — determinism stays
confined to the start.

## G5 — determinism

Bit-identical results for a fixed seed regardless of `ntasks` / thread count, by
the same discipline as `run_pt`:

- per-start `Xoshiro`s are split from the master **in start order, before any
  spawning**; each stream is consumed only inside its own start (init draw →
  anneal); the polish consumes **no RNG at all** (grep `_minimize!` for
  `rand` — nothing);
- starts write disjoint result slots; no shared accumulators, no atomics — the
  only synchronization is the `@sync` barrier;
- all reductions (`gsup`, `‖G‖²`, secant sums, `_total_energy`) are serial loops,
  never pairwise-reassociated;
- the winner is `argmin(energies)` — Julia's first-minimum rule is the
  deterministic tie-break (lowest start index);
- with explicit `inits`, `_initial_config` consumes no RNG — harmless, since every
  start still owns a dedicated stream.

Gated: `ntasks = 1` vs `4` compare `==` on configs/energies/gradnorms/best, for
`cycles ∈ {1, 2}`; default seeds are fresh per call, recorded, and replayable.

## G6 — API rationale

- **No Optim.jl.** The package's culture is a self-contained deterministic core;
  the descent is ~120 lines against kernels that already existed. An external
  optimizer would move the bit-reproducibility contract into a dependency's line
  searches. (Sunny.jl's `minimize_energy!` wraps Optim.jl LBFGS over stereographic
  coordinates — a fine design; not this one.)
- **Names.** `minimize_energy` (no bang — nothing user-visible is mutated; a fresh
  result is returned, like `run_mc`) and `find_ground_state` (says what you get,
  not how). One shared `GroundStateResult`, with the per-start table in start order
  as a degeneracy/landscape diagnostic.
- **`converged = false`, never a throw.** Non-convergence is a *result*, not a
  usage error; the honest iterate is still the best available answer.
- **The strongest escape is PT, not more annealing knobs.** `run_pt` replicas
  random-walk the temperature ladder — the principled "temperature up-down". The
  recipe `find_ground_state(H; inits = pt.final_configs, anneal_sweeps = 0)`
  polishes every replica deterministically; the guide names it the production
  route for rugged landscapes. Population annealing and basin hopping are future
  candidates, mentioned in the guide, not implemented.
- **Heuristic, and documented as such.** No finite stochastic search certifies a
  global minimum; the per-start energy spread is the cheap self-diagnostic.

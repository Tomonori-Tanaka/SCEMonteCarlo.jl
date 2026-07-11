# Decision record — update schemes and their stationarity

Status: landed (M3–M4). Owner: `src/updates.jl`, `src/run.jl`;
gates in `test/unit/test_metropolis.jl`, `test/unit/test_overrelaxation.jl`.

## U1 — sequential site scan

Sites are updated in deterministic order `1:n_sites`. Each single-site kernel is
π-reversible (below); a composition of π-stationary kernels is π-stationary (the
composition itself is not reversible, which is irrelevant for sampling). Sequential
scan consumes no RNG for site selection and keeps runs bit-reproducible.

## U2 — Metropolis proposal and the RNG-consumption contract

The proposal is the symmetric two-component mixture proven in SCETools: antipodal
flip with probability 0.2 (inter-lobe ergodicity on bimodal single-site potentials)
+ Rodrigues rotation by `step·randn` about a uniform random axis (sign-symmetric
angle × uniform axis ⇒ symmetric). Acceptance `ΔE ≤ 0 || rand < exp(−βΔE)`, with
the uniform drawn **only when `ΔE > 0`** — the RNG-consumption contract every
kernel follows, so trajectories are a pure function of `(seed, schedule)`.
ΔE is exact for any body order (`ΔE = c_s·ΔZ`, `c_s` independent of `e_s`).

## U3 — adaptive step, thermalization only

`step ← clamp(step·exp((a − target)/2), 1e-3, π)` every `adapt_interval` sweeps on
a windowed acceptance `a`, **only during thermalization**. At the measurement
boundary the step freezes (`ChainState.frozen`). Why: a step that keeps responding
to chain history makes the transition kernel history-dependent — the chain is no
longer a fixed π-reversible kernel and measured expectations carry a finite-run
adaptation bias; freezing also keeps checkpoint resume bit-identical. The frozen
value is reported per temperature as `final_step`.

Fixture caveat baked into the gate: on a lattice with decoupled (free) sites the
acceptance has a floor (free sites always accept), so the adaptation target must be
tested on an all-coupled model.

## U4 — overrelaxation = deterministic involutive reflection + Metropolis accept

Proposal: `e′ = S(e) = 2(e·ĥ)ĥ − e`, with `ĥ` the direction of the local `l = 1`
field read off the leave-one-out coefficients (tesseral slots
`Z_{1,-1} ∝ y, Z_{1,0} ∝ z, Z_{1,1} ∝ x` ⇒ `h = (c₄, c₂, c₃)`). Accept with the
standard Metropolis rule on the exact ΔE.

**Stationarity.** `S` is a deterministic involution (`S∘S = id`) and an isometry of
the sphere (unit Jacobian), and its axis depends only on the *other* spins (`c_s`
is `e_s`-independent). For such proposals Metropolis acceptance satisfies detailed
balance pointwise: `π(e)·A(e→Se) = min(π(e), π(Se)) = π(Se)·A(Se→e)`. Hence each
site kernel is π-reversible and the sweep is π-stationary.

Two limits: (a) **pure `l = 1` site channel** — the reflection conserves `e·h`,
so `ΔE ≡ 0` and every move is accepted: exactly classical microcanonical
overrelaxation, with zero special-casing (machine gate; this also pins the tesseral
axis extraction — a wrong component order breaks `ΔE ≡ 0`); (b) **general SCE** —
the `l ≥ 2` / multi-body remainder is corrected exactly by the accept step.

A wrong axis is a *correctness no-op* (any `e`-independent axis + MH accept is
stationary) — it only costs acceptance/decorrelation efficiency.

**Not ergodic alone** (pure case conserves `e·h` per move): only ever mixed into
compound sweeps — 1 Metropolis sweep + `or_per_metropolis` OR sweeps. Sites with no
`l = 1` channel (precomputed `site_has_l1` mask) or vanishing field are skipped and
not counted as attempts.

**Efficiency reality check** (from the gate work): on a strongly anisotropic model
the l=1 reflection is nearly random w.r.t. the dominant `l ≥ 2` energy, and the OR
acceptance collapses at low temperature — OR pays off in exchange-dominated
(l=1-heavy) systems, which is its classical use case.

## U5 — energy-drift policy

Every `renorm_interval` compound sweeps and at each thermalization→measurement
boundary: renormalize all spins, rebuild the tesseral rows, recompute the total
energy, record `|E_incr − E_recomp|` into `max_drift` (reported per temperature),
warn once per run above `1e-8·max(1, |E|)`, and re-anchor. The schedule is
deterministic, so it does not interfere with bit-reproducible resume.

## U6 — metastability is the fixture's problem, not the sampler's

The random anisotropic two-site fixture freezes into seed-dependent basins below
`kT ≈ 0.15` (two *pure Metropolis* chains disagree far beyond error bars while each
reports small `τ_int` — the classic broken-ergodicity failure of within-basin error
bars). Statistical cross-checks between update schemes must run at temperatures
where the fixture demonstrably equilibrates (`kT = 0.5`). This is also the
motivation for parallel tempering (M5).

# Decision record — constrained Monte Carlo (CMC) for finite-temperature anisotropy

Status: in progress (spec `261121-constrained-mc`, agreed 2026-11-21). This
record is the standing contract for the CMC sampler; sections marked
*(pending)* are filled when the owning milestone lands. Owner: `src/cmc.jl`
(sampler, driver, observables), `src/observables.jl` (`total_torque`),
`src/checkpoint.jl` (kind `"cmc"`).

Primary source: P. Asselin et al., PRB 82, 054415 (2010), arXiv:1006.3507
(acceptance rule p. 7 step 7 and Eq. (A11); torque identity App. C, Eq. (C15)).
The weighted compensation follows Y. Toga et al., arXiv:1606.00333; the hybrid
(partly constrained) variant follows the VAMPIRE implementation used by
M. O. A. Ellis et al., arXiv:1706.05188.

## M1 — the constrained ensemble

Constrained sites `𝒞 = { s : site_active(s) and w_{a(s)} ≠ 0 }` with signed
per-cell-atom weights `w_a`; `M = Σ_{s∈𝒞} w_{a(s)} e_s`, `M_∥ = M·n̂`,
`M_⊥ = M − M_∥ n̂`. The sampled measure, exact at finite N, is

```
π(dξ) ∝ Π_s dΩ_s · exp(−βE(ξ)) · δ_{S²}(M̂(ξ), n̂),   δ_{S²}(M̂, n̂) = |M|² δ²(M_⊥) Θ(M_∥)
```

so that `F(n̂) = −kT ln P(n̂) + const`, `P` being the canonical density of `M̂`
per solid angle (Asselin's F). Inactive sites are excluded from `M`, from the
move set, and from every observable, and stay bitwise frozen. Signed weights
need no special case: the gauge map `ẽ_s = sign(w_s) e_s` sends the signed
problem exactly onto the `|w|` problem (compensation, sign preservation,
Jacobian, and proposal symmetry all commute with it).

Weights: `:auto` resolves to `H.magmoms` when present, else all ones;
`:magmoms` requires moments; an explicit vector is indexed by the cell atoms of
`H` (the reduced cell after `reduce_cell`). `magmoms ≥ 0` is enforced
(`zeeman-field.md` Z1), so signed weights only come from an explicit vector.

## M2 — the pair move *(pending M3/M4)*

For a source `i ∈ 𝒞` and partner `j`: rotate `e_i` by the symmetric Rodrigues
proposal (no flips in CMC); compensate
`e_j⊥′ = e_j⊥ − (w_i/w_j)(e_i⊥′ − e_i⊥)`; null move unless `|e_j⊥′|² < 1`;
`e_j∥′ = σ_j √(1 − |e_j⊥′|²)` with `σ_j = (e_j∥ ≥ 0 ? +1 : −1)`; null move
unless `M_∥′ > 0`; accept with

```
ℓ = −βΔE + 2 ln(M_∥′/M_∥) + ln|e_j∥| − ln|e_j∥′|
```

`ΔE` is the exact joint difference by sequential leave-one-out (the single-site
contract of `hamiltonian-tiling.md` applied twice; any body order).

**Null moves count as attempts.** Re-drawing after a geometric failure samples
`π(x)(1 − r(x))` instead of `π` (measured 8–13σ bias at N = 3 in the spec's
investigation).

## M3 — partner rule, sweep, RNG *(pending M4)*

Partner uniform over `𝒞 \ {i}` (state-independent; a same-weight-class rule is
rejected because it conserves each class's transverse sum and breaks
ergodicity). One CMC sweep visits the active sites in site order: a pair
attempt for `s ∈ 𝒞`, a plain single-site Metropolis attempt for active
`w = 0` sites (hybrid CMC). All randomness of an attempt comes from
`site_rngs[s]` (partner index → rotation axis → rotation angle → accept
uniform only when `ℓ < 0`). Unlike U1 of `updates-stationarity.md`, the CMC
sweep consumes RNG for partner selection and is serial only — the global
`M_∥` couples every pair, so color classes do not commute.

## M4 — determinism *(pending M5/M7)*

Single chain: serial, bitwise reproducible for a given seed within the scope of
`pt-threads-determinism.md` P6. Direction lanes: lane seeds are drawn in order
`rand(Xoshiro(seed), UInt64)` for `k = 1, 2, …`; results are bitwise
independent of `ntasks`. Re-initialization (`carryover = false`) is
deterministic and consumes no RNG.

## M5 — torque conventions *(pending M2)*

`total_torque(H, config) = Σ_{s active} G_s × e_s = −Σ e_s × ∂E/∂e_s` (full
system; `G` from `energy_gradient!`, Zeeman included; never weighted). Identity,
exact at finite N: `⟨T⟩ = −n̂ × ∂F/∂n̂`, i.e. `⟨T⟩·ê_φ = −∂F/∂θ`,
`⟨T⟩·ê_θ = (1/sinθ) ∂F/∂φ`, `⟨T⟩·n̂ = 0`, with
`ê_φ = (−sinφ, cosφ, 0)`, `ê_θ = (cosθ cosφ, cosθ sinφ, −sinθ)`. Observables
report per training cell (`÷ n_cells`, as `:M`). Partial torques of
term-subset Hamiltonians sum to the total (plus the analytic Zeeman part);
an individual partial torque is a free-energy gradient only to O(K).

## M6 — checkpoint *(pending M6)*

Kind `"cmc"`, additive under schema v3; `constraint/` group written by
`_write_header`; see `checkpoint-schema.md`.

## M7 — out of scope for v1

Constrained PT, block constraints, per-site weights, |M| constraints,
overrelaxation in CMC, colour-parallel CMC sweeps, GPU, constrained
`minimize_energy`, post-processing fits (K(T), κ_lm).

## Mutation record *(filled per milestone)*

| Gate | Mutation | Observed shift | σ | Result |
|---|---|---|---|---|

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

## D1 — the constrained ensemble

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

**Constraint, resolution, and initialization (landed M3, `src/cmc.jl`).**
`MagnetizationConstraint(direction; weights)` / `(θ, φ; weights)` normalizes the
direction (`n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`; zero, non-finite, or non-finite
angles are refused) and validates the weights (`:auto | :uniform | :magmoms` or
a vector of finite reals, copied); signed zeros are folded to `+0.0` so that
`==`, `isequal`, `hash` (field-wise) agree. Resolution re-checks finiteness.
Resolution against `H` (`_resolve_constraint`) refuses `:magmoms` without
moments, a weight vector of the wrong length, and `|𝒞| < 2`; it stores the
per-site weights with `site_w[s] = 0` off `𝒞` (inactive sites included).
The start of a constrained chain is deterministic and consumes no RNG:
the default pattern is `e_s = sign(w_s) n̂` on `𝒞` and `n̂` on every other site;
a given `init` is rotated rigidly (every site) by the minimal rotation taking its
`M̂` to `n̂`, refused when `|M| < 1e-8 Σ_{s∈𝒞}|w_s|`. When `M̂` is within 1e-8 of
`−n̂` the minimal rotation is ill-conditioned (its plane is rounding noise), so
the rotation is the half turn about `normalize(n̂ × x̂)` (`ŷ` when `n̂ ∥ x̂`)
followed by the residual minimal rotation — the half turn alone (to rounding)
when `M̂ = −n̂`. Rotated spins are re-normalized and clamped into `[-1, 1]`.

## D2 — the pair move *(proposal landed M3; the attempt and the sweep pending M4)*

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

The proposal is the pure function `_cmc_pair_proposal` (M3): it returns
`(e_j′, ΔE, ℓ)`, or `(e_j, NaN, −Inf)` for a null move, without touching the
state. Both null tests are written NaN-safe (`!(s2 < 1)`, `!(M_∥′ > 0)`), so a
NaN anywhere in the inputs is a null move. A current state with `M_∥ ≤ 0` (or
NaN) is a caller error and throws, rather than producing a `log` of a negative
ratio. A partner with `e_j·n̂ = 0` exactly gives `ℓ = −Inf` with a finite `ΔE`
(always rejected; the reverse move has zero density).

`e_j′` is re-normalized and clamped into `[-1, 1]`. The constructed
`e_j⊥′ + e_j∥′ n̂` is unit only to rounding, and its components overshoot 1 in a
measurable fraction of near-axis proposals (≈ 13 % of the UN-normalized
constructions in the M3 test that aims at `e_j′ = x̂`). The division by the
computed norm alone already brings every component into `[-1, 1]` (with
correctly rounded arithmetic the computed norm is ≥ every |component| and the
rounded quotient is monotone); the clamp is kept as a free guard. The bound
matters twice: the tesseral harmonics throw a `DomainError` for `|e·ẑ| > 1`
(the Legendre recursion `dnPl` reads only `u[3]`), and a checkpoint restore
refuses any component outside `[-1, 1]` (`_validate_unit_column`). The change is
deterministic and rounding-sized, so it cannot bias the measure at any
statistical resolution; `M_∥′` and the Jacobian use `e_j′·n̂` of the
re-normalized vector, so `ℓ` is the log ratio of the move actually made. A sign
flip of `e_j′·n̂` against `σ_j` is a null move (a guard: not reachable while
`s2 < 1`, since `|e_j∥′| ≥ ~1e-8` dwarfs any rounding).

**Null moves count as attempts.** Re-drawing after a geometric failure samples
`π(x)(1 − r(x))` instead of `π` (measured 8–13σ bias at N = 3 in the spec's
investigation).

## D3 — partner rule, sweep, RNG *(pending M4)*

Partner uniform over `𝒞 \ {i}` (state-independent; a same-weight-class rule is
rejected because it conserves each class's transverse sum and breaks
ergodicity). One CMC sweep visits the active sites in site order: a pair
attempt for `s ∈ 𝒞`, a plain single-site Metropolis attempt for active
`w = 0` sites (hybrid CMC). All randomness of an attempt comes from
`site_rngs[s]` (partner index → rotation axis → rotation angle → accept
uniform only when `ℓ < 0`). Unlike U1 of `updates-stationarity.md`, the CMC
sweep consumes RNG for partner selection and is serial only — the global
`M_∥` couples every pair, so color classes do not commute.

## D4 — determinism *(pending M5/M7)*

Single chain: serial, bitwise reproducible for a given seed within the scope of
`pt-threads-determinism.md` P6. Direction lanes: lane seeds are drawn in order
`rand(Xoshiro(seed), UInt64)` for `k = 1, 2, …`; results are bitwise
independent of `ntasks`. Re-initialization (`carryover = false`) is
deterministic and consumes no RNG.

## D5 — torque conventions

`total_torque(H, config) = Σ_{s active} G_s × e_s = −Σ e_s × ∂E/∂e_s` (full
system; `G` from `energy_gradient!`, Zeeman included; never weighted). Identity,
exact at finite N: `⟨T⟩ = −n̂ × ∂F/∂n̂`, i.e. `⟨T⟩·ê_φ = −∂F/∂θ`,
`⟨T⟩·ê_θ = (1/sinθ) ∂F/∂φ`, `⟨T⟩·n̂ = 0`, with
`ê_φ = (−sinφ, cosφ, 0)`, `ê_θ = (cosθ cosφ, cosθ sinφ, −sinθ)`. Observables
report per training cell (`÷ n_cells`, as `:M`). Partial torques of
term-subset Hamiltonians sum to the total (plus the analytic Zeeman part);
an individual partial torque is a free-energy gradient only to O(K).

The analytic Zeeman torque is `T_Z = +MU_B_EV_T (M_μ × B)`, `M_μ = Σ_{s active}
m_{a(s)} e_s` (from `E_Z = −MU_B_EV_T Σ m (e·B)` and `T·w = −dE/dα`: a moment is
turned toward `+B`). `torque_observable(name, Hpart, H)` admits a partial
Hamiltonian only if it matches `H` in `n_cell_atoms`, `dims`, and `n_sites`, carries
no field, and every template equals (atoms, shifts, ls, folded, scaled coef) a
fitted template of `H`; the observable is bound to that `H` object and refuses a
run on any other.

## D6 — checkpoint *(pending M6)*

Kind `"cmc"`, additive under schema v3; `constraint/` group written by
`_write_header`; see `checkpoint-schema.md`.

## D7 — out of scope for v1

Constrained PT, block constraints, per-site weights, |M| constraints,
overrelaxation in CMC, colour-parallel CMC sweeps, GPU, constrained
`minimize_energy`, post-processing fits (K(T), κ_lm).

## Mutation record *(filled per milestone)*

| Gate | Mutation | Observed shift | σ | Result |
|---|---|---|---|---|
| G2 | naive ΔE (`c_j` from the unmodified rows) | per sharing case, max \|naive − exact\| > 1e-5 (asserted lower bound); smallest measured 0.059 (F4 end pair (1,4)) | — (deterministic) | killed: `ΔE ≈ exact` and `worst < 1e-13` on all sharing cases; invisible on non-sharing pairs, as designed |
| G2 | column `i` not restored | — | — | killed: bitwise `zrows` restore on every case |
| G2 | both columns written tentatively before `ΔE_i` | O(ΔE) | — | killed: `ΔE ≈ exact` on every sharing case |
| G3 | torque sign (`e × G`) | O(T) | — | killed: rotated-energy difference, Zeeman share, G17 sum |
| G3 | wrong axis (cyclic components) | O(T) | — | killed: same assertions |
| G3 | Zeeman double-counted in `total_torque` | O(T_Z) ≈ 1e-3 eV | — | killed: rotated-energy difference on FZ, Zeeman share |
| G3/G17 | analytic Zeeman torque zero / sign flipped | O(T_Z) | — | killed: hand-written `μ_B Σ m e × B`, G17 sum on FZ |
| G17 | partial torque on a site-shifted configuration | O(T_part) | — | killed: G17 sum and per-part rotated-energy differences |
| G17 | door without the template match | — | — | killed: relabelled-atom, perturbed-coefficient, and foreign-term `Hpart` rejections |
| G17 | partial torque not divided by `n_cells` | factor 2 (dims = (2,1,1)) | — | killed: G17 sum and per-part differences |
| G2 | partner row tabulated inside the tentative-write window | — | — | killed: `zrows` bitwise intact after a harmonics `DomainError` (only that assertion; values unchanged) |
| G2 | an allocation inside the window | — | — | killed: `@allocated == 0` on F3a, F4, FZ |
| G3 | torque sign / wrong axis (closed forms) | O(T) | — | also killed by the uniaxial `T·ŷ = −N K sin 2θ` and in-plane `T·ê_θ = N K sin 2φ` checks |
| G17 | template key without `shifts` | — | — | killed: shifts-only `Hpart` rejection |
| G17 | no field door | — | — | killed: field-on-`Hpart` rejection with the field-door message |
| G17 | observable not bound to its `H` | — | — | killed: measuring on an equal but distinct `H` throws |
| G1 | compensation ratio `\|w_i/w_j\|` | O(1) partner error on mixed-sign pairs | — (deterministic) | killed: G1 `\|M⊥\| ≤ 1e-12 Σ\|w\|` on Fmix and F2w− (F2w+ is blind, as designed); G4 tallies `ref`, `null_s2`, `finite` |
| G1 | lab-frame compensation (`ẑ` in place of `n̂`) | O(1) | — | killed: G1 `\|M⊥\|` on all three fixtures; G4 tallies `ref`, `sign`, `rev_*`, null classification; the aimed-at-`x̂`/`ẑ` targets |
| G1/G4 | `j == i` admitted | — | — | killed: `AssertionError` for `i = j` on null-geometry inputs (the non-null case is caught by `_pair_delta_energy!` regardless) |
| G1/G4 | zero-weight partner assert removed | — | — | killed: the Fmix zero-weight partner must raise `AssertionError` |
| G4 | Jacobian exponent 2 → 1 | `ln(M_∥′/M_∥)` per move (> 1e-3 on ≥ 500 triples) | — | killed: tally `ℓ` (independent formula) on all five fixtures; reversibility passes, as expected |
| G4 | `\|z\|` ratio missing / inverted | `ln\|e_j∥/e_j∥′\|` per move | — | killed: tally `ℓ` on all five fixtures, and the `e_j∥ = 0 ⇒ ℓ = −Inf` case |
| G4 | `σ_j` forced to +1 | O(1) partner error when `e_j∥ < 0` | — | killed: tallies `ref`, `sign`, `rev_e`, `rev_ℓ`, `rev_ΔE`, `finite`, `null_M` |
| G4 | unweighted `M_∥′` update in the proposal | O(1) on weighted fixtures | — | killed: tallies `ℓ`, `null_M`, `rev_*` on F2w± and Fmix; invisible on uniform weights, as designed |
| G4 | `M_∥′ > 0` null test removed | — | — | killed: `DomainError` (`log` of a negative ratio) and the null-move counts |
| G4 | NaN-unsafe `s2 ≥ 1` alone / `M_∥′ ≤ 0` alone | none | — | **survives — equivalent mutants**: a NaN that reaches one test reaches the other (`s2` NaN ⇒ `e_j′` NaN ⇒ `M_∥′` NaN; `M_∥′` NaN only via a NaN `s2`), so each guard alone already makes every NaN input a null move. Both kept as defense in depth |
| G4 | NaN-unsafe, both tests | — | — | killed: a NaN proposal must return `(e_j, NaN, −Inf)` |
| G4 | sign guard `(e_j′·n̂ ≥ 0) == (σ_j > 0)` removed | none | — | **survives — guard on an unreachable region**: with `s2 < 1`, `\|e_j∥′\| ≥ ~1e-8` dwarfs any rounding, so no input flips the sign |
| G4 | `e_j′` not re-normalized | `e_j′·ẑ` one ulp above 1 | — | killed: the proposal aimed at `e_j′ = ẑ` throws `DomainError` in the harmonics; tally `domain` on the aimed-at-`x̂` proposals |
| G4 | the clamp in `_unit_clamped` removed (division only) | none | — | **survives — equivalent**: with correctly rounded arithmetic the division by the computed norm already keeps every component in `[-1, 1]`; the clamp is a guard |
| G0 | `sign(w)` dropped from the default pattern | `M̂` unchanged (Σw n̂ still ∥ n̂) | — | killed: the per-site default pattern and `M_∥ = Σ\|w\|` on Fmix |
| G0 | unweighted `M` in `_cmc_magnetization` | O(1) misrotation | — | killed: tallies `Mhat`, `axis_fixed`, `plane`; the weighted `\|M_init\|` door |
| G0 | inactive site admitted to `𝒞` | — | — | killed: Fmix `cset`/`site_w`, the `\|𝒞\| < 2` door with the only partner inactive, the antiparallel starts, tally `Mhat` |
| G0 | `\|M_init\|` door removed | — | — | killed: `[v, −v]`, the 1e-9-tilted pair, and the F3 (2, 1, 1) case at `\|M\| = 3.5e-8` are refused |
| G0 | door scaled by `\|𝒞\|` instead of `Σ\|w\|` | threshold 3e-8 instead of 4e-8 | — | killed: F3 weights (2, 1, 1) with `\|M\| = 3.5e-8` must throw |
| G0 | door scaled by `Σ w²` | threshold 6e-8 instead of 4e-8 | — | killed: the same fixture with `\|M\| = 4.5e-8` must pass |
| G0 | half turn for every obtuse start (non-minimal) | O(1) | — | killed: tallies `axis_fixed`, `plane` (the axis `M̂₀ × n̂` is fixed) |
| G0 | near-antiparallel branch removed | `M̂ = −n̂` exactly is left unrotated | — | killed: the exact antiparallel starts (`n̂ = ẑ`, `n̂ = x̂`); tilts ≥ 1e-12 rad are handled by the plane formula alone |
| G0 | signed-zero fold of the direction / of the weights removed (inner constructor; file-based mutation, restored and `cmp`-checked) | — | — | killed: `MagnetizationConstraint(0.0, 2.0; weights = [-0.0, 1])` vs `([0, 0, 1]; weights = [0.0, 1])` must agree in `==`, `isequal`, `hash` |
| G0 | resolve-time finiteness re-check removed | — | — | killed: a weight vector mutated to NaN after construction must be refused |

All other M3 mutants were applied in memory (function definitions re-evaluated
into the loaded module, source untouched, `shasum` identical before and after).

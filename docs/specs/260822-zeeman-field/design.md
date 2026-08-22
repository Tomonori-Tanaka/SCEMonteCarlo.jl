# Design: Zeeman term — constant site moments in a uniform field

Status: draft v2 (2026-08-22) — body-1 cluster-term representation.

## Summary

With `SCEFitting.Harmonics.N1 = √(3/4π)` and the `lm_index` order, the
`l = 1` tesseral row of a unit vector is `Z_{1,−1} = N1·y`, `Z_{1,0} = N1·z`,
`Z_{1,1} = N1·x` (rows 2, 3, 4). Hence

```
−μ_B m_a (e · B) = Σ_m hz_a[m] · Z_{1,m}(e),
hz_a = −(MU_B_EV_T · m_a / N1) · (B_y, B_z, B_x)      (rows 2, 3, 4)
```

is a **body-1 cluster term** on atom `a`. The constructor appends, for every
cell atom with `m_a ≠ 0`, one

```julia
ScaledTerm(1.0, [a], [SVector(0, 0, 0)], [1], hz_a)   # coef, atoms, shifts, ls, folded
```

to `H.terms` after the fitted terms have been scaled. Everything downstream is
body-generic and reads `H.terms` / the programs built from it, so the term
flows through `site_coeffs!` (general path, zero factors, `p = 1.0`,
`c[tgt] += w·1.0 ≡ c[tgt] += w` bitwise), `_total_energy`, both reference
kernels, `delta_energy`, the Metropolis accept, the overrelaxation axis
`(c[4], c[2], c[3])` (now the total local field including `B`),
`_site_grad` / `energy_gradient!` / the descent's `_gradient!`,
`_site_energy_scale` (default `gtol` / annealing ladder), `_renormalize!`,
PT swap energies, `_fingerprint`, the CSR adjacency (`site_active`,
`site_has_l1`), `_color_sites` (an isolated vertex gets colour 1; a site that
also has fitted instances keeps its colour bitwise), and the device tables /
`_entry_walk_partial` / `_entry_walk_grad` (general branch with an empty
factor range — the same functions the keyed / lane references call).

`coef = 1.0` and `folded = hz_a` make the programs' `sent_w = 1.0·hz === hz`
exact. Routing the term through a raw `MultipoleTerm` with `coef = 1/√(4π)`
and letting the ctor apply `(4π)^{1/2}` is **not** allowed — it is 1 ulp off.

Two pieces of state remain explicit because the terms cannot carry them: the
moments and the field themselves (for `:M` / `:M_B`, `zeeman_energy`,
`has_field`, the fingerprint disambiguation, and the checkpoint record), and
the fitted / synthetic boundary `n_fitted_terms`.

## Module layout

| Target | Change |
|---|---|
| `src/units.jl` | `MU_B_EV_T` (CODATA 2018 `9.2740100783e-24 / 1.602176634e-19` eV/T — the value and name of SLCEDynamics' constant), exported next to `KB_EV`; docstring states the eV-fitted-model assumption |
| `src/hamiltonian.jl` | ctor keywords `magmoms`, `field`; door validation; `_zeeman_terms(magmoms, B)::Vector{ScaledTerm}` (the single generator — the step-(b) extension point); terms appended after scaling; `lmax = max(lmax, 1)` whenever a Zeeman term is appended (**mandatory** — `lmax` is otherwise computed from the fitted terms only and an all-`l = 0` list would make `site_coeffs!` write `c[2:4]` out of bounds under `@inbounds`); new fields `magmoms::Union{Nothing,Vector{Float64}}`, `field::SVector{3,Float64}`, `n_fitted_terms::Int`; `has_field(H)`, `zeeman_energy(H, config)`; `Base.show` prints "`N` fitted + `M` zeeman terms" and the field |
| `src/reduce.jl` | `TiledHamiltonian(red::ReducedCell; dims, magmoms, field)` forwards the keywords (length = reduced-cell atom count) |
| `src/observables.jl` | `:M` (3, μ_B per training cell, over `site_active` — the `_sublattice_m` pattern) when `magmoms !== nothing`; `:M_B` when `has_field(H)`; `standard_observables(H)` appends them |
| `src/checkpoint.jl` | `_fingerprint`: after the term loop, `magmoms !== nothing` ⇒ mix a tag, every `m_a`, the three field components; writer adds `zeeman/{magmoms, field}` when present (informational, additive) |
| `src/energy.jl` | **no code change**; `energy_gradient!` docstring: the Zeeman part is in `G` when `has_field(H)`; names SLCEDynamics' `gzee` as the double-counting hazard |
| `src/minimize.jl`, `src/updates.jl`, `src/state.jl`, `src/pt.jl`, `src/gpu/*.jl` | **no code change** (verified path by path — see Impact) |
| `test/unit/test_zeeman.jl` (new — also holds the field variants of the inactive / parallel / PT / checkpoint gates), `test_units.jl` (`MU_B_EV_T` value), `test_gpu.jl` (body-1 fixtures in the G3 / G7 / direct-ΔE gates, device Langevin, GPU-PT resume), `fixtures.jl` (`_assert_same_result` shared) | tests |
| `docs/specs/zeeman-field.md` | new decision record Z1–Z5 (below) |
| `docs/specs/{hamiltonian-tiling,checkpoint-schema,binning-observables}.md` | T4 / scale-once exception, fingerprint scope, `:M` / `:M_B` definitions |
| `docs/src/guide/field.md` (+ `make.jl`), `observables.md`, `running.md`, `ground_states.md`, `gpu.md`, `api.md`, `SPEC.md`, `CHANGELOG.md`, `CLAUDE.md` coupled sites, `.claude/agents/` | documentation and scaffolding sweep |

## API

```julia
const MU_B_EV_T = 9.2740100783e-24 / 1.602176634e-19   # eV/T, exported

# All three constructor forms accept the same two keywords
TiledHamiltonian(model::SCEPredictor; dims = (1, 1, 1),
                 magmoms::Union{Nothing,AbstractVector{<:Real}} = nothing,   # μ_B per cell atom
                 field::Union{Nothing,AbstractVector{<:Real},NTuple{3,Real}} = nothing)  # tesla
TiledHamiltonian(n_cell_atoms, terms::Vector{MultipoleTerm}; dims, magmoms, field)
TiledHamiltonian(red::ReducedCell; dims, magmoms, field)

# Public, unexported
has_field(H::TiledHamiltonian)::Bool                    # magmoms given and B ≠ 0
zeeman_energy(H::TiledHamiltonian, config::SpinConfig)::Float64
                                                        # −MU_B_EV_T Σ_s m_{a(s)} e_s·B; 0.0 otherwise
# Documented struct fields: H.magmoms, H.field, H.n_fitted_terms
```

Door validation: `length(magmoms) == n_cell_atoms`, every `m_a` finite and
`≥ 0` (a negative magnitude is a direction flip `e_s` already carries);
`field` finite, length 3; `field !== nothing` **requires** `magmoms !==
nothing` (a field without magnitudes is an error, not a silent zero).
`magmoms` alone is allowed: no term is appended, `:M` becomes available, the
fingerprint records the moments.

Step (b) will add a second keyword (`moment_model::MomentModel`), validated
XOR against `magmoms` in the `temperature` xor `kT` style, feeding the same
`_zeeman_terms` generator. A union-typed `moments` keyword was rejected: it
fights the `SpinDatum.magmoms` / SLCEDynamics `magmom` naming and forecloses
the XOR door.

Observables appended by `standard_observables(H)`:

- `:M` — `Σ_{s active} m_{a(s)} e_s / n_cells` (3 components, μ_B per
  training cell — the `_sublattice_m` normalization, not `:m`'s per-active-site
  one). A frozen direction — `m_a = 0`, or a moment-carrying SCE-inactive
  sublattice without a field — can never leak in (B3 of
  `binning-observables.md`); with a field every `m_a ≠ 0` site owns an
  instance and is counted.
- `:M_B` — `M · B̂` (scalar) when `has_field(H)`.

## Types and conventions

- **Sign.** The moment of site `s` is `m_{a(s)} μ_B e_s`, so `E_Z = −μ_s · B`:
  a ferromagnet in `B ∥ ẑ` aligns with `+ẑ`. Cross-check: the OR axis
  `h = (c[4], c[2], c[3])` gains `−(μ_B m_a/N1)·B`, i.e. the total field with
  the right sign.
- **Units.** `B` in tesla, `m_a` in μ_B, `E_Z` in eV through `MU_B_EV_T`; the
  model is assumed eV-fitted exactly as `KB_EV` assumes for `temperature`.
  A model in other units must pass `field` pre-scaled (documented).
- **Scale-once exception (T4 note).** The `(4π)^(body/2)` rule applies to the
  fitted terms; the Zeeman terms are constructed in consumer form and bypass
  it. `N1` enters only as the tesseral ↔ Cartesian conversion and is read from
  `SCEFitting.Harmonics.N1` (one definition, shared with the OR axis).
- **Append rule (Z3).** The ctor appends one Zeeman term per atom with
  `hz_a ≠ 0`, i.e. `m_a ≠ 0` **and** `B ≠ 0`. With `magmoms` alone (or
  `B = 0`) no term is appended and the Hamiltonian's energy, adjacency,
  activation, colouring, and every pre-existing observable are bitwise those
  of the field-free model — only `H.magmoms`, the fingerprint input, and the
  `:M` observable are added. This keeps the activation rule exactly "has an
  adjacent instance" and leaves `binning-observables.md` B3 intact: an
  SCE-inactive, moment-carrying sublattice stays frozen and excluded from
  `:m` / `:sublattice_m` / `:M` without a field (a frozen direction is not a
  moment; the "free rotor" alternative of appending zero-`folded` terms was
  rejected because it silently dilutes the B3 observables the moment
  `magmoms` is supplied). With a field such a sublattice owns a real
  instance, is active, and enters `:m` / `:sublattice_m` / `:M` — a genuine
  field-induced polarization, documented in B3 as the one way a non-SCE site
  becomes magnetic. `m_a = 0` atoms never get a term.
- **`H.terms` semantics (public `ScaledTerm` tier).** `H.terms[1:n_fitted_terms]`
  are the scaled fitted SALC terms in `multipole_terms` order;
  `H.terms[n_fitted_terms+1:end]` are the synthetic Zeeman terms, one per
  moment-carrying atom in atom order. `SPEC.md` states this; `Base.show`
  reports the split. Tests that mean "fitted terms" slice accordingly.
- **Fingerprint (Z4).** The synthetic terms are already mixed through the
  term loop, but they see only the product `m_a·B`: without a field no term
  exists at all (`magmoms` alone would collide with field-free), and
  `(m, B)` / `(m/2, 2B)` emit identical terms (same energy, different `:M`).
  The explicit `magmoms` / `field` mixing (only when `magmoms !== nothing`)
  removes both collisions. Field-free Hamiltonians mix nothing new. *(As
  landed: true through `30f1797`; the follow-up schema-v3 mixer fix —
  `checkpoint-schema.md` C3, 2026-08-22 — then changed every fingerprint on
  purpose, superseding the byte-identity requirement.)*
- **Checkpoint.** The Zeeman group needs no schema bump *(the v3 bump is the
  mixer fix, not this group)*. When `H.magmoms !== nothing` the writer
  adds `zeeman/magmoms` (`Vector{Float64}`) and `zeeman/field`
  (`Vector{Float64}(3)`) — informational; the reader verifies through the
  fingerprint only (plain-data rule C1, additive like the `gpu_pt` kind).
- **No new chain-state field.** `ChainState` / `GPUChainState` are untouched;
  the `_swap_payload!` partition tests need no change.
- **`energy_gradient!` semantics.** `G[s]` includes `−μ_B m_a (B − (e_s·B)
  e_s)` when `has_field(H)`. The docstring changes from "moment magnitudes are
  the caller's" to "the Zeeman part is in `G` when `has_field(H)`; a dependent
  that applies its own field (SLCEDynamics' `gzee` / `b_ext`) must assert
  `!has_field(H)`". The effective-field formula `B_s = −G[s]/(m_s μ_B)`
  stays.
- **Overrelaxation with a field.** The reflection is about the total local
  field; exact for pure-`l = 1` models (Zeeman is `l = 1`). On a Zeeman-only
  site the reflection about `B̂` preserves `e·B̂`, so OR there is accepted
  work that moves nothing along the field — no bias (OR is not ergodic by
  design), documented. Once a field is set every moment-carrying site has
  `site_has_l1`, so the `or_per_metropolis` "no `l = 1` channel" door no
  longer fires for such models — correct, and documented as a behaviour
  change.
- **`_site_energy_scale` value.** A Zeeman-only sublattice contributes
  `|coef|·sum(abs, folded) = MU_B_EV_T·m_a·‖B‖₁/N1 ≈ 2.05·MU_B_EV_T·m_a·‖B‖₁`
  — an over-estimate of the true half-range `μ_B m_a ‖B‖₂`, which is the
  function's documented role; no change.

## Impact on coupled sites

- [x] `hamiltonian.jl` ↔ SCEFitting introspection contract: `MultipoleTerm`
      consumption unchanged; new dependency on `SCEFitting.Harmonics.N1` —
      added to the "Dependency boundary" of `SPEC.md`. New contract on the
      public `ScaledTerm` tier (`H.terms` split) — `SPEC.md`.
- [x] `energy.jl` 4-function contract ↔ `updates.jl`: no code change; the
      gates "ΔE ≡ total-energy difference", "own-spin independence",
      "program ≡ reference (bitwise)" re-run with a field. The body-1 general
      path is already under the bitwise gate (`test_energy.jl`, `l = 2`
      body-1 term).
- [x] `lm_index` ↔ `zlm_row!` ↔ OR axis: `hz_a` is written in `lm_index`
      order (rows 2, 3, 4 = y, z, x) — the mapping the OR axis reads; an
      upstream reorder now also breaks the Zeeman identity gate.
- [x] `reduce.jl` ↔ tiling ↔ `geometry.jl`: keyword forwarding only;
      `reduce_cell` consumes `MultipoleTerm`s (pre-ctor), never `H.terms`.
- [x] Gradient kernels (`_site_grad` / `site_gradient` / `energy_gradient!` /
      `minimize.jl`): read `c` → automatic; `_site_energy_scale` reads
      `H.terms` → automatic (value above); bitwise `==` gates among the three
      stay valid; FD gate re-run with a field; ground-state gate with
      **default** `gtol` / ladder added.
- [x] Checkpoint writer ↔ reader ↔ `checkpoint-schema.md`: explicit
      conditional mixing + additive group; schema v2; record updated; the
      field-free pin guards the unconditional part.
- [x] Colouring ↔ sweeps ↔ `updates-stationarity.md`: a body-1 instance is
      never shared, so fitted-model colours are bitwise unchanged and
      moment-only sites are isolated vertices (colour 1); U1 unchanged;
      `test_parallel.jl` re-run with a field.
- [x] Device row / kernel / gradient ↔ host references: no code change; the
      general branch with an empty factor range is currently **untested** on
      the device path — G3 / G7 bitwise gates gain a body-1 fixture (both a
      Zeeman-only site and a site with fitted + Zeeman instances); an
      all-body-1 model yields zero-length `sfac_row` / `sfac_slot` device
      arrays (legal; covered by the same fixture on the CPU backend, noted for
      the CUDA validation).
- [x] CPU PT ↔ GPU PT: all lanes share one `H`; `_swap_accepts` is
      temperature-only and unchanged; no new chain-state field, so the
      `_swap_payload!` partition tests in `test_pt.jl` / `test_gpu_pt.jl` are
      untouched — both files gain a field-carrying run.
- [x] Inactive-site convention: `m_a = 0`, or no field ⇒ no term ⇒ today's
      behaviour bitwise, including `:m` / `:sublattice_m` / `n_active`
      (`test_inactive.jl` gains a field variant and a `magmoms`-only
      variant); `m_a ≠ 0` with a field ⇒ active by the existing "has an
      adjacent instance" rule — no new rule. B3 of `binning-observables.md`
      gains the sentence that a field is the one way a non-SCE site becomes
      magnetic (and then counts in `:m`).
- [x] `.claude/agents/`: the coupled-sites list in `code-reviewer` /
      `numerical-reviewer` / `spec-reviewer` gains the Zeeman bullet
      (append rule, `H.terms` split, `lmax` bump); the test map gains
      `test_zeeman.jl`.
- [x] `SPEC.md` / `docs/src/api.md` / guide pages: new export `MU_B_EV_T`;
      public-unexported `has_field`, `zeeman_energy`; guide page `field.md`.

## Test strategy

New file `test/unit/test_zeeman.jl`; existing gates extended in place where a
field variant is the natural twin. Oracles:

| Gate | Oracle |
|---|---|
| Field identity: `total_energy(H_B, cfg) − total_energy(H_0, cfg) ≈ −MU_B_EV_T Σ_s m_a e_s·B` | hand-written sum in the test (no package function); **absolute** tolerance `10·√n_sites·eps(\|E_SCE\|)` (interleaved accumulator); dimer + `_biquadratic_model`, `dims = (1,1,1)` and `(2,2,2)` (replication = 8× the cell Zeeman energy) |
| `zeeman_energy` ≡ that hand sum | same hand sum |
| Structure: `n_fitted_terms`, `length(H.terms) == n_fitted + count(m_a ≠ 0)`, `site_active` / `site_has_l1` on moment-only sites, colour of an isolated site, `lmax == 1` on an all-`l = 0` list with a field | counts by construction; the `lmax` case is the silent-OOB guard |
| `ΔE ≡ total-energy difference` with field | exact identity (1e-12), existing pattern |
| `site_coeffs!` own-spin independence with field | exact identity, existing pattern |
| Program ≡ reference kernels with field | bitwise `==` (existing gate, field fixture added) |
| Gradient: `G_B[s] − G_0[s] ≈ −MU_B_EV_T m_a (B − (e·B) e)` | closed form (tangent projection of a constant), absolute tolerance as above; plus the FD gate re-run |
| Free moments: `⟨e·B̂⟩ = L(x)`, `x = β MU_B_EV_T m B` | Langevin function `L(x) = coth x − 1/x` (fixture `_langevin`), dimer fixture's free atoms 3–4 given `m ≠ 0`; tolerance = measured σ with headroom (record σ, seed, and the mutation size — a sign flip of `MU_B_EV_T` must fail) |
| Same on the CPU-backend device path (`gpu_run_sweeps!`) | same Langevin law |
| `magmoms` alone, moment placed on the SCE-inactive atoms 3–4 | `length(H.terms)`, `site_active`, `site_has_l1`, colouring, `total_energy` / `site_coeffs!` / `energy_gradient!` of any configuration, and a seeded `run_mc` result including `:m` / `:sublattice_m` all `==` the field-free `H` bitwise; `:M` present (frozen atoms excluded), `:M_B` absent |
| `ReducedCell` form with a field | field identity on `TiledHamiltonian(reduce_cell(...); field, magmoms)` vs the same hand sum (the third constructor form) |
| OR with field on a pure-`l = 1` model | `ΔE ≡ 0`, acceptance 1 (exact reflection symmetry of a linear energy) |
| Ground state of free moments in a field, default `gtol` / ladder | `e_s == B̂` to `gtol` (closed form) |
| GPU sweep ≡ keyed reference, gradient ≡ lane reference, body-1 fixture | bitwise `==` (existing gates, new fixture) |
| Device entry-walk `ΔE` vs `site_coeffs!`+`delta_energy` with field | tolerance (accumulation order differs), existing pattern |
| MC / PT / GPU-PT resume with field | bit-identical resume (mid-measure, `_poison_pair`, position asserted) |
| Fingerprint: `B = 0` vs field-free, `(m, B)` vs `(m/2, 2B)`, different `field` ⇒ resume errors | error raised |
| Fingerprint of a field-free dimer == captured value | **labelled regression pin** (change detector): captured at `c7a354a` before this spec; recapture only on an intended fingerprint change (schema-version territory) |
| `:M`, `:M_B` on a fixed configuration | hand-computed sums |
| Zero-moment sublattice with a field | bitwise frozen + excluded (existing `test_inactive.jl` pattern) |
| Door validation | errors: length mismatch, negative / non-finite `m_a`, `field` without `magmoms`, non-finite field |

## Forward compatibility (informational — not promised here)

`predict_moment` evaluates `y_a = ê_a·m_a(e)` with `ê_a = e_a`; each pointed
term carries `Z_{l_mark}(e_a) × Π Z_{l_env}(e_n)`. Multiplying by
`(e_a·B) ∝ Z_1(e_a)` and recoupling the marked site's `Z_{l_mark}·Z_1` with
real Gaunt coefficients gives `L ∈ {l_mark − 1, l_mark + 1}` on the same site
— the body order is unchanged, `allunique` holds, and single-spin `ΔE` stays
exact. `_zeeman_terms` is the extension point; `H.lmax` would become
`max(lmax_SCE, lmax_env, lmax_mark + 1)`. Prerequisites outside this package:
a `moment_terms(::MomentModel)` introspection surface (this package must not
reach into `model.basis`), `MomentModel` persistence (`save` throws today),
and an independently-oracled real-tesseral Gaunt kernel. `:M` would then need
its own pointed-expansion evaluation.

## Risks and open items

- **eV assumption.** `MU_B_EV_T` hard-codes eV, like `KB_EV`. A non-eV model
  silently gets the wrong field scale — same exposure as `temperature` today;
  documented, not guarded (the model carries no unit).
- **Dependent packages.** SLCEDynamics-style consumers add their own Zeeman
  term on top of `energy_gradient!`; `has_field` plus the docstring are the
  guard. Field-free `H` is byte-identical, so nothing changes unless a field
  is set.
- **`:m` with a field.** A moment-carrying SCE-inactive sublattice enters
  `:m` / `:sublattice_m` once a field is set (it is then a genuine, polarized
  magnetic site). Finite-size-scaling users run at `B = 0`, where nothing
  changes; documented in B3.
- **`H.terms` on the public tier.** "Every element is a fitted SALC" stops
  being true; the `n_fitted_terms` split and `SPEC.md` note are the
  mitigation. No consumer in this package or in SLCEDynamics parses
  `H.terms` (verified: `_fingerprint`, `_site_energy_scale`, `Base.show`,
  reference kernels, `bench/fixtures.jl` count, two field-free test counts).
- **Fingerprint collisions.** Handled by the explicit mixing; the pin guards
  the field-free value.

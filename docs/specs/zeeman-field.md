# Decision record — external field (constant moments, body-1 Zeeman templates)

Status: landed (spec `260822-zeeman-field`, 2026-08-22); device gates re-claimed
on an A100 the same day (`bench/bench_gpu_zeeman.jl`, `.claude/bench_log.md` #4:
all gates pass on CUDA, +3 % sweep cost with a field). Owner:
`src/hamiltonian.jl` (`_resolve_zeeman`, `_zeeman_terms`, `has_field`,
`zeeman_energy`), `src/observables.jl` (`:M`, `:M_B`), `src/checkpoint.jl`
(`_fingerprint`, the `zeeman/` group); gates in `test/unit/test_zeeman.jl` and
the Zeeman fixtures of `test/unit/test_gpu.jl`.

## Z1 — sign and units

`TiledHamiltonian(...; magmoms, field)` adds

```
E_Z = − MU_B_EV_T · Σ_s m_{a(s)} (e_s · B)
```

with `m_a ≥ 0` in μ_B per training-cell atom (`a(s) = site_atom(s)`), `B` in
tesla, uniform, and `MU_B_EV_T = 9.2740100783e-24 / 1.602176634e-19` eV/T (the
value and the name of SLCEDynamics' constant). The moment of site `s` is
`m_{a(s)} μ_B e_s` — SCEFitting's `directions` are moment directions — so a
ferromagnet in `B ∥ ẑ` aligns with `+ẑ`. The model is assumed eV-fitted, exactly
as `temperature` assumes through `KB_EV`; a model in other units passes a
pre-scaled field. No `(4π)` factor of any kind enters the term.

Doors: `length(magmoms) == n_cell_atoms`, every `m_a` finite and `≥ 0` (a sign
belongs to the direction); `field` a finite 3-vector; a field requires
`magmoms`. `magmoms` alone is allowed (Z3).

Naming ledger: `magmoms` follows SCEFitting's `SpinDatum.magmoms`; `field` is
this package's constructor keyword for the quantity SLCEDynamics calls `b_ext`
on `LLGProblem` — a deliberate per-package choice, not a drift. `MU_B_EV_T` is
exported like `KB_EV`; SLCEDynamics exports an equal constant of the same name,
so a session loading both must qualify it (documented in the guide).

## Z2 — representation: body-1 cluster templates, no kernel changes

With `Z_{1,m} = N1·(y, z, x)_m` in `lm_index` order (rows 2, 3, 4; `N1 =
√(3/4π)` read from `SCEFitting.Harmonics.N1`, the one definition the
overrelaxation axis also relies on),

```
−μ_B m_a (e · B) = Σ_m hz_a[m] Z_{1,m}(e),   hz_a = −(MU_B_EV_T m_a / N1)·(B_y, B_z, B_x)
```

is an exact body-1 cluster term. The constructor appends, for every cell atom
with `hz_a ≠ 0`, `ScaledTerm(1.0, [a], [0], [1], hz_a)` to `H.terms` **after**
the fitted terms have been scaled. `coef = 1.0` with `folded = hz_a` makes the
program weights `1.0·hz === hz` exact; routing the term through a raw
`MultipoleTerm` with `coef = 1/√(4π)` and letting the scale-once rule apply
`(4π)^{1/2}` is 1 ulp off and is not done. This is the one exception to the
"scale applied once to every term" statement of `hamiltonian-tiling.md` T4.

Everything downstream is body-generic and reads `H.terms` or the programs built
from it, so the term flows, with **no code change**, through: `site_coeffs!`
(general path, zero factors, `p = 1.0`, `c[tgt] += w·1.0 ≡ c[tgt] += w` bitwise),
`_total_energy`, both reference kernels (program ≡ reference stays bitwise),
`delta_energy` and the Metropolis accept, the overrelaxation axis
`(c[4], c[2], c[3])` (now the total local field), `_site_grad` /
`energy_gradient!` / the descent, `_site_energy_scale` (default `gtol` and
annealing ladder — a Zeeman-only sublattice contributes
`MU_B_EV_T m_a ‖B‖₁ / N1`), `_renormalize!`, PT swap energies, the CSR adjacency
(`site_active`, `site_has_l1`), `_color_sites` (an isolated site gets colour 1;
sites with fitted instances keep their colour bitwise), `_fingerprint`, and the
device tables / `_entry_walk_partial` / `_entry_walk_grad` (general branch with
an empty factor range — the same functions the keyed / lane references call).

Two things the templates cannot carry stay explicit on `H`: `magmoms` and
`field` themselves (observables, `zeeman_energy`, `has_field`, fingerprint
disambiguation, the checkpoint record) and the boundary `n_fitted_terms`
(`H.terms[1:n_fitted_terms]` = scaled fitted SALCs in `multipole_terms` order;
the rest = Zeeman templates in atom order). `ScaledTerm` / `H.terms` are on
the public tier, so "every element of `H.terms` is a fitted SALC" is no longer
true — `SPEC.md` says so; `Base.show` reports the split.

`lmax` is computed from the fitted terms only; appending a Zeeman template
sets `lmax = max(lmax, 1)` explicitly, otherwise an all-`l = 0` model would
index rows 2:4 of `c` out of bounds under `@inbounds` (silent, not an error).

Rejected: a dedicated per-atom `l = 1` coefficient offset threaded through
`site_coeffs!`, `_total_energy`, both reference kernels, both device entry
walks (with a lane-1 guard and a signed-zero-safe conditional), a new
activation rule, `_site_energy_scale`, and the fingerprint — eleven files and
five hand-mirrored addend sites for the same physics, and no path to
configuration-dependent moments (Z6).

## Z3 — append rule and activation

One template per atom with `m_a ≠ 0` **and** `B ≠ 0`. With `magmoms` alone (or
`B = 0`) no template is appended and the Hamiltonian's terms, activation,
colouring, energies, coefficient vectors, gradients, sweeps, and every
pre-existing observable are bitwise the field-free ones — only `H.magmoms`, the
fingerprint input, the `:M` observable, and the checkpoint record are added.

Consequently the activation rule is unchanged ("has an adjacent instance"): a
site no fitted cluster touches whose atom carries a moment is active in a
field (it owns a Zeeman instance; exact `ΔE`, exact gradient; its
overrelaxation move reflects about `B̂` and preserves `e·B̂` exactly), and stays
frozen and excluded without one. `m_a = 0` atoms never get a template.
`binning-observables.md` B3 records the one way a non-SCE site becomes
magnetic (a field) and that it then counts in `:m` / `:sublattice_m` / `:M`.

Rejected: appending zero-`folded` templates without a field ("free rotors") —
it would activate moment-carrying SCE-inactive sublattices as soon as
`magmoms` is given and silently dilute the B3 observables `:m` /
`:sublattice_m`.

Behaviour changes a user may notice with a field: the `or_per_metropolis` door
("no `l = 1` channel") no longer fires for a model whose only `l = 1` channel is
the field; the overrelaxation move on a Zeeman-only site is accepted work that
moves nothing along `B̂` (no bias — OR is not ergodic by design).

## Z4 — fingerprint and checkpoint

`_fingerprint` already mixes every template, but the templates see only the
product `m_a·B`: `magmoms` without a field emits none (collides with
field-free), and `(m, B)` / `(m/2, 2B)` emit identical ones (same energy,
different `:M`). So, **only when `magmoms !== nothing`**, the fingerprint
additionally mixes a tag `1`, every `m_a`, and the three field components.
Field-free Hamiltonians mix nothing new. (The original requirement that every
stored field-free fingerprint stay byte-identical held through `30f1797`; the
schema-v3 mixer fix below then changed every fingerprint on purpose, and the
pin in `test_zeeman.jl` now detects changes to the v3 algorithm instead.)

The checkpoint writer adds `zeeman/magmoms` (`Vector{Float64}`) and
`zeeman/field` (`Vector{Float64}(3)`) when `magmoms` is given — informational,
additive (the reader verifies through the fingerprint only). The group itself
needed no schema bump; the mixer fix below did (`checkpoint-schema.md` C3).

Dependent packages: the Zeeman part is in `energy_gradient!` when
`has_field(H)`; `has_field` is the predicate a consumer must assert against
before applying a field of its own. The SLCEDynamics-style `LLGProblem`
(`b_ext` / `gzee` on top of `energy_gradient!`) has **no such assertion yet** —
the double-counting guard is documented here and in the docstring, and
enforcing it on the dependent side is an open follow-up, not a closed contract.

Sign bits: the v2 `_fp_mix` (plain FNV-1a XOR-multiply) carried a `Float64`
sign bit only into bit 63 of the hash, so a plain mix of the field words let
`B → −B` — or any single-component flip — cancel whenever the number of
moment-carrying atoms is odd (the elemental ferromagnet has one): the resume
door would have been blind to a reversed field. The first fix (`30f1797`) mixed
the field words twice, once bit-rotated, inside the `magmoms !== nothing`
block, leaving field-free fingerprints untouched but the same linearity alive
inside fitted `folded` tensors (recorded as `@test_broken`). The decision
(2026-08-22) was to fix the mixer itself: schema v3 folds the top half after
every multiply and adds a final avalanche (`checkpoint-schema.md` C3), so no
sign-flip pattern cancels anywhere in the payload, the field words are mixed
once like everything else, and every fingerprint changed.

## Z5 — observables and gates

`standard_observables(H)` appends `:M = Σ_{s active} m_{a(s)} e_s / n_cells`
(μ_B per training cell — the `:sublattice_m` normalization, not `:m`'s
per-active-site one) when `magmoms` is given, and `:M_B = M·B̂` when
`has_field(H)`. Conventions are authoritative in `binning-observables.md` B3.

Gates (`test/unit/test_zeeman.jl`, oracles independent of the implementation):
field identity `total_energy(H_B) − total_energy(H_0)` against a hand-written
sum within an **absolute** `10·√n_sites·eps(max(|E_SCE|, |E_Z|))` (the Zeeman
instances are interleaved into one accumulator — a relative 1e-12 would fail
on large fixtures); structure (templates, activation, colouring, `lmax` on an
all-`l = 0` list, `show`); doors; `ΔE ≡` total-energy difference, own-spin
independence, program ≡ reference bitwise with a field; gradient closed form
`−μ_B m_a (B − (e·B) e)` + finite differences; `magmoms`-only bitwise identity
(terms, activation, energies, and a seeded `run_mc` including `:m` /
`:sublattice_m`); zero-moment sublattice frozen with a field; `ReducedCell`
form; `:M` / `:M_B` hand sums; Langevin law `⟨e·B̂⟩ = L(β μ_B m B)` for a free
moment (σ ≈ 0.009 measured, atol 0.04, a sign-flip mutation moves the mean by
2L ≫ atol) on the host and on the CPU-backend device path; overrelaxation on a
pure-`l = 1` model plus field exact (odd sweep count); ground state along `B̂`
with the default `gtol` / ladder; serial ≡ parallel with a field; fingerprint
identity cases; MC / PT / GPU-PT resume bit-identical with a field and mismatch
errors for rescaled / different / absent fields. `test_gpu.jl`'s G3 / G7
bitwise gates and the direct-`ΔE` tolerance gate carry Zeeman fixtures
(Zeeman-only site, fitted + Zeeman sites, all-body-1 model with zero-length
factor tables).

## Z6 — forward compatibility (not promised)

SCEFitting's pointed moment model `m_a(e)` (`MomentModel`, `predict_moment`,
2026-08-21) evaluates `ê_a·m_a(e)` with `ê_a = e_a`; each term carries
`Z_{l_mark}(e_a) × Π Z_{l_env}(e_n)`. Multiplying by `(e_a·B) ∝ Z_1(e_a)` and
recoupling the marked site's `Z_{l_mark}·Z_1` with real Gaunt coefficients
gives `L ∈ {l_mark − 1, l_mark + 1}` on the same site — the body order is
unchanged, the distinct-member-sites invariant holds, and single-spin `ΔE`
stays exact. `_zeeman_terms` is the extension point (`H.lmax` would become
`max(lmax_SCE, lmax_env, lmax_mark + 1)`; the device path gates `lmax ≤ 6`).
Prerequisites outside this package: a `moment_terms(::MomentModel)`
introspection surface (this package must not reach into `model.basis`),
`MomentModel` persistence (`save` throws today), and an independently-oracled
real-tesseral Gaunt kernel. `:M` would then need its own pointed-expansion
evaluation. The follow-up keyword is `moment_model`, validated XOR against
`magmoms` in the `temperature` xor `kT` style — a union-typed `moments` keyword
was rejected (it fights `SpinDatum.magmoms` / SLCEDynamics `magmom` and
forecloses the XOR door).

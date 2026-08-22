# Requirements: Zeeman term — constant site moments in a uniform field

Status: draft v2 (2026-08-22) — v1 (dedicated coefficient offset) replaced by
the body-1 cluster-term representation after the numerical / api /
maintainability reviews.

## Goal

Let a `TiledHamiltonian` carry an external magnetic field: `E = E_SCE + E_Z` with

```
E_Z = − μ_B Σ_s m_{a(s)} (e_s · B)        [model energy units, eV]
```

where `m_a ≥ 0` (μ_B) is a **constant** moment magnitude per training-cell atom
`a(s) = site_atom(s)` and `B` (tesla) is **uniform** over the supercell. Every
consumer of the energy — Metropolis / overrelaxation sweeps, total energy,
`energy_gradient!`, ground-state descent, CPU and GPU PT, checkpoints — sees the
same `E`, so `M(B, T)` curves, field-cooled ground states, and field-aware
effective fields for dependent packages become possible without any of them
re-implementing the term.

## Background

The package holds **unit directions only**: the fitted SCE model carries no
moment magnitudes (`SCEPredictor` has none; `SpinDatum.magmoms` lives on the
data side of SCEFitting), `standard_observables` reports dimensionless
direction averages, and `energy_gradient!` defers magnitudes to the caller
(`B_s = −G[s]/(m_s μ_B)`, "moment magnitudes are the caller's"). There is no
Zeeman term anywhere (nor in SLCEMonteCarlo.jl).

The Zeeman energy of a site is a **body-1 cluster term**: `−μ_B m_a (e·B)` is
exactly `Σ_m hz_a[m] Z_{1,m}(e)` with the `l = 1` tesseral row. The package's
term machinery is body-generic and already exercised on body-1 terms
(`test_hamiltonian.jl` single-site `l = 1` field term, `test_energy.jl`
program ≡ reference gate, `test_metropolis.jl` Langevin gate), so the term is
represented as one `ScaledTerm` per moment-carrying cell atom, appended to
`H.terms` by the constructor. No energy, update, gradient, colouring,
fingerprint, or device kernel changes: every coupled site inherits the term
through the existing contracts instead of mirroring a new addend.

SCEFitting gained the pointed site-moment model `m_a(e)` on 2026-08-21
(`MomentModel` / `predict_moment`). Under the term representation that model
recouples (real Gaunt product on the marked site) into multi-body cluster
terms of the **same** body order, so step (b) keeps exact single-spin `ΔE`.
It is **not** promised by this spec: it is blocked upstream on a
`moment_terms(::MomentModel)` introspection surface, `MomentModel`
persistence, and a real-tesseral Gaunt kernel (none exist today).

Decision records touched: `hamiltonian-tiling.md` (T4 contract note; the
scale-once rule gains its one exception), `checkpoint-schema.md` (fingerprint
scope), `binning-observables.md` (new magnetization observables). A new record
`docs/specs/zeeman-field.md` holds the Zeeman contract itself.

## Scope

Includes:

- `TiledHamiltonian(...; magmoms, field)` on all three constructor forms
  (`SCEPredictor`, raw term list, `ReducedCell`); `magmoms` in μ_B per cell
  atom, `field` in tesla; `MU_B_EV_T` next to `KB_EV` in `units.jl` (the name
  matches SLCEDynamics' existing constant).
- One body-1 `ScaledTerm` per cell atom with `m_a ≠ 0`, appended **only when
  a nonzero field is set**. Without a field, `magmoms` adds state (`H.magmoms`,
  the `:M` observable, fingerprint input) and nothing else — activation,
  colouring, energies, and every pre-existing observable stay bitwise
  field-free, so `binning-observables.md` B3 is untouched. `n_fitted_terms`
  records the boundary between fitted and synthetic terms.
- `has_field(H)` predicate and documented `H.magmoms` / `H.field` fields, so a
  dependent package that adds its own Zeeman term can assert it is not
  double-counting; `zeeman_energy(H, config)` closed-form helper.
- Observables: `:M = Σ_{active} m_{a(s)} e_s / n_cells` (μ_B per training
  cell, 3 components) whenever `magmoms` is given; `:M_B = M · B̂` when
  `has_field(H)`.
- Checkpoint identity: `model_fingerprint` mixes `magmoms` and `field`
  explicitly **only when present** — the synthetic terms see only the
  product `m_a·B`, so `magmoms` without a field (no term) would collide with
  field-free and `(m, B)` with `(m/2, 2B)`; every existing field-free
  fingerprint is unchanged; an informational `zeeman/{magmoms, field}` group
  is written additively (schema stays v2).
- Test coverage of the body-1 path on the device (G3 / G7 bitwise gates gain
  a body-1 fixture — the empty-factor general branch is unexercised today).
- Docs: guide page `docs/src/guide/field.md`, `observables.md` /
  `running.md` / `ground_states.md` / `gpu.md` touch-ups, `api.md`,
  `SPEC.md` (public `ScaledTerm` tier: `H.terms` is no longer "fitted SALCs
  only"), the decision records above, `CHANGELOG.md`.

Excludes:

- Configuration-dependent moments `m_a(e)` from `MomentModel` (follow-up
  spec; prerequisites listed in Background).
- Per-site (cell-resolved) moments, site-dependent or time-dependent fields,
  field tempering across PT rungs, demagnetization / dipolar terms.
- Automatic reduction of `magmoms` through `reduce_cell` (the caller supplies
  moments for the cell it tiles).
- Differential susceptibility evaluables along `B̂` (computable from `:M_B`
  at two fields; may follow once the observable has been used).
- Any change to SCEFitting or SLCEMonteCarlo.

## Invariants

- Spins stay unit vectors; `m_a = 0` sites keep today's inactive convention
  bitwise (no term is emitted for them).
- `(4π)^(body/2)` applied once to the **fitted** terms in the ctor; `j0`
  excluded. The Zeeman terms are appended already in consumer form (`coef =
  1.0`, no `4π`) — the one documented exception to scale-once (they are not
  fitted SALCs).
- `temperature` xor `kT`; β only in accept steps; Zeeman coefficients are in
  model energy units like every other entry of `c`.
- Sign and unit convention: the moment of site `s` is `m_{a(s)} μ_B e_s` (the
  SCEFitting `directions` are moment directions), `E_Z = −μ·B`, `B` in tesla,
  `μ_B = MU_B_EV_T` eV/T — the eV-fitted-model assumption `KB_EV` already makes.
- Detailed balance / stationarity of every update (U1 colour classes): a
  body-1 instance is never shared by two sites, so the conflict graph of the
  fitted model is unchanged and a moment-only site is an isolated vertex.
- Exact single-spin `ΔE` (distinct member sites per instance holds trivially).
- Bitwise contracts: program kernels ≡ reference kernels, serial ≡ parallel,
  PT thread-count independence, resume ≡ uninterrupted (mid-measure), GPU
  kernel ≡ keyed reference, gradient kernel ≡ lane reference — all by
  construction, since no kernel changes; re-gated with a field.
- **Field-free behaviour is byte-identical**: with `magmoms === nothing` no
  term is appended, no fingerprint input is added, and every energy,
  coefficient vector, gradient, sweep, observable, fingerprint, and checkpoint
  file is exactly what it is today. With `magmoms` but no field, the same
  holds for everything except the fingerprint, the `:M` observable, and the
  informational checkpoint group — in particular the B3 observables `:m` /
  `:sublattice_m` and `n_active` are unchanged.
- Checkpoint schema version stays 2; dependent-package-facing names
  (`energy_gradient!`, `model_fingerprint`, `_gradient_lane_ref!`,
  `philox_*`) keep their signatures and, for field-free Hamiltonians, their
  values.

## Completion criteria

- [ ] `make test-all` passes (4 threads); `make docs` builds strict.
- [ ] Field identity gate: `total_energy(H_B, cfg) − total_energy(H_0, cfg)`
      equals the hand-written `−MU_B_EV_T Σ_s m_a e_s·B` within an
      **absolute** tolerance `10·√n_sites·eps(|E_SCE|)` (the Zeeman
      instances are interleaved into one accumulator) on the dimer and
      `_biquadratic_model` fixtures, `dims = (1,1,1)` and `(2,2,2)`.
- [ ] `ΔE ≡ total-energy difference`, `site_coeffs!` own-spin independence,
      program ≡ reference (bitwise), and the gradient finite-difference gates
      all re-run **with a field**.
- [ ] Statistical gate with an independent oracle: free moments in a field
      reproduce the Langevin law `⟨e·B̂⟩ = L(β μ_B m B)` within a stated σ
      with headroom, on the host and on the CPU-backend device path.
- [ ] `magmoms`-only gate: without a field, a moment on an SCE-inactive
      sublattice leaves `H.terms`, activation, colouring, every
      per-configuration energy / coefficient / gradient, and a seeded
      `run_mc` result (including `:m` / `:sublattice_m`) bitwise equal to the
      field-free `H`; `:M` present, `:M_B` absent.
- [ ] Third constructor form: field identity on a `ReducedCell`-built `H`.
- [ ] Overrelaxation gate: a pure-`l = 1` model plus field still has `ΔE ≡ 0`
      and acceptance 1.
- [ ] Ground-state gate: free moments in a field end at `e_s = B̂` with the
      **default** `gtol` / ladder (closed form; exercises `_site_energy_scale`
      on a Zeeman-only sublattice).
- [ ] GPU gates: sweep ≡ keyed reference and gradient ≡ lane reference
      bitwise on a fixture containing body-1 terms (with and without fitted
      terms on the same site); entry-walk `ΔE` vs `site_coeffs!`+`delta_energy`
      within tolerance with a field.
- [ ] Checkpoint gates: MC / PT / GPU-PT resume bit-identical with a field;
      resume against a different `field` or `magmoms` errors — including the
      `B = 0` vs field-free and `(m, B)` vs `(m/2, 2B)` pairs; the field-free
      fingerprint of the dimer fixture equals its value captured at `c7a354a`
      (labelled change-detector pin).
- [ ] Observable gate: `:M` / `:M_B` on a fixed configuration equal the
      hand-computed sums.
- [ ] Inactive gate: `magmoms` with a zero entry keeps that sublattice bitwise
      frozen and excluded, with and without a field.
- [ ] Structure gate: `n_fitted_terms`, `length(H.terms)`, `site_active`,
      `site_has_l1`, colour of a moment-only site, and `lmax ≥ 1` on an
      all-`l = 0` term list with a field (the silent out-of-bounds case).
- [ ] Decision record `docs/specs/zeeman-field.md` written; T4 / scale-once
      note, checkpoint, binning-observables records updated; `SPEC.md`,
      `api.md`, guide pages, `CHANGELOG.md` updated; `.claude/agents/` swept
      for the new coupled site; Tier 2 panel run.

## References

- Related decision records: `docs/specs/hamiltonian-tiling.md` (T2, T4, T5),
  `docs/specs/updates-stationarity.md` (U1), `docs/specs/checkpoint-schema.md`
  (C2), `docs/specs/binning-observables.md` (B3), `docs/specs/gpu-prototype.md`
  (G3, G4, G7 — unchanged contracts, new fixture).
- Existing body-1 usage: `test/unit/test_hamiltonian.jl` (single-site `l = 1`
  field term, hand-checked `c0·N1·e_z`), `test/unit/test_energy.jl` (body-1
  term in the bitwise program ≡ reference gate), `test/unit/test_metropolis.jl`
  (Langevin gate on that term).
- Upstream: `SCEFitting.Harmonics.N1` and the `Z_{1,m} = N1·(y, z, x)`
  convention (`SCEFitting.jl/src/basis/harmonics.jl`); the overrelaxation axis
  extraction in `src/updates.jl` already relies on it.
- Sibling: `SLCEDynamics.jl/src/units.jl` `MU_B_EV_T` (same value; name
  adopted here), `src/integrators.jl` `_omega` (adds its own `gzee` on top of
  `energy_gradient!` — the double-counting hazard `has_field` is for).
- Follow-up spec (not started): configuration-dependent moments from
  `SCEFitting.MomentModel`; prerequisites in Background.

# External field

```@meta
CurrentModule = SCEMonteCarlo
```

A fitted SCE model carries spin **directions** only — no moment magnitudes and
no external field. To put the system in a uniform field you supply what the
model lacks, a constant moment magnitude per training-cell atom, together with
the field:

```julia
H = TiledHamiltonian(model; dims = (8, 8, 8),
                     magmoms = [2.2, 2.2, 0.0],     # μ_B per training-cell atom
                     field = (0.0, 0.0, 2.0))       # tesla, uniform
```

This adds the Zeeman energy

```math
E_Z = -\mu_B \sum_s m_{a(s)}\, \mathbf e_s \cdot \mathbf B
```

to every energy the package computes — sweeps, `total_energy`,
`energy_gradient!`, the ground-state search, CPU and GPU parallel tempering,
checkpoints — with `a(s)` the training-cell atom of site `s`
([`site_atom`](@ref)).

## Conventions

- The moment of site `s` is ``m_{a(s)}\,\mu_B\,\mathbf e_s``: the SCEFitting
  `directions` are moment directions, so a ferromagnet in ``\mathbf B \parallel
  \hat z`` aligns with ``+\hat z``.
- `field` is in tesla, `magmoms` in ``\mu_B`` (``\ge 0`` — a sign belongs to
  the direction), and the conversion to the model's energy unit is
  [`MU_B_EV_T`](@ref) (eV/T). This assumes an **eV-fitted** model, exactly as
  `temperature` does through [`KB_EV`](@ref); a model in other units must pass
  a pre-scaled field.
- A field requires `magmoms` (an error otherwise). `magmoms` without a field
  (or with `B = 0`) changes nothing about the dynamics: it only records the
  moments, enables the `:M` observable, and enters the model fingerprint.

## How it is represented

The Zeeman energy of a site is linear in its direction, i.e. an `l = 1`
cluster term of body order one. The constructor therefore appends one
`ScaledTerm` per atom with ``m_a \ne 0`` to `H.terms` — after the fitted terms,
with `coef = 1.0` and no ``(4\pi)`` scale (they are not fitted SALCs) —
and **nothing else changes**: the energy kernels, the overrelaxation axis
(now the total local field including ``\mathbf B``), the gradients, the
colouring, the default ground-state tolerances, the device tables, and the
fingerprint all see the term through the ordinary term machinery.
`H.terms[1:H.n_fitted_terms]` are the fitted terms, the rest the Zeeman
templates; `show(H)` reports the split and the field.

Two consequences worth knowing:

- A site no fitted cluster touches but whose atom carries a moment becomes
  **active** in a field (it owns a Zeeman instance): it is swept, it enters
  `:m` / `:sublattice_m` / `:M`, and its overrelaxation move reflects about
  ``\hat{\mathbf B}``. Without a field it stays frozen and excluded, as any
  inactive site.
- A model whose fitted terms carry no `l = 1` channel gains one through the
  field, so `or_per_metropolis` is accepted for it.

The decision record is `docs/specs/zeeman-field.md`.

## Observables

With `magmoms`, [`standard_observables`](@ref) grows by

- `:M` — ``\sum_{s\,\text{active}} m_{a(s)}\,\mathbf e_s / n_{\text{cells}}``,
  the magnetization in ``\mu_B`` **per training cell** (3 components; the
  `:sublattice_m` normalization, not `:m`'s per-active-site one);
- `:M_B` — ``\mathbf M \cdot \hat{\mathbf B}``, when the field is nonzero.

An ``M(B, T)`` curve is then a loop over **nonzero** fields (`:M_B` exists only
with a field; read `:M` at ``B = 0``):

```julia
fields = [(0.0, 0.0, B) for B in 0.5:0.5:4.0]
MB = map(fields) do B
    H = TiledHamiltonian(model; dims = (8, 8, 8), magmoms = mm, field = B)
    r = run_mc(H; temperature = 300.0, sweeps_therm = 2_000, sweeps_measure = 20_000,
               or_per_metropolis = 2, seed = 1)
    r.points[1].stats[:M_B].mean[1]          # μ_B per training cell along B̂
end
```

Two caveats for field sweeps:

- If a moment-carrying sublattice has no fitted cluster, it is inactive at
  ``B = 0`` and active for any ``B \ne 0``, so `n_active` and every
  direction observable (`:m`, `:absm`, `:m2`, `:m4`, the derived ``χ`` / ``U``)
  change their site set across ``B = 0`` — they are not comparable between the
  zero-field point and the rest of the curve. `:M` sums over active sites and
  changes its site set the same way.
- `standard_evaluables()` keeps the zero-field finite-size-scaling estimators
  (``χ`` from `|m|`, the Binder ``U``); a differential susceptibility along
  ``\hat{\mathbf B}`` is not provided — difference `:M_B` between two fields.

`SCEMonteCarlo.zeeman_energy(H, config)` ([`zeeman_energy`](@ref)) returns the
closed-form field contribution of a configuration, and
`SCEMonteCarlo.has_field(H)` ([`has_field`](@ref)) tells whether `H` carries
one — both are public but not exported, so qualify them. [`MU_B_EV_T`](@ref)
is exported like `KB_EV`; SLCEDynamics exports a constant of the same name and
value, so a session that loads both packages must qualify it
(`SCEMonteCarlo.MU_B_EV_T`).

## Checkpoints and dependent packages

The model fingerprint includes the moments and the field, so a resume against
a different field (or against a field-free twin) errors instead of silently
continuing the wrong physics; the file also carries an informational
`zeeman/{magmoms, field}` group. Field-free Hamiltonians are byte-identical to
before — their fingerprints and checkpoint files are unchanged.

A dependent package that builds its own effective field from
[`energy_gradient!`](@ref) must know that the Zeeman part is **already in**
`G` when `has_field(H)`; applying a field of its own on top would count it
twice — assert `!has_field(H)` there.

## Scope

Constant, per-atom moments in a uniform, static field. Not covered: per-site
(cell-resolved) moments, site- or time-dependent fields, field tempering across
parallel-tempering rungs, dipolar / demagnetizing terms, and
configuration-dependent moments ``m_a(\mathbf e)`` (SCEFitting's pointed
moment model — a planned follow-up that uses the same term representation).
`reduce_cell` does not translate moments: pass `magmoms` indexed by the
**reduced** cell's atoms to `TiledHamiltonian(red; ...)`.

# Constrained Monte Carlo: finite-temperature anisotropy

```@meta
CurrentModule = SCEMonteCarlo
```

In an ordinary Monte Carlo run of a finite magnet the direction of the
magnetization wanders: the anisotropy only biases *where* it spends its time,
and reading the anisotropy constant ``K(T)`` off that bias needs an enormous
histogram. **Constrained Monte Carlo** (CMC; Asselin et al. 2010) instead holds
the direction ``\hat n`` of the (weighted) magnetization fixed and lets every
other degree of freedom — the magnitude of ``\mathbf M``, the transverse
fluctuations, the spin waves — equilibrate. The mean torque on the system is then
exactly the gradient of the free energy of that direction,

```math
\langle \mathbf T \rangle = -\hat n \times \frac{\partial F}{\partial \hat n},
```

so a scan over directions gives ``F(\hat n)`` and with it ``K_1(T)``, ``K_2(T)``,
or a general set of anisotropy coefficients — at a cost of a few independent
runs, one per direction.

Use it for the temperature dependence of the magnetic anisotropy of an ordered
magnet (ferro- or ferrimagnet; an antiferromagnet with a suitable choice of
weights), and to split it into single-ion and two-ion contributions. It is not
the tool above the ordering temperature (the direction of a vanishing
magnetization says nothing about the anisotropy — the run warns) or for anything
that needs the magnetization to rotate freely (susceptibilities, Binder
cumulants: use [`run_mc`](@ref) / [`run_pt`](@ref)).

The algorithm and the identities behind it are on the
[theory page](../theory/constrained_mc.md); the standing contract is the
decision record `docs/specs/constrained-mc.md`.

## A worked example

A hand-built ferromagnet on a simple-cubic lattice, one atom per cell, so the
page runs in seconds: isotropic nearest-neighbor exchange ``J = -1``, an extra
``e_z e_z'`` coupling on the bonds along ``z`` (a two-ion anisotropy), and a
uniaxial single-ion ``l = 2`` term — both with the easy axis ``z``. The terms are
written as the raw `MultipoleTerm`s a fitted model would hand over
(`multipole_terms(model)`); with a fitted model you would build
`H = TiledHamiltonian(model; dims)` instead and keep `terms =
multipole_terms(model)` for the partial torques below.

```@example cmc
using SCEMonteCarlo, SCEFitting
using LinearAlgebra, StaticArrays

N1 = SCEFitting.Harmonics.N1        # Z_1m(e) = N1 · (e_y, e_z, e_x)[m + 2]
o = SVector(0, 0, 0)
# both directed members of the bond along d; the raw coefficient undoes the
# (4π)^(2/2) scale of the constructor, N1², and the double count, so E_bond = J e·e′
bond(J, d, folded) = [MultipoleTerm(J / (2 * N1^2 * 4π), 2, [1, 1], [o, s * d],
                                    [1, 1], folded) for s in (1, -1)]
iso = Matrix(1.0I, 3, 3)            # Σ_m Z_1m(e) Z_1m(e′) ∝ e·e′
zz = diagm([0.0, 1.0, 0.0])         # the z components only
terms = [bond(-1.0, SVector(1, 0, 0), iso); bond(-1.0, SVector(0, 1, 0), iso);
         bond(-1.0, SVector(0, 0, 1), iso); bond(-0.03, SVector(0, 0, 1), zz);
         MultipoleTerm(-0.02, 1, [1], [o], [2], [0.0, 0.0, 1.0, 0.0, 0.0])]
dims = (4, 4, 4)
H = TiledHamiltonian(1, terms; dims)
```

The zero-temperature anisotropy is an energy difference of uniform states, per
cell — the reference the finite-temperature numbers will be compared with:

```@example cmc
n = n_sites(H)                      # 64 sites = 64 cells
uniform(e) = fill(SVector{3,Float64}(e), n)
Kref(Hx) = (total_energy(Hx, uniform((1, 0, 0))) -
            total_energy(Hx, uniform((0, 0, 1)))) / n
H1 = TiledHamiltonian(1, filter(t -> t.body == 1, terms); dims)   # single-ion part
H2 = TiledHamiltonian(1, filter(t -> t.body == 2, terms); dims)   # two-ion part
(K0 = Kref(H), K0_1ion = Kref(H1), K0_2ion = Kref(H2))
```

### A θ scan with lanes

One [`MagnetizationConstraint`](@ref) per direction, in the ``a``–``c`` plane
(``\varphi = 0``; `MagnetizationConstraint(θ, φ)` uses the physics convention
``\hat n = (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)``). A vector
of constraints runs as independent **lanes** over threads, one chain per
constraint. Each direction appears twice: duplicate constraints are independent
replicas (each lane gets its own seed), which is how the error bars below are
checked. The observables are a *function* of the constraint, because `:M_par` and
`:constraint_residual` measure along the direction they were built for; the
partial torques of `H1` and `H2` ride along.

```@example cmc
obs(c) = [cmc_observables(H, c); torque_observable(:torque_1ion, H1, H);
          torque_observable(:torque_2ion, H2, H)]
θs = range(0, π / 2; length = 7)
R = 2                                            # replicas per direction
cons = [MagnetizationConstraint(θ, 0.0) for θ in θs for _ = 1:R]
rs = run_cmc(H, cons; kT = 0.7, sweeps_therm = 500, sweeps_measure = 4_000,
             measure_interval = 5, observables = obs, seed = 1)
rs[7]                                            # one lane at θ = π/4
```

`rs[k]` is a [`CMCResult`](@ref): `points` (one [`TempResult`](@ref) per
temperature), the `constraint` with its weights resolved, the `final_config`,
and the lane's `seed`. The result vector is bitwise the same for any thread
count and any `ntasks`, and lane `k` alone is reproducible with
`run_cmc(H, cons[k]; seed = rs[k].seed, observables = obs(cons[k]), …)`.

### Reading the torque

`:torque` is `total_torque(H, config) / n_cells`
([`SCEMonteCarlo.total_torque`](@ref total_torque), public but unexported): the
full-system torque ``\mathbf T = \sum_s \mathbf G_s \times
\mathbf e_s`` (``\mathbf G`` the energy gradient — Zeeman part included, every
term with unit weight, never the constraint weights), **per cell of `H`**,
Cartesian, in the model's energy unit. Its sign is fixed by a rigid rotation of
every spin by the angle ``\alpha`` about a unit axis ``\mathbf w``:
``\mathbf T\cdot\mathbf w = -dE/d\alpha``. With ``f = F/n_\text{cells}``,

```math
\langle\mathbf T\rangle\cdot\hat e_\varphi = -\frac{\partial f}{\partial\theta},\qquad
\langle\mathbf T\rangle\cdot\hat e_\theta = \frac{1}{\sin\theta}\frac{\partial f}{\partial\varphi},\qquad
\langle\mathbf T\rangle\cdot\hat n = 0,
```

with ``\hat e_\varphi = (-\sin\varphi, \cos\varphi, 0)`` and ``\hat e_\theta =
(\cos\theta\cos\varphi, \cos\theta\sin\varphi, -\sin\theta)``. The projections are
left to you because the polar frame is singular at ``\theta = 0, \pi``. Here
``\varphi = 0``, so ``\hat e_\varphi = \hat y`` and the component we need is
`mean[2]`. Pool the two replicas of each direction (mean of the means,
``\sigma = \sqrt{\sum_r \sigma_r^2}/R``) and check the binning errors against the
replica scatter (``\chi^2`` per degree of freedom ≈ 1 means honest errors):

```@example cmc
function pooled(key)
    m, σ, χ2 = zeros(length(θs)), zeros(length(θs)), 0.0
    for k in eachindex(θs)
        reps = [r.points[1].stats[key] for r in rs[(k - 1) * R + 1:k * R]]
        m[k] = sum(s -> s.mean[2], reps) / R
        σ[k] = sqrt(sum(s -> s.err[2]^2, reps)) / R
        χ2 += sum(s -> ((s.mean[2] - m[k]) / s.err[2])^2, reps)
    end
    return m, σ, χ2 / (length(θs) * (R - 1))
end
τ, στ, χ2 = pooled(:torque)                      # τ(θ) = ⟨T⟩·ê_φ per cell
(τ = round.(τ; sigdigits = 4), σ = round.(στ; sigdigits = 2), χ2_per_dof = χ2)
```

``\tau(0) = \tau(\pi/2) = 0`` within the errors, as the symmetry requires, and
``\tau < 0`` in between: the free energy rises away from the easy axis.

## Extracting the anisotropy

**Uniaxial fit.** For ``f(\theta) = K_1\sin^2\theta + K_2\sin^4\theta + \dots``,

```math
-\tau(\theta) = \sin 2\theta\,\bigl(K_1 + 2K_2\sin^2\theta + \dots\bigr),
```

which is linear in the ``K``s: a weighted least-squares fit with weights
``1/\sigma^2`` (rows divided by ``\sigma``), and the parameter errors from
``(A^\top A)^{-1}``:

```@example cmc
s2 = sin.(2 .* θs)
A = [s2 2 .* s2 .* sin.(θs) .^ 2] ./ στ
K = A \ (-τ ./ στ)
Kerr = sqrt.(diag(inv(A' * A)))
(K1 = K[1], K1_err = Kerr[1], K2 = K[2], K2_err = Kerr[2])
```

**One-point estimate.** At ``\theta = \pi/4`` the formula reduces to
``-\tau(\pi/4) = K_1 + K_2`` (plus higher terms) — a single direction gives the
effective constant, which is what the Callen–Callen-type scaling plots usually
show. Here θs[4] = π/4:

```@example cmc
(one_point = -τ[4], err = στ[4], K1_plus_K2 = K[1] + K[2])
```

**Free energy by integration**, as a consistency check:
``f(\theta) - f(0) = -\int_0^\theta \tau\,d\theta'`` (Gong et al. 2019, Eq. (8));
over ``[0, \pi/2]`` it should equal ``K_1 + K_2 + \dots``. Simpson on the seven
nodes:

```@example cmc
wS = [1, 4, 2, 4, 2, 4, 1] .* (step(θs) / 3)
(ΔF = -sum(wS .* τ), err = sqrt(sum((wS .* στ) .^ 2)))
```

Fitting the torques directly is more precise than integrating them (Ellis et al.
2017) and needs no grid, so treat the integral as the check, not the estimate.

At ``k_BT = 0.7|J|`` (about half the ordering temperature) the effective constant
is about two thirds of its zero-temperature value `K0`. The small ``K_2`` is not
interpreted here: 64 spins carry finite-size effects (see the practical notes
below).

**General crystals.** Expand ``F(\hat n) = \sum_{l\,\text{even},\,m}
\kappa_{lm}\,Z_{lm}(\hat n)`` over the combinations the point group allows; the
torque ``\langle\mathbf T\rangle = -\sum \kappa_{lm}\,\hat n\times\nabla
Z_{lm}(\hat n)`` is again linear in the ``\kappa_{lm}``, so stack both tangent
components of the torque at directions covering the irreducible wedge (polar
**and** azimuthal) and solve one weighted least-squares problem. Note that the
Callen–Callen laws (``\kappa_l \sim m^{l(l+1)/2}`` for single-ion terms) hold for
the spherical-harmonic coefficients, not for ``K_1``: for ``F = \kappa_2 P_2 +
\kappa_4 P_4``, ``K_1 = -\tfrac32\kappa_2 - 5\kappa_4`` and ``K_2 =
\tfrac{35}{8}\kappa_4``, so ``K_1`` mixes two powers of ``m`` whenever ``K_2 \neq
0``. The fits are post-processing and not part of the package; the derivations
are in the spec's research notes (`docs/specs/261121-constrained-mc/research/`,
Japanese).

## Partial torques: single-ion vs two-ion

[`torque_observable`](@ref)`(name, Hpart, H)` measures the torque of a
term-subset Hamiltonian `Hpart` per cell of `H`. Build each `Hpart` with the
public constructor from the **same term list** that built `H`, without a field —
`TiledHamiltonian(n_atoms, filter(pred, terms); dims)` with `terms =
multipole_terms(model)`, or `red.terms` / `red.n_atoms` after
[`reduce_cell`](@ref). `torque_observable` refuses a subset of any other term
list (another cell, another fit — every template of `Hpart` must equal a fitted
template of `H`), a field-carrying `Hpart`, or a mismatch in `dims`, and the
observable is bound to the `H` object it was built with. Over a disjoint
cover of the fitted terms the partial torques sum to the total, configuration by
configuration (plus the analytic `:torque_zeeman` when `H` carries a field):

```@example cmc
τ1, σ1, _ = pooled(:torque_1ion)
τ2, σ2, _ = pooled(:torque_2ion)
m = sum(r -> r.points[1].stats[:M_par].mean[1], rs) / length(rs)
(sum_rule = maximum(abs.(τ1 .+ τ2 .- τ)),
 single_ion = (ratio = -τ1[4] / Kref(H1), m3 = m^3),
 two_ion = (ratio = -τ2[4] / Kref(H2), m2 = m^2))
```

The single-ion share scales roughly like ``m^3`` and the two-ion share like
``m^2`` — the classical expectations for an ``l = 2`` single-ion term and a
bilinear two-ion term.

Three things to know:

- **Disjointness and coverage are not enforced.** A subset that is empty (for
  example `t.body ≥ 3` on a pair model) cannot be built at all —
  `TiledHamiltonian` throws an `ArgumentError` for an empty term list — so skip
  that part instead.
- **Interpretation is first-order.** The decomposition of ``K`` into single-ion
  and two-ion shares is unique only to first order in the anisotropy: beyond it,
  each ``\langle\mathbf T_c\rangle`` differs from a free-energy gradient by
  covariance terms between the parts, so an individual partial torque is not the
  gradient of a "partial free energy". The sum is always exact.
- **Cost.** Every torque observable is one full energy gradient per measurement
  (about one sweep); see `measure_interval` below.

## Choosing the weights

The constrained magnetization is ``\mathbf M = \sum_{s\in\mathcal C}
w_{a(s)}\,\mathbf e_s`` over the active sites whose cell atom has a nonzero
weight. `weights` is `:auto` (the moments `magmoms` of `H` when it has them, else
all ones), `:uniform`, `:magmoms`, or an explicit per-cell-atom vector in the
atom order of `H` (the **reduced** cell's order after [`reduce_cell`](@ref)).
Explicit weights may be negative (an antiparallel sublattice enters with its sign
flipped) or zero (that sublattice is not constrained and moves by plain
single-spin Metropolis — the *hybrid* rule of Ellis et al.). Which ``\mathbf M``
you constrain decides which free energy you get.

- **A ferromagnet**: `:auto` or `:uniform` — the net magnetization.
- **FePt with an independent small-moment Pt** (Pt spins as their own degrees of
  freedom, ``m_\text{Pt}/m_\text{Fe} \approx 1/8``): `:auto` constrains the net
  moment (the weighted compensation of Toga et al.). It is correct, but pairs
  with a large weight ratio are mostly null moves (a small Pt spin cannot absorb
  a Fe rotation), so the adapted step can collapse — watch `acceptance_metropolis`
  and the step-floor warning. The alternative is to constrain Fe only — an
  explicit vector with `m_Fe` on the Fe atoms and `0.0` on the Pt atoms, in the
  atom order of `H` — with Pt as hybrid sites: that is the
  free energy of the **Fe** magnetization direction, with the Pt spins
  equilibrating around it.
- **Ferrimagnets, Nd₂Fe₁₄B**: the moments are non-negative and the directions
  carry the sign, so `:auto` constrains the **net** moment (antiparallel
  sublattices partly cancel). Signed weights (``-m`` on the antiparallel
  sublattice) constrain the sublattice-aligned order parameter instead; the two
  agree at ``T = 0`` and differ at finite temperature. Toga et al. weight every
  sublattice; the hybrid alternative constrains Fe only (`w = 0` on Nd), whose
  free energy is that of the Fe direction. On the package's Nd₂Fe₁₄B smoke run
  (synthetic anisotropy) the two choices agreed within the errors, but the
  difference between the full and the Fe-only free energy is physics near a
  spin-reorientation transition — decide which one your question is about. Note
  that DFT spin moments as weights are not the 4f total moments.
- **Antiferromagnets and compensation points**: the net moment nearly vanishes,
  so a positive-weight constraint is weak — the run warns when
  ``M_\parallel/\sum|w| < 0.05`` at the end of thermalization. Give the
  antiparallel sublattice a negative weight (``\mathbf M`` becomes the Néel
  vector), or make it a hybrid sublattice. The weights are **per cell atom**: a
  Néel vector can be written only when the two sublattices are different atoms of
  the cell of `H`. If the magnetic cell is larger than the chemical cell (e.g. a
  G-type order on a one-atom primitive cell), the weights cannot express it; use
  a model whose cell contains the magnetic cell (per-site weights are not part of
  this version).

## Practical advice

- **Start from the ground state** when sublattices are antiparallel. Without
  `init` every constrained site starts along ``\operatorname{sign}(w)\,\hat n``
  (and every other site along ``\hat n``) — a ferromagnetic start for `:auto`
  weights, which is correct but thermalizes slowly. Pass a ground state; it is
  rotated rigidly so that its weighted magnetization points along each lane's
  ``\hat n``:

  ```julia
  gs = find_ground_state(H; nstarts = 8, seed = 1)
  rs = run_cmc(H, cons; temperature = 300.0, init = gs.config, seed = 2)
  ```

- **Thin the torque with `measure_interval`.** Each torque measurement costs one
  energy gradient, about one sweep; partial torques add one each. Consecutive
  sweeps are strongly correlated anyway, so a `measure_interval` of a few sweeps
  loses little statistics and saves most of the time.
- **Production error bars come from replicas.** The log-binning error of a single
  chain can be optimistic: on the Nd₂Fe₁₄B smoke run the replica scatter showed
  binning errors about 20 % too small at 6000 measurements. Run duplicate
  constraints (a few per direction), pool them, and check the replica ``\chi^2``
  as above.
- **``F`` is a finite-``N`` quantity.** It is the free energy of the direction of
  ``\mathbf M`` of *this* finite system; its measure-dependent part is of order
  ``k_BT/N`` per spin. For bulk numbers, run two or more sizes and extrapolate.
- **Temperature ladders.** `temperature` / `kT` may be a collection, run in the
  given order with the chain carried over (an annealing ladder, high → low) or
  restarted from the constrained start with `carryover = false` (deterministic,
  no RNG consumed).
- **Checkpoint long scans.** With `checkpoint = "scan.jld2"` lane `k` writes
  `scan.lane<k>.jld2`, an ordinary constrained-run file holding the lane's
  constraint and seed. A lane that fails does not stop the others (the failures
  are thrown together at the end, in lane order), and every completed lane's file
  returns its result through [`resume`](@ref)`(file, H)`; resume the unfinished
  lanes one by one — there is no resume of the whole set. Files from an earlier
  call with the same stem are not removed. See [checkpointing](checkpointing.md)
  and [parallelism](parallelism.md).
- **Warnings.** At the end of every temperature's thermalization the run warns
  when the constraint is weak (above the ordering temperature, or cancelling
  weights) and when the adapted step sits at its lower clamp (many null moves —
  a large weight ratio). Each warning names `kT` and ``\hat n``.

## Not supported

Constrained replica exchange, `sweep_tasks` (the CMC sweep is serial — the pair
moves couple through ``M_\parallel``) and overrelaxation (`or_per_metropolis`;
both keywords throw), GPU sampling, block or per-site constraints, a constraint
on ``|\mathbf M|``, and the post-processing fits. One call samples one
constraint per lane; for many directions use the lane method.

## References

- P. Asselin et al., *Constrained Monte Carlo method and calculation of the
  temperature dependence of magnetic anisotropy*, Phys. Rev. B **82**, 054415
  (2010), arXiv:1006.3507 — the pair move, its acceptance rule, and the torque
  identity.
- Y. Toga et al., arXiv:1606.00333 — weighted compensation for unequal moments
  (Nd₂Fe₁₄B).
- M. O. A. Ellis et al., arXiv:1706.05188 — the hybrid (partly constrained)
  scheme and direct torque fits.
- Q. Gong et al., arXiv:1902.05636 — Nd₂Fe₁₄B anisotropy from torque integration
  and ``K_1``–``K_3`` fits.

Notes on each are in the repository's `references/` directory.

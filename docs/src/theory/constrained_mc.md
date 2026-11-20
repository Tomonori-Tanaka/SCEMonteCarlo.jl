# Theory: constrained Monte Carlo

```@meta
CurrentModule = SCEMonteCarlo
```

The sampler behind [`run_cmc`](@ref) (Asselin et al., Phys. Rev. B **82**, 054415
(2010); weighted compensation after Toga et al., arXiv:1606.00333; the hybrid rule
after Ellis et al., arXiv:1706.05188). The standing contract, with every
convention, door, and gate, is the decision record
[`docs/specs/constrained-mc.md`](https://github.com/Tomonori-Tanaka/SCEMonteCarlo.jl/blob/main/docs/specs/constrained-mc.md)
(D1–D5 below); usage is in the [guide](../guide/constrained_mc.md).

## The constrained ensemble (D1)

Each cell atom ``a`` carries a signed weight ``w_a``. The constrained sites are
``\mathcal C = \{s : s \text{ active},\ w_{a(s)} \neq 0\}`` and the constrained
magnetization is

```math
\mathbf M = \sum_{s\in\mathcal C} w_{a(s)}\,\mathbf e_s,\qquad
M_\parallel = \mathbf M\cdot\hat n,\qquad
\mathbf M_\perp = \mathbf M - M_\parallel\,\hat n .
```

The sampled measure, defined exactly at finite ``N``, is the canonical one
restricted to ``\hat{\mathbf M} = \hat n`` by the delta function **on the
sphere**,

```math
\pi(d\xi) \propto \prod_s d\Omega_s\; e^{-\beta E(\xi)}\;
\delta_{S^2}\bigl(\hat{\mathbf M}(\xi), \hat n\bigr),\qquad
\delta_{S^2}(\hat{\mathbf M}, \hat n) = |\mathbf M|^2\,\delta^2(\mathbf M_\perp)\,
\Theta(M_\parallel).
```

The factor ``|\mathbf M|^2 = M_\parallel^2`` is the Jacobian between the
transverse components of ``\mathbf M`` and the solid angle of its direction.
With it, the normalization of ``\pi`` is the canonical probability density
``P(\hat n)`` of ``\hat{\mathbf M}`` per solid angle, and

```math
F(\hat n) = -k_BT \ln P(\hat n) + \text{const}
```

is the free energy of the magnetization direction of this finite system (Asselin's
``F``). Dropping the factor gives another ensemble whose free energy differs by
``O(k_BT/N)`` per spin; that measure-dependent part is why ``F`` is a finite-size
quantity to be extrapolated in ``N``. Inactive sites are excluded from
``\mathbf M`` and from every observable, and stay frozen.

## The weighted pair move (D2)

A move picks a source ``i\in\mathcal C`` and a partner ``j \in \mathcal C
\setminus \{i\}``, rotates ``\mathbf e_i \to \mathbf e_i'`` by a symmetric
Rodrigues proposal (uniform axis, angle `step · randn`, no flips), and lets
``\mathbf e_j`` restore ``\mathbf M_\perp``:

```math
\mathbf e_{j\perp}' = \mathbf e_{j\perp} - \frac{w_i}{w_j}\bigl(\mathbf e_{i\perp}' -
\mathbf e_{i\perp}\bigr),\qquad
e_{j\parallel}' = \sigma_j\sqrt{1 - |\mathbf e_{j\perp}'|^2},\qquad
\sigma_j = \operatorname{sign} e_{j\parallel} .
```

It is a **null move** — rejected, the state unchanged — when
``|\mathbf e_{j\perp}'| \ge 1`` (the compensating spin would leave the sphere) or
when ``M_\parallel' \le 0`` (the magnetization would cross into the other
hemisphere). Otherwise it is accepted with probability ``\min(1, e^{\ell})``,

```math
\ell = -\beta\,\Delta E + 2\ln\frac{M_\parallel'}{M_\parallel}
     + \ln|e_{j\parallel}| - \ln|e_{j\parallel}'| ,
```

where ``\Delta E`` is the exact joint energy change of the two spins, computed by
two sequential single-site differences (the second one against the tabulated
harmonics with ``\mathbf e_i'`` already in place), so it is exact for any body
order even when ``i`` and ``j`` share a cluster.

**Derivation.** Hold every spin except ``i`` and ``j`` fixed. The constraint
fixes ``w_j\mathbf e_{j\perp} = \mathbf C - w_i\mathbf e_{i\perp}`` for a constant
``\mathbf C``, so ``\mathbf e_i`` is free on its sphere and ``\mathbf e_j`` is
determined up to the hemisphere ``\sigma_j``. Integrating ``\pi`` over
``\mathbf e_j`` uses the projection of the sphere onto the plane
``\perp \hat n``, ``d\Omega_j = d^2 e_{j\perp}/|e_{j\parallel}|`` (on each
hemisphere), and ``\delta^2(w_j\mathbf e_{j\perp} - \dots) =
\delta^2(\mathbf e_{j\perp} - \dots)/w_j^2``. The density of the pair state on
this fiber, with respect to ``d\Omega_i``, is therefore

```math
\rho(\mathbf e_i) \propto e^{-\beta E}\;\frac{M_\parallel^2}{w_j^2\,|e_{j\parallel}|}.
```

The rotation of ``\mathbf e_i`` is symmetric and the compensation is a
deterministic function of ``\mathbf e_i'`` that keeps ``\sigma_j``, so the reverse
move (same ordered pair) has the same proposal density, and the partner is drawn
with a state-independent probability. The Metropolis ratio is the ratio of fiber
densities: ``w_j^2`` cancels and ``\ell`` above follows. Null moves are exactly the
proposals that land where ``\rho = 0``.

**Null moves must be counted.** A null move is an ordinary rejected attempt: the
chain stays put and the attempt counts. Re-drawing a new proposal until one is
geometrically valid instead turns the kernel into ``K(x\to y)/(1 - r(x))``, with
``r(x)`` the null-move probability from ``x``; its stationary distribution is
``\pi(x)\,(1 - r(x))``, not ``\pi`` — a bias the package's investigation measured
at 8–13σ on three spins. Counting them also means the acceptance that drives the
thermalization-only step adaptation includes the null moves, and
`acceptance_metropolis` of a constrained run is the accepted fraction of all
attempts.

**The partner rule.** The partner is uniform over ``\mathcal C\setminus\{i\}``.
Restricting it to a partner of the same weight would conserve the transverse sum of
each weight class separately and break ergodicity.

**Signed weights — the gauge map.** With ``\tilde{\mathbf e}_s =
\operatorname{sign}(w_s)\,\mathbf e_s``, ``\mathbf M = \sum_s |w_s|\,\tilde{\mathbf
e}_s``: the signed problem maps exactly onto the ``|w|`` problem. The compensation
(the ratio ``w_i/w_j`` carries the relative sign), the hemisphere ``\sigma_j``, the
Jacobian (``|e_{j\parallel}|`` is unchanged by a sign flip), and the symmetry of the
proposal all commute with the map, so negative weights need no special case: an
antiparallel sublattice with a negative weight is the ``|w|`` problem in a relabeled
frame.

## The sweep and the hybrid rule (D3)

One sweep visits the active sites in site order: a pair attempt with source ``s``
for ``s \in \mathcal C``, a plain single-spin Metropolis attempt (flip proposal
included) for an active site with ``w = 0``. A weight-0 site does not enter
``\mathbf M``, so moving it leaves the constraint intact, and its single-site
kernel preserves ``\pi`` (its conditional density is just ``e^{-\beta E}``): the
**hybrid** rule — part of the system constrained, the rest free. A fixed-order
composition of ``\pi``-preserving kernels preserves ``\pi``.

**Why the sweep is serial.** The single-spin sweeps update a colour class in
parallel because kernels of sites that share no cluster commute. Pair moves never
do: every acceptance reads the global ``M_\parallel``, which every accepted pair
changes, and the partner can be any site of ``\mathcal C``. So the CMC sweep runs
on one task (`sweep_tasks` is refused); the parallelism is across directions
(lanes). Every attempt draws only from its source site's own random stream, in a
fixed order (partner, rotation axis, angle, and the accept uniform only when
needed), so a run is bit-reproducible for a fixed seed and independent of the
thread count.

## The torque identity (D5)

The torque of a configuration is ``\mathbf T = \sum_{s\,\text{active}}
\mathbf G_s\times\mathbf e_s = -\sum_s \mathbf e_s\times\partial E/\partial\mathbf
e_s``, with sign fixed by a rigid rotation of every spin by ``\alpha`` about the
unit axis ``\mathbf w``: ``\mathbf T\cdot\mathbf w = -dE/d\alpha``. Every term
counts with unit weight — the weights shape the constraint, not the energy.

Rotate every spin rigidly by ``R_\alpha`` (angle ``\alpha`` about ``\mathbf w``).
The product measure is invariant and ``\hat{\mathbf M}`` rotates with the spins,
so substituting ``\xi \to R_\alpha\xi`` gives
``P(R_\alpha\hat n)/P(\hat n) = \langle e^{-\beta[E(R_\alpha\xi) -
E(\xi)]}\rangle_{\hat n}`` and, differentiating at ``\alpha = 0``,
``\frac{d}{d\alpha}F(R_\alpha\hat n) = \langle dE/d\alpha\rangle =
-\langle\mathbf T\rangle\cdot\mathbf w``. With
``dR_\alpha\hat n/d\alpha = \mathbf w\times\hat n``, for every axis ``\mathbf w``,

```math
\langle\mathbf T\rangle = -\hat n\times\frac{\partial F}{\partial\hat n},
```

exact at finite ``N`` and independent of the weights (they enter only through
which ``\hat{\mathbf M}`` is constrained). In the polar frame
``\hat e_\theta = \partial\hat n/\partial\theta``,
``\hat e_\varphi = (-\sin\varphi, \cos\varphi, 0)``:

```math
\langle\mathbf T\rangle\cdot\hat e_\varphi = -\frac{\partial F}{\partial\theta},\qquad
\langle\mathbf T\rangle\cdot\hat e_\theta = \frac{1}{\sin\theta}\frac{\partial F}{\partial\varphi},\qquad
\langle\mathbf T\rangle\cdot\hat n = 0 .
```

The observables report per cell of `H` (``\mathbf T/n_\text{cells}``, the gradient
of ``F/n_\text{cells}``). A field contributes the analytic Zeeman torque
``\mu_B(\mathbf M_\mu\times\mathbf B)``, ``\mathbf M_\mu = \sum m_{a(s)}\mathbf
e_s`` (a moment is turned toward ``+\mathbf B``).

**Partial torques.** For a disjoint split of the fitted terms, the torques of the
term-subset Hamiltonians sum to the total configuration by configuration (plus the
Zeeman part). Each partial torque alone, ``\langle\mathbf T_c\rangle``, equals
``-\hat n\times\partial F_c/\partial\hat n`` for a well-defined share ``F_c`` only
to first order in the anisotropy: beyond it, the constrained average of one part
picks up covariance terms with the others, so the single-ion / two-ion split of
``K`` is unique only in the ``O(K)`` limit.

**Linear response.** To first order in a single-ion term
``\sum_s v(\mathbf e_s)`` the anisotropy is the isotropic average over the
fluctuations of each spin about ``\hat{\mathbf M}``; for ``v \propto P_l`` this is
``K_\text{eff}/K = \langle P_l(\mathbf e\cdot\hat{\mathbf M})\rangle_\text{iso}``,
the classical Callen–Callen result (``\sim m^{l(l+1)/2}`` at low temperature). The
package checks its sampler against this identity, against direct histograms of
``\hat{\mathbf M}`` from unconstrained runs, and against rotated-iid references
for three- and four-body models (the "Manual validation" section of the decision
record).

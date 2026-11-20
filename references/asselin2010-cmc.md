# Asselin 2010 — Constrained Monte Carlo for the temperature dependence of anisotropy

- **Citation**: P. Asselin, R. F. L. Evans, J. Barker, R. W. Chantrell, R. Yanes,
  O. Chubykalo-Fesenko, D. Hinzke, U. Nowak, *Constrained Monte Carlo method and
  calculation of the temperature dependence of magnetic anisotropy*,
  Phys. Rev. B 82, 054415 (2010).
- **DOI / URL**: 10.1103/PhysRevB.82.054415; arXiv:1006.3507
- **Local PDF**: `papers/asselin2010.pdf` (gitignored)
- **Tags**: constrained Monte Carlo, magnetic anisotropy, torque, Callen–Callen

## Summary

Fixes the direction of the total magnetization with a two-spin move (a primary
spin moves freely, a compensating spin restores the transverse components) and
a Metropolis acceptance carrying the projection Jacobian
`(M_z'/M_z)² |S_jz|/|S_jz'|`. The mean internal torque equals
`−M̂ × ∂F/∂M̂`, so the free-energy landscape follows by integrating or fitting
torques. Reproduces the Callen–Callen `M³` (uniaxial) and `M¹⁰` (cubic) laws and
the `M^2.1` law of L1₀ FePt.

## Relevance to this package

The sampler of `docs/specs/constrained-mc.md` (M1–M3) and the torque identity
(M5). Page references used there: acceptance rule p. 7 step 7 and Eq. (A11)
(arXiv version); torque identity App. C, Eq. (C15).

## Notes

- The `M_z²` factor corresponds to constraining with a delta function on the
  sphere; dropping it gives a different (also torque-consistent) ensemble that
  differs by O(kT/N) per spin.
- Kinematic rejections (negative square-root argument, `M_z' ≤ 0`) must be
  counted as null moves.
- App. B proves ergodicity for equal moments only.
- Fig. 3's x-axis unit "[K]" is a typo for radians; the sign of Fig. 1's torque
  depends on the (unstated) rotation plane.

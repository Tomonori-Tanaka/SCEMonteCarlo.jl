# Toga 2016 — Weighted constrained Monte Carlo for Nd₂Fe₁₄B

- **Citation**: Y. Toga et al., *Monte Carlo analysis for finite-temperature
  magnetism of Nd₂Fe₁₄B permanent magnet* (journal version not checked here).
- **DOI / URL**: arXiv:1606.00333
- **Local PDF**: `papers/toga2016.pdf` (gitignored)
- **Tags**: constrained Monte Carlo, ferrimagnet, unequal moments

## Summary

Applies constrained Monte Carlo to a multi-sublattice permanent magnet with
unequal moments; the compensating spin absorbs `(m_i/m_j)` times the primary
spin's transverse change, with the acceptance rule otherwise unchanged.

## Relevance to this package

The weighted compensation rule of `docs/specs/constrained-mc.md` M2.

## Notes

The procedure's "return to (A)" after a kinematic failure is ambiguous about
counting the failure as an attempt; this package counts it (null move). Its
"M′ < 0" is read as `M_∥′ ≤ 0`.

# Ellis 2017 — Hybrid constrained Monte Carlo and direct torque fitting

- **Citation**: M. O. A. Ellis et al., *Manifestation of higher-order
  inter-granular exchange in magnetic recording media*.
- **DOI / URL**: arXiv:1706.05188
- **Local PDF**: `papers/ellis2017.pdf` (gitignored)
- **Tags**: constrained Monte Carlo, hybrid constraint, VAMPIRE

## Summary

Uses VAMPIRE's hybrid constrained Monte Carlo (only a subset of spins is
constrained, the rest move by ordinary Metropolis) with per-grain constraints,
and reports that fitting torques directly is more precise than integrating them
to a free energy.

## Relevance to this package

The hybrid rule (active zero-weight sites move by single-site Metropolis) in
`docs/specs/constrained-mc.md` M3, and the guide's recommendation to fit torques
directly.

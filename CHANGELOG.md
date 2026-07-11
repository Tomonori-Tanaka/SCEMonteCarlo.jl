# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `run_mc`: single-temperature and warm-started ladder (annealing) runs of
  single-spin Metropolis with the exact `ΔE = c_s·ΔZ` kernel, symmetric
  flip+Rodrigues proposal, thermalization-only adaptive step (frozen during
  measurement), periodic renormalize + energy re-anchoring with drift tracking,
  and bit-reproducible seeding. Results as `MCResult` / `TempResult` with a
  summary-table printer.
- Composable measurement layer: `Observable` / `Evaluable` with the standard set
  (`E`, `E²`, `m`, `|m|`, `m²`, `m⁴`, per-sublattice magnetization) and derived
  `C/k_B`, |m|-connected `χ`, Binder `U = ⟨m⁴⟩/⟨m²⟩²` (conventions:
  `docs/specs/binning-observables.md`).
- Error analysis: streaming `LogBinner` (log-binning plateau errors + `τ_int`,
  O(levels) memory), `BinStore` + leave-one-bin-out `jackknife` for derived
  quantities.
- `TiledHamiltonian`: the fitted SCE unfolded onto an `N₁×N₂×N₃` supercell from the
  public `multipole_terms` introspection (per-site integer `shifts`, toroidal wrap),
  with template-once + CSR-instance memory layout and the `(4π)^(body/2)` scale
  applied exactly once. Supports self-image (`AllImages`) clusters when `dims` keeps
  the images distinct sites.
- The 4-function energy contract: `total_energy`, `site_coeffs!` (leave-one-out
  coefficients — exact single-spin `delta_energy` for any body order), and
  `site_gradient` (on-sphere, via `Harmonics.grad_Zlm_unsafe`).
- Package scaffold: module skeleton, temperature control (`KB_EV`, `resolve_kt` —
  kelvin XOR model-energy-unit keywords), test harness (`TEST_MODE`
  default/all/unit/aqua/jet with Aqua + JET), Documenter docs skeleton.

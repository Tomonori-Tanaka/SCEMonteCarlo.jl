# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

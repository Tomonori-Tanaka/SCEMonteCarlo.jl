# SCEMonteCarlo.jl

Classical spin Monte Carlo for fitted SCE (symmetry-adapted cluster expansion)
models from [SCEFitting.jl](https://github.com/Tomonori-Tanaka/SCEFitting.jl).

- **Supercell tiling** — replicate the fitted training-cell Hamiltonian onto an
  `N₁ × N₂ × N₃` supercell from the public `multipole_terms` introspection alone.
- **Cell reduction** — re-express a supercell-fitted model in a user-chosen smaller
  cell (`reduce_cell`, verified — structure and couplings must actually have that
  periodicity), so MC sizes are not locked to training-cell multiples.
- **Updates** — single-spin Metropolis with an adaptive proposal step, plus
  overrelaxation sweeps (involutive reflection + Metropolis correction, exact for
  any body order).
- **Runs** — single temperature, warm-started annealing sweeps (`run_mc`), and
  replica exchange over threads (`run_pt`), bit-reproducible for a fixed seed.
- **Observables** — energy, specific heat, `|m|`, susceptibility, Binder cumulant,
  per-sublattice magnetization, and user-defined observables/evaluables, with
  autocorrelation-aware log-binning errors and jackknifed derived quantities.
- **Checkpoint/restart** — versioned JLD2 schema; a resumed run is bit-identical
  to an uninterrupted one.

Temperatures are absolute, under exactly one of two keywords: `temperature`
(kelvin) or `kT` (the model's energy units).

```julia
using SCEMonteCarlo, SCEFitting

model = SCEFitting.load(SCEPredictor, "model.toml")
H = TiledHamiltonian(model; dims = (4, 4, 4))

result = run_mc(H; temperature = [1200, 900, 600, 300],   # annealing ladder
                sweeps_therm = 2_000, sweeps_measure = 20_000, seed = 1)

pt = run_pt(H; temperature = range(200, 1400; length = 16), seed = 1)
```

Development status: v0. See `SPEC.md` and `docs/` for the architecture and
decision records.

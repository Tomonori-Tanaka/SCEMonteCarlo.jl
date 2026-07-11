# Parallel tempering

```@meta
CurrentModule = SCEMonteCarlo
```

[`run_pt`](@ref) runs one chain (**lane**) per rung of a strictly monotone
temperature ladder, all sweeping concurrently over threads (start Julia with
`-t N`). Every `exchange_interval` sweeps, adjacent rungs attempt to swap their
configurations with probability `min(1, exp((βᵢ−βⱼ)(Eᵢ−Eⱼ)))` — so a
low-temperature rung keeps escaping metastable basins by round-tripping through
the hot end. This is the tool for the broken-ergodicity failure mode where
independent chains freeze into different basins while each reports small,
confident error bars.

```julia
pt = run_pt(H; temperature = range(250, 1300; length = 16),
            exchange_interval = 10, seed = 1)
pt.points[1]           # coldest rung's TempResult (points follow ladder order)
pt.swap_acceptance     # length nrungs−1, the ladder diagnostic
```

## Choosing the ladder — the size scaling matters

A swap accepts when `(βᵢ−βⱼ)(Eᵢ−Eⱼ) = O(1)`. Since `E` is extensive and adjacent
energies differ by `≈ C·ΔT` per site, the acceptance scales like
`exp(−n_sites · C · (ΔT/T)²)` — **the rung count must grow like
`√(n_sites · C)`** for a fixed temperature span. A ladder that works on the
training cell can be uselessly sparse on a 4×4×4 supercell: in the Nd₂Fe₁₄B
smoke test (4352 sites), 8 rungs over 250–1300 K produced *zero* accepted swaps —
correct physics, not a bug. Watch `swap_acceptance`, aim for O(0.2–0.5) per pair,
and tighten the ladder (geometric spacing is a good start) where it collapses —
typically around the specific-heat peak.

`exchange_interval` sets the sweeps between attempts; ~10 is a reasonable
default (frequent enough to profit, cheap against the sweep cost).

## Semantics worth knowing

- **Lane = fixed temperature.** Exchanges swap the chain *payload*
  (configuration + energy); RNG, adapted step, and accumulators stay with the
  lane, so `points[r]` is directly the equilibrium physics at `kts[r]`.
- Exchanges run during **thermalization and measurement alike**.
- Step adaptation is per-lane and thermalization-only, as in [`run_mc`](@ref).
- **Determinism**: results are bit-identical for a fixed seed regardless of
  `ntasks` and the thread count — every random decision has a dedicated RNG whose
  consumption order is fixed by the segment schedule
  (`docs/specs/pt-threads-determinism.md`).
- `init` seeds every lane (default: independent random starts).

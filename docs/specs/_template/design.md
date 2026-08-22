# Design: <title>

Status: draft (YYYY-MM-DD)

## Summary

<!-- 1-3 paragraphs describing the chosen design. -->

## Module layout

| Target | Change |
|---|---|
| `src/<file>.jl` | <!-- e.g., add `_foo!` to the 4-function contract --> |

## API

```julia
# Example
function new_fn(st::ChainState, H::TiledHamiltonian; opt::Bool = false)::Bar
```

## Types and conventions

<!-- Impact on physics conventions, units, determinism contracts. New
     invariants. A new `ChainState` / `GPUChainState` field is classified in
     the `_swap_payload!` partition tests and the checkpoint schema. -->

## Impact on coupled sites

<!-- Which "Coupled code sites" in CLAUDE.md does this touch? -->

- [ ] `hamiltonian.jl` ↔ SCEFitting introspection contract:
- [ ] `energy.jl` 4-function contract ↔ `updates.jl`:
- [ ] `lm_index` ↔ `zlm_row!` ↔ OR axis:
- [ ] `reduce.jl` ↔ tiling ↔ `geometry.jl` ordering:
- [ ] Gradient kernels (`_site_grad` / `site_gradient` / `energy_gradient!` / `minimize.jl`):
- [ ] Checkpoint writer ↔ reader ↔ `checkpoint-schema.md`:
- [ ] Coloring ↔ sweeps ↔ `updates-stationarity.md`:
- [ ] Device row / kernel / gradient ↔ host references ↔ upstream recursions:
- [ ] CPU PT ↔ GPU PT:
- [ ] Inactive-site convention:
- [ ] `.claude/agents/` references:
- [ ] `SPEC.md` / `docs/src/api.md` / guide pages:

## Test strategy

<!-- For every new gate name the ORACLE (exact identity / hand calculation /
     labeled pin; for statistics the measured σ and the mutation size). -->

## Risks and open items

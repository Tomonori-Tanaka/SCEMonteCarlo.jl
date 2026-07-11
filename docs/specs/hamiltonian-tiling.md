# Decision record — supercell tiling and the Hamiltonian memory layout

Status: landed (M1). Owner: `src/hamiltonian.jl`, `src/energy.jl`;
gates in `test/unit/test_hamiltonian.jl`, `test/unit/test_energy.jl`.

## T1 — unfold `MultipoleTerm.shifts`, one instance per (term, cell)

`multipole_terms(model)` emits every directed cluster member once, with per-site
integer **training-cell** lattice translations `shifts` (`shifts[1] = 0` anchored —
verified in SCEFitting: orbit members retain the anchored candidate-cluster shifts).
Tiling onto `dims = (N₁, N₂, N₃)` is pure integer bookkeeping:

```
for every supercell cell t and every term:
    member i sits at site_index(atoms[i], mod.(t + shifts[i], dims))
```

Toroidal (periodic) boundary conditions; no ½ / 1/N factors (the introspection
contract already makes terms plain summands, `Σ terms = predict_energy − intercept`).
Site indexing is atom-fastest, cells column-major:
`site = atom + n_cell_atoms·(c₁ + N₁(c₂ + N₂c₃))`, so `site_atom(s) = mod1(s,
n_cell_atoms)` is the sublattice id.

Consequences pinned by machine-precision gates:

- `dims = (1,1,1)` degenerates to the training cell exactly:
  `total_energy == predict_energy − intercept` (1e-12).
- A periodically replicated configuration has exactly `prod(dims) ×` the cell energy.
- Tiling **replicates** the fitted finite-range couplings; it does not manufacture
  longer-range physics the training cell could not resolve.

## T2 — distinct member *sites*, not distinct atoms

The exact single-spin ΔE (`ΔE = c_s·ΔZ`) needs each instance to touch a site at most
once, so `site_coeffs!`'s leave-one-out vector is independent of that site's spin.
The invariant is checked per term at the **site level under the actual `dims`**:
`(atomᵢ, mod.(shiftsᵢ, dims))` pairwise distinct (the wrapped relative pattern is
cell-independent, so cell 0 covers all instances).

- Minimum-image fitted models (the default) have distinct atoms per cluster outright
  (SCEFitting drops reused-atom clusters in enumeration) → valid for any `dims`.
- `AllImages` (spin-spiral) models may legitimately couple an atom to its own
  periodic image (`atoms = [a, a]`, `shifts = [0, R]`). These tile fine when `dims`
  keeps the images distinct sites, and are **rejected with a clear error** when the
  wrap folds them together (e.g. `dims = (1,1,1)`) — enlarging `dims` is the fix.
  Gate: the hand-built ±x self-image chain (`test_hamiltonian.jl`).

## T3 — memory: templates once + compact CSR instances

`ScaledTerm` templates hold each fitted term's `folded` payload **once** (with the
`(4π)^(body/2)` scale applied there — the package's single application site);
instances are integer CSR lists (`inst_term`, `inst_ptr`/`inst_sites`) plus a
per-site adjacency (`site_ptr`/`site_inst`/`site_slot`, and a `site_has_l1` mask for
the overrelaxation axis). This is the SpinClusterMC lesson: duplicating per-instance
payloads is what blew that package to multi-GB caches; the index-only layout costs
≈ 13 MB for the Nd₂Fe₁₄B l02 case (4692 terms, 4×4×4, 4352 sites, 300k instances).

Rejected alternative — fully on-the-fly instance reconstruction (no instance list):
saves the MB but puts mod-arithmetic and shift resolution inside the innermost ΔE
loop, and is much harder to test in isolation. The CSR adjacency also precomputes
each site's member slot, replacing the `findfirst` of the SCETools single-cell
kernel.

## T4 — the 4-function energy contract

Everything above the Hamiltonian touches energy only through
`total_energy` / `site_coeffs!` / `delta_energy` / `site_gradient` — the seam a
future kernel optimization (e.g. body-grouped instance batches) must preserve.
`site_gradient` uses `Harmonics.grad_Zlm_unsafe` (on-sphere, tangent-projected;
`e·∇E = 0`) and is diagnostics/tests only.

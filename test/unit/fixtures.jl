# Shared fixtures for the unit suite. `MC` aliases the package so internal (non-exported)
# names resolve as `MC._name`.

using SCEMonteCarlo
using SCEFitting
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean, std
using Test

const MC = SCEMonteCarlo

# Classical Langevin function L(x) = coth(x) − 1/x.
_langevin(x) = coth(x) - 1 / x

# --- fitted-model fixtures (mirroring SCETools' MC suite) --------------------------

# A clean ferromagnetic Heisenberg dimer: 4 atoms in a column, pair cutoff couples
# only atoms 1–2 (atoms 3–4 free); the single active SALC is the isotropic l=1 pair.
function _dimer_model()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, pair_cutoff = 2.6, lmax = [1],
                               isotropy = true))
    return SCEPredictor(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))  # < 0 ⇒ ferro
end

# The pair coupling J of the dimer (E = J e₁·e₂), read off the tiled energies of the
# aligned / anti-aligned configurations — no dependence on SCETools' ExchangeModel.
function _dimer_J(H::MC.TiledHamiltonian)
    up = SVector(0.0, 0.0, 1.0)
    aligned = MC.SpinConfig([up for _ = 1:H.n_sites])
    anti = copy(aligned)
    anti[1] = -up
    return (total_energy(H, aligned) - total_energy(H, anti)) / 2
end

# A genuine higher-multipole two-atom model (l ≤ 2, anisotropic, random couplings).
function _biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, pair_cutoff = 1.5, lmax = [2],
                               isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

# A periodic chain of one atom per cell coupled to its ±x neighbor images: the
# smallest fixture whose physics *requires* nonzero shifts (self-image pair — only
# representable on dims with N₁ ≥ 2). Hand-built raw MultipoleTerms: both directed
# members of the +x bond, E = J e_(0)·e_(+x) summed over cells.
function _chain_terms(J)
    n1 = SCEFitting.Harmonics.N1                      # Z_1m = N1 * (y, z, x)[m+2]
    folded = zeros(3, 3)
    folded[1, 1] = folded[2, 2] = folded[3, 3] = 1.0  # Σ_m Z_1m(a) Z_1m(b) ∝ a·b
    raw = J / (2 * n1^2 * (4π))                       # both members + (4π)^(2/2) scale
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(raw, 2, [1, 1], [z, x], [1, 1], copy(folded)),
            MultipoleTerm(raw, 2, [1, 1], [z, -x], [1, 1], copy(folded))]
end

# Random unit spin from `rng` (Gaussian-normalized — uniform on S²).
_rand_spin(rng) = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))

# Random configuration on the sites of `H`.
_rand_config(rng, H::MC.TiledHamiltonian) =
    MC.SpinConfig([_rand_spin(rng) for _ = 1:H.n_sites])

# 3×n matrix view of a SpinConfig (for predict_energy cross-checks).
_config_matrix(config) = reduce(hcat, [Vector(e) for e in config])

# Tile a training-cell configuration periodically onto the supercell of `H`.
function _tile_config(H::MC.TiledHamiltonian, cell_config::MC.SpinConfig)
    config = MC.SpinConfig(undef, H.n_sites)
    for s = 1:H.n_sites
        config[s] = cell_config[MC.site_atom(H, s)]
    end
    return config
end

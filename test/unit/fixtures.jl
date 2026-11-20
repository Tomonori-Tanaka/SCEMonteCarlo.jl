# Shared fixtures for the unit suite. `MC` aliases the package so internal (non-exported)
# names resolve as `MC._name`.

using SCEMonteCarlo
using SCEFitting
using Spglib: Spglib          # activates SCEFitting's SpglibBackend extension
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
function _dimer_crystal()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    return Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
end

function _dimer_model()
    b = SCEBasis(_dimer_crystal(), BasisSpec(; nbody = 2, cutoff = 2.6,
                                             lmax = [1], isotropy = true))
    return SCEPredictor(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))  # < 0 ⇒ ferro
end

# A fitted model that genuinely IS a 2× stack of a 1-atom cell: two atoms along z,
# every SALC given the same coefficient, so the Hamiltonian keeps the half-cell
# translation symmetry whatever the orbit granularity of the symmetry backend.
function _stacked_chain_model()
    lat = Lattice(Matrix(4.0 * I(3)))
    cr = Crystal(lat, [0 0; 0 0; 0.0 0.5], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.1, lmax = [1],
                               isotropy = true))
    return SCEPredictor(b, 0.0, fill(-0.02, n_salcs(b))), cr
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
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                               isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

# Anisotropic (l ≤ 2) variants of the same 2× z-stack.
# With the spglib backend the SALC orbits are translation-closed, so ANY coefficient
# vector keeps the half-cell periodicity — several channels per cluster then survive
# a genuine fitted reduction (the (coef, folded) sub-partition of `reduce_cell`).
# With `NoSymmetry()` each bond is its own orbit and the SALC tensor bases of
# translation-partner bonds need not align channel-by-channel, so even equal
# coefficients on every SALC genuinely BREAK the periodicity.
function _stacked_anisotropic_model(backend; fill_coefs::Bool = false)
    lat = Lattice(Matrix(4.0 * I(3)))
    cr = Crystal(lat, [0 0; 0 0; 0.0 0.5], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.1, lmax = [2],
                               isotropy = false); backend = backend)
    jphi = fill_coefs ? fill(0.03, n_salcs(b)) :
           0.05 .* randn(MersenneTwister(41), n_salcs(b))
    return SCEPredictor(b, 0.0, jphi), cr
end

# A fitted model whose training cell is a NON-diagonal (√2×√2, det M = 2) supercell
# of a 1-atom cell: the checkerboard. NN ±x/±y bonds all bridge the two cosets.
function _checkerboard_model()
    lat = Lattice([1.0 1.0 0; -1.0 1.0 0; 0 0 4.0])
    cr = Crystal(lat, [0 0.5; 0 0.5; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                               isotropy = true))
    return SCEPredictor(b, 0.0, fill(-0.02, n_salcs(b))), cr
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

# Hand-built 3-body chain cluster (0, +x, +2x) with a few dense folded entries —
# the smallest fixture whose contraction programs take the TRIPLET fast path
# (`site_col < 0`; asserted where used). Not a physically motivated coupling —
# kernel-equivalence gates compare arithmetic, not physics.
function _threebody_terms(J)
    folded = zeros(3, 3, 3)
    folded[2, 2, 2] = 1.0
    folded[1, 3, 1] = 0.7
    folded[3, 1, 2] = -0.4
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(J, 3, [1, 1, 1], [z, x, 2 * x], [1, 1, 1], folded)]
end

# Hand-built 4-body chain cluster (0, +x, +2x, +3x): body ≥ 4 has no fast path,
# so its programs take the GENERAL contraction branch (`site_col == 0` — the
# sfac/inst_sites indirection chain; asserted where used). Needs dims N₁ ≥ 4.
function _fourbody_terms(J)
    folded = zeros(3, 3, 3, 3)
    folded[2, 2, 2, 2] = 1.0
    folded[1, 3, 2, 1] = 0.6
    folded[3, 2, 1, 3] = -0.3
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(J, 4, [1, 1, 1, 1], [z, x, 2 * x, 3 * x], [1, 1, 1, 1],
                          folded)]
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

# ⟨e₁·e₂⟩ of the dimer's coupled pair, as a user observable. Lives here rather than in
# whichever test file happened to need it first: `test_metropolis.jl`, `test_pt.jl` and
# `test_overrelaxation.jl` all use it, so defining it in one of them made the other two
# fail with `UndefVarError` when run alone or in a different `runtests.jl` order.
# [Backported from SLCEMonteCarlo.jl 2b252da.]
_corr12_obs() = Observable(:corr12, 1, (cfg, E, H) -> dot(cfg[1], cfg[2]))

# The interrupted-writer pattern: a completed mc file ALWAYS ends at the completed
# marker — `_mc_loop!` writes an unconditional end-of-temperature boundary
# checkpoint — so a resume built on a finished run returns the stored result
# without re-running a sweep, and a resume-equals-uninterrupted assertion on it
# compares the file with itself. Mid-run continuation teeth therefore REQUIRE a
# writer that actually stops: the poison observable throws at the n-th
# measurement, leaving the file at the last periodic (or boundary) write, and
# `resume` completes the run from there. The benign twin re-supplies the name at
# resume (the checkpoint validates observable names/ncomps, never functions) and
# returns the same 0.0 the poison did before it fired, so the continued
# statistics stay bit-comparable. The PT gates of test_checkpoint.jl need none of this:
# `_pt_run!` has no end-of-run write, so their files land mid-measure by interval
# arithmetic (asserted where it matters).
# [Backported from SLCEMonteCarlo.jl 9f722e9.]
function _poison_pair(n::Int)
    cnt = Ref(0)
    return Observable(:poison, 1,
                      (cfg, E, H) -> (cnt[] += 1) >= n ? error("poison interrupt") :
                                     0.0),
           Observable(:poison, 1, (cfg, E, H) -> 0.0)
end

# Bitwise comparison of two run results (MCResult / PTResult): every point's kT,
# stats (mean / err / tau_int / count), acceptances, final step, and drift. Shared
# by test_checkpoint.jl (resume ≡ uninterrupted) and test_zeeman.jl.
function _assert_same_result(a, b)
    @test length(a.points) == length(b.points)
    for (pa, pb) in zip(a.points, b.points)
        @test pa.kT == pb.kT
        @test sort(collect(keys(pa.stats))) == sort(collect(keys(pb.stats)))
        for k in keys(pa.stats)
            @test pa.stats[k].mean == pb.stats[k].mean
            @test pa.stats[k].err == pb.stats[k].err
            @test isequal(pa.stats[k].tau_int, pb.stats[k].tau_int)
            @test pa.stats[k].count == pb.stats[k].count
        end
        @test pa.acceptance_metropolis == pb.acceptance_metropolis
        @test isequal(pa.acceptance_or, pb.acceptance_or)
        @test pa.final_step == pb.final_step
        @test pa.max_drift == pb.max_drift
    end
end

# --- constrained-MC fixtures (docs/specs/constrained-mc.md) ------------------------
#
# Hand-written raw `MultipoleTerm` lists, all with zero shifts, so every fixture
# also tiles onto dims = (1,1,1) (the small-N statistical gates) and a dims with
# N₁ ≥ 2 holds cross-cell site pairs that share NO instance. Coefficients are
# arbitrary numbers, not fitted physics: wherever a gate needs a coupling constant
# it reads it off energy differences, never from these literals. Seeds are fixed so
# every caller sees the same tensors.

const _CMC_Z = SVector(0, 0, 0)

# Identity `folded` of an l ⊗ l pair: Σ_m Z_lm(a) Z_lm(b) ∝ P_l(a·b) by the addition
# theorem (real tesseral harmonics are an orthogonal transform of the complex ones),
# i.e. a rotation-invariant (isotropic) pair coupling.
_iso_folded(l) = Matrix{Float64}(I, 2l + 1, 2l + 1)

# F2g — 2 atoms: an isotropic l = 1 pair plus a body-1 l = 2 term per atom with all
# m = −2..2 entries random (φ-dependent anisotropy).
function _f2g_terms()
    rng = MersenneTwister(2101)
    terms = [MultipoleTerm(-0.02, 2, [1, 2], [_CMC_Z, _CMC_Z], [1, 1], _iso_folded(1))]
    for a = 1:2
        push!(terms, MultipoleTerm(0.01, 1, [a], [_CMC_Z], [2], randn(rng, 5)))
    end
    return terms
end

# FZ — F2g in a field: moments (μ_B per cell atom) and a field (tesla) chosen so the
# Zeeman torque is not negligible next to the fitted one (MU_B_EV_T·m·|B| ~ 1e-3 eV).
_fz_magmoms() = [2.2, 1.3]
_fz_field() = SVector(3.0, -5.0, 8.0)

# F3 — 3 atoms: isotropic l = 1 pairs on all three bonds, a 3-body ls = [1,1,1] term
# with a dense random `folded` (odd under time reversal: Σl = 3), and the same
# body-1 l = 2 term on every atom.
function _f3_terms()
    rng = MersenneTwister(3101)
    z = _CMC_Z
    terms = [MultipoleTerm(-0.02, 2, [a, b], [z, z], [1, 1], _iso_folded(1))
             for (a, b) in ((1, 2), (1, 3), (2, 3))]
    push!(terms, MultipoleTerm(0.01, 3, [1, 2, 3], [z, z, z], [1, 1, 1],
                               randn(rng, 3, 3, 3)))
    onsite = randn(rng, 5)
    for a = 1:3
        push!(terms, MultipoleTerm(0.01, 1, [a], [z], [2], copy(onsite)))
    end
    return terms
end

# F3a — F3 with anisotropic two-ion terms added (an l = 1 ⊗ l = 1 pair whose
# `folded` is a random non-identity matrix, an l = 2 ⊗ l = 1 pair, an l = 2 ⊗ l = 2
# pair) and the single-ion `folded` drawn independently per atom. An isotropic pair's
# torque vanishes configuration by configuration and an atom-uniform single-ion term
# is blind to an atom relabelling — this fixture makes both visible (G17).
function _f3a_terms()
    rng = MersenneTwister(3102)
    z = _CMC_Z
    terms = filter(t -> t.body != 1, _f3_terms())
    push!(terms, MultipoleTerm(0.015, 2, [1, 2], [z, z], [1, 1], randn(rng, 3, 3)))
    push!(terms, MultipoleTerm(0.01, 2, [2, 3], [z, z], [2, 1], randn(rng, 5, 3)))
    push!(terms, MultipoleTerm(0.008, 2, [1, 3], [z, z], [2, 2], randn(rng, 5, 5)))
    for a = 1:3
        push!(terms, MultipoleTerm(0.01, 1, [a], [z], [2], randn(rng, 5)))
    end
    return terms
end

# F4 — 4 atoms: one 4-body ls = [1,1,1,1] term (the `_fourbody_terms` shape: the
# m = 0 entry folded[2,2,2,2] plus a few off-diagonal ones; body ≥ 4 runs the
# GENERAL contraction branch of `site_coeffs!`) and an isotropic pair (1,2), so the
# pair (1,2) shares two instances while (1,4) shares only the 4-body one.
function _f4_terms()
    z = _CMC_Z
    folded = zeros(3, 3, 3, 3)
    folded[2, 2, 2, 2] = 1.0
    folded[1, 3, 2, 1] = 0.6
    folded[3, 2, 1, 3] = -0.3
    folded[1, 1, 3, 2] = 0.45
    return [MultipoleTerm(0.02, 4, [1, 2, 3, 4], [z, z, z, z], [1, 1, 1, 1], folded),
            MultipoleTerm(-0.02, 2, [1, 2], [z, z], [1, 1], _iso_folded(1))]
end

# Number of instances of `H` that contain both sites `i` and `j` (read off the
# instance CSR) — the non-vacuity check of the "shares / does not share" cases.
function _n_shared_instances(H::MC.TiledHamiltonian, i::Integer, j::Integer)
    n = 0
    for inst in eachindex(H.inst_term)
        members = view(H.inst_sites, H.inst_ptr[inst]:(H.inst_ptr[inst + 1] - 1))
        n += (i in members) && (j in members)
    end
    return n
end

# F2u — 2 atoms: the F2g isotropic pair plus a uniaxial body-1 l = 2 term per atom
# (only the m = 0 entry, index 3 of the l = 2 row: the anisotropy is φ-independent).
# Not used by the M3 gates; reserved for the statistical gates of the CMC sweep (G6).
function _f2u_terms()
    folded = zeros(5)
    folded[3] = 1.0
    terms = [MultipoleTerm(-0.02, 2, [1, 2], [_CMC_Z, _CMC_Z], [1, 1], _iso_folded(1))]
    for a = 1:2
        push!(terms, MultipoleTerm(0.01, 1, [a], [_CMC_Z], [2], copy(folded)))
    end
    return terms
end

# F2w± — F2g (`_f2g_terms`) under the explicit per-atom weights (1, 2) and (1, −2).
# The weights belong to the constraint, not to the Hamiltonian.
_f2w_weights(sgn) = [1.0, 2.0 * sgn]

# Fmix — 8 atoms on dims = (1,1,1) (site s = atom s): a ring of isotropic l = 1 pairs
# over atoms 1..7, two anisotropic pairs, a 3-body ls = [1,1,1] term, and a random
# body-1 l = 2 term on each of atoms 1..7. Atom 8 appears in no term, so site 8 is
# INACTIVE — and carries a NONZERO weight (it must still stay out of 𝒞). Atom 7 is
# active with weight 0 (a hybrid single-spin site). The other weights are unequal
# with mixed signs, so a wrong sign or a |w| ratio is visible.
const _FMIX_INACTIVE = 8
const _FMIX_ZERO_W = 7
_fmix_weights() = [1.0, -0.4, 1.5, -1.2, 0.8, 2.0, 0.0, 0.7]

function _fmix_terms()
    rng = MersenneTwister(8101)
    z = _CMC_Z
    terms = [MultipoleTerm(-0.02, 2, [a, mod1(a + 1, 7)], [z, z], [1, 1],
                           _iso_folded(1)) for a = 1:7]
    push!(terms, MultipoleTerm(0.015, 2, [1, 4], [z, z], [1, 1], randn(rng, 3, 3)))
    push!(terms, MultipoleTerm(0.01, 2, [2, 5], [z, z], [2, 1], randn(rng, 5, 3)))
    push!(terms, MultipoleTerm(0.01, 3, [3, 5, 6], [z, z, z], [1, 1, 1],
                               randn(rng, 3, 3, 3)))
    for a = 1:7
        push!(terms, MultipoleTerm(0.01, 1, [a], [z], [2], randn(rng, 5)))
    end
    return terms
end

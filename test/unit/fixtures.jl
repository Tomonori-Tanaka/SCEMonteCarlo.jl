# Shared fixtures for the unit suite. `MC` aliases the package so internal (non-exported)
# names resolve as `MC._name`.

using SCEMonteCarlo
using SCEFitting
using Spglib: Spglib          # activates SCEFitting's SpglibBackend extension
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean, std
using Printf: @printf
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

# --- constrained-MC test-side oracles ---------------------------------------------
#
# Shared by test_cmc_sweep.jl (the sweep harness) and test_cmc_driver.jl (the same
# oracles through `run_cmc`). Written from the formulas in the comments — none of
# them calls the package's CMC code.

_sw_dir(θ, φ) = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
_sw_eθ(θ, φ) = SVector(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
_sw_eφ(θ, φ) = SVector(-sin(φ), cos(φ), 0.0)

# An orthonormal frame (t₁, t₂) of the plane ⊥ n̂.
function _sw_frame(n̂)
    ref = abs(n̂[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t1 = normalize(cross(n̂, ref))
    return (t1, cross(n̂, t1))
end

# Weighted magnetization over the constrained sites (active, w ≠ 0), summed here.
function _sw_M(config, w, act)
    M = zero(SVector{3,Float64})
    for s in eachindex(config)
        (act[s] && w[s] != 0) || continue
        M += w[s] * config[s]
    end
    return M
end

# Gauss–Legendre nodes and weights on [-1, 1] (Newton on the three-term recursion).
function _sw_gauss_legendre(n)
    x = zeros(n)
    w = zeros(n)
    for i = 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        pp = 0.0
        for _ = 1:100
            p1, p2 = 1.0, 0.0
            for k = 1:n
                p1, p2 = ((2k - 1) * z * p1 - (k - 1) * p2) / k, p1
            end
            pp = n * (z * p1 - p2) / (z^2 - 1)
            dz = p1 / pp
            z -= dz
            abs(dz) < 1e-16 && break
        end
        p1, p2 = 1.0, 0.0
        for k = 1:n
            p1, p2 = ((2k - 1) * z * p1 - (k - 1) * p2) / k, p1
        end
        pp = n * (z * p1 - p2) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * pp^2)
    end
    return x, w
end

# Gauss–Legendre rule mapped to [a, b].
function _sw_gl(a, b, n)
    x, w = _sw_gauss_legendre(n)
    return (a + b) / 2 .+ (b - a) / 2 .* x, (b - a) / 2 .* w
end

# --- dimer energy in closed form ------------------------------------------------------
#
# A two-atom fixture with an isotropic l = 1 pair and body-1 l = 2 terms has, on the
# sphere, E(e₁, e₂) = J e₁·e₂ + e₁ᵀA₁e₁ + e₂ᵀA₂e₂ + C: an l = 2 tesseral harmonic is a
# quadratic form of the unit vector. J and A_a are read off `total_energy`
# differences (J from the aligned / anti-aligned pair as `_dimer_J`; A_a from six
# directions of one spin with the other fixed at ẑ: diagonal from x̂, ŷ, ẑ,
# off-diagonal from (x̂+ŷ)/√2 etc., the isotropic constant absorbed into the
# diagonal), C from one more energy. The reconstruction is then checked against
# `total_energy` on random configurations, so the quadrature runs on a verified
# closed form — never on guessed coupling constants.
function _sw_dimer_model(H)
    J = _dimer_J(H)
    ẑ = SVector(0.0, 0.0, 1.0)
    axes3 = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), ẑ)
    E(e1, e2) = total_energy(H, MC.SpinConfig([e1, e2]))
    function quad_form(g)
        A = zeros(3, 3)
        for a = 1:3
            A[a, a] = g(axes3[a])
        end
        for a = 1:3, b = (a + 1):3
            A[a, b] = A[b, a] = g(normalize(axes3[a] + axes3[b])) - (A[a, a] + A[b, b]) / 2
        end
        return SMatrix{3,3}(A)
    end
    A1 = quad_form(e -> E(e, ẑ) - J * dot(e, ẑ))
    A2 = quad_form(e -> E(ẑ, e) - J * dot(ẑ, e))
    C = E(ẑ, ẑ) - (J + A1[3, 3] + A2[3, 3])
    Ef(e1, e2) = J * dot(e1, e2) + dot(e1, A1 * e1) + dot(e2, A2 * e2) + C
    rng = MersenneTwister(6601)
    worst = maximum(1:200) do _
        e1, e2 = _rand_spin(rng), _rand_spin(rng)
        abs(Ef(e1, e2) - E(e1, e2))
    end
    return Ef, J, A1, A2, worst
end

# Constrained partition function of a two-spin system with weights (w₁, w₂),
# integrating over spin 1 (choose |w₁| ≤ |w₂|: the integrand is then smooth; the
# other choice has an endpoint 1/√ singularity). From
# Z = ∫dΩ₁dΩ₂ e^{−βE} |M|² δ²(M⊥) Θ(M∥): δ²(w₁e₁⊥ + w₂e₂⊥) fixes
# e₂⊥ = −(w₁/w₂) e₁⊥ with Jacobian 1/w₂², and dΩ₂ = d²e₂⊥ / |e₂∥| over the two
# hemispheres σ = ±, e₂∥ = σ√(1 − |e₂⊥|²):
#   Z = ∫dΩ₁ Σ_σ M∥² / (w₂² |e₂∥|) e^{−βE(e₁, e₂)} Θ(M∥) Θ(1 − |e₂⊥|²),
#   M∥ = w₁ (e₁·n̂) + w₂ e₂∥.
# For w₁ = w₂ this is 4∫_{c>0} dΩ c e^{−βE(e, 2c n̂ − e)}; Θ(M∥) then jumps at
# c₁ = 0, which is why the c₁ range is split there. Gauss–Legendre in c₁ on each
# half, the periodic trapezoid in the azimuth ψ.
function _sw_Z2(Ef, β, w1, w2, n̂, nc, nψ)
    t1, t2 = _sw_frame(n̂)
    Z = 0.0
    for (a, b) in ((-1.0, 0.0), (0.0, 1.0))
        cs, ws = _sw_gl(a, b, nc)
        for (c1, wc) in zip(cs, ws), k = 0:(nψ - 1)
            ψ = 2π * k / nψ
            s = sqrt(1 - c1^2)
            e1p = s * cos(ψ) * t1 + s * sin(ψ) * t2
            e1 = e1p + c1 * n̂
            e2p = -(w1 / w2) * e1p
            p2 = dot(e2p, e2p)
            p2 < 1 || continue
            for σ in (1.0, -1.0)
                e2par = σ * sqrt(1 - p2)
                Mpar = w1 * c1 + w2 * e2par
                Mpar > 0 || continue
                Z += wc * (2π / nψ) * Mpar^2 / (w2^2 * abs(e2par)) *
                     exp(-β * Ef(e1, e2p + e2par * n̂))
            end
        end
    end
    return Z
end

# Reference torque components (⟨T⟩·ê_φ, ⟨T⟩·ê_θ, ⟨T⟩·n̂) at (θ, φ) from
# F = −kT ln Z₂: ⟨T⟩·ê_φ = −∂F/∂θ, ⟨T⟩·ê_θ = (1/sinθ) ∂F/∂φ, ⟨T⟩·n̂ = 0
# (docs/specs/constrained-mc.md D5), with the five-point central difference
# (truncation O(h⁴) ~ 1e-10 relative at h = 0.01). Also returns the worst relative
# change of Z between the (nc, nψ) grid and the doubled grid over the stencil.
function _sw_torque_ref(Ef, β, w1, w2, θ, φ; nc = 40, nψ = 48, h = 0.01)
    worst = 0.0
    function F(θ_, φ_)
        Za = _sw_Z2(Ef, β, w1, w2, _sw_dir(θ_, φ_), nc, nψ)
        Zb = _sw_Z2(Ef, β, w1, w2, _sw_dir(θ_, φ_), 2nc, 2nψ)
        worst = max(worst, abs(Za - Zb) / Zb)
        return -log(Zb) / β
    end
    d(f) = (-f(2h) + 8f(h) - 8f(-h) + f(-2h)) / (12h)
    dFθ = d(x -> F(θ + x, φ))
    dFφ = d(x -> F(θ, φ + x))
    return (-dFθ, dFφ / sin(θ), 0.0), worst
end

# Constant-energy fixture: a body-1 l = 0 term on every atom makes every site active
# while the energy is a constant (ΔE ≡ 0 exactly: the l = 0 row never changes).
_sw_const_terms(n) = [MultipoleTerm(0.3, 1, [a], [_CMC_Z], [0], [1.0]) for a = 1:n]

const _SW_VERBOSE = get(ENV, "SCEMC_TEST_VERBOSE", "0") == "1"

# Aggregate check of statistical rows (name, value, σ, exact, εmax): every
# |dev| < k σ and every σ < εmax (≈ 2–3× the measured σ, so a gate cannot pass by
# being noisy); prints the first offender. Returns the worst |dev|/σ.
# Set SCEMC_TEST_VERBOSE=1 to print every row (used when re-measuring a gate).
function _sw_check_rows(label, rows; k = 5.0)
    if _SW_VERBOSE
        for r in rows
            @printf("%-4s %-24s value %+.6e  exact %+.6e  σ %.3e  dev/σ %+6.2f\n",
                    label, r.name, r.value, r.exact, r.σ, (r.value - r.exact) / r.σ)
        end
    end
    bad = filter(r -> !(abs(r.value - r.exact) < k * r.σ && r.σ < r.εmax), rows)
    isempty(bad) || @error "statistical gate: first offender" first(bad) length(bad)
    @test isempty(bad)
    return maximum(r -> abs(r.value - r.exact) / r.σ, rows)
end

# Shared helpers of the constrained-MC manual validation scripts
# (bench/validate_cmc_*.jl, docs/specs/constrained-mc.md "Manual validation").
#
# `include`d by each script. Everything here is test-side arithmetic written from the
# formulas in the comments: frames, rotations, Simpson weights, lane pooling, and the
# verdict rows. Nothing here calls the package's constrained-MC internals.

using SCEMonteCarlo
using SCEMonteCarlo: SpinConfig
using SCEFitting
using LinearAlgebra
using StaticArrays: SVector, SMatrix
using Statistics: mean, std
using Printf: @printf, @sprintf
using Random: Xoshiro, MersenneTwister

const MC = SCEMonteCarlo

# --- frames ---------------------------------------------------------------------------
# Physics convention n̂(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ); ê_φ is the axis of the
# rigid rotation that increases θ (so ⟨ê_φ·T⟩ = −∂F/∂θ), ê_θ the one that moves φ
# (⟨ê_θ·T⟩ = (1/sinθ) ∂F/∂φ); docs/specs/constrained-mc.md D5.
vdir(θ, φ) = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
veθ(θ, φ) = SVector(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
veφ(θ, φ) = SVector(-sin(φ), cos(φ), 0.0)

# Uniform random unit vector (normalized Gaussian).
rand_unit(rng) = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))

# Rotation of `e` by the angle `α` about the unit axis `w` (Rodrigues).
rot_about(e, w, α) = cos(α) * e + sin(α) * cross(w, e) + (1 - cos(α)) * dot(w, e) * w

# The minimal rotation matrix taking the unit vector `a` onto the unit vector `b`
# (Rodrigues with v = a × b, c = a·b: R = I + [v]× + [v]×² / (1 + c)); for the
# measure-zero antiparallel case a half turn about any axis ⊥ a.
function min_rot(a::SVector{3,Float64}, b::SVector{3,Float64})
    c = dot(a, b)
    if c < -1 + 1e-12
        p = abs(a[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
        w = normalize(cross(a, p))
        return SMatrix{3,3}(2 * w * w' - I)
    end
    v = cross(a, b)
    K = SMatrix{3,3}(0.0, v[3], -v[2], -v[3], 0.0, v[1], v[2], -v[1], 0.0)
    return SMatrix{3,3}(I + K + K * K / (1 + c))
end

# Composite Simpson weights on n (odd) equally spaced nodes with spacing Δ.
function simpson_weights(n::Integer, Δ::Real)
    isodd(n) || error("Simpson needs an odd number of nodes")
    w = [k == 1 || k == n ? 1.0 : (iseven(k) ? 4.0 : 2.0) for k = 1:n]
    return w .* (Δ / 3)
end

# --- lane / replica pooling -----------------------------------------------------------
# Pool R independent replica estimates (mean m_r, binning error s_r) of one quantity:
# the plain mean with σ = √(Σ s_r²)/R, and χ²/(R−1) of the replica means about it (a
# check that the binning errors are honest: ≈ 1, ≫ 1 means underestimated errors).
function pool(ms::AbstractVector, ss::AbstractVector)
    R = length(ms)
    m = mean(ms)
    σ = sqrt(sum(abs2, ss)) / R
    χ2 = R > 1 ? sum(((ms .- m) ./ ss) .^ 2) / (R - 1) : NaN
    return (; mean = m, σ = σ, chi2 = χ2)
end

# Pool component `c` of statistic `key` (or a function of the stat vector, with its
# error vector) over a vector of results.
pool_stat(rs, key::Symbol, c::Integer) =
    pool([r.points[1].stats[key].mean[c] for r in rs],
         [r.points[1].stats[key].err[c] for r in rs])

# --- verdict rows ---------------------------------------------------------------------
# One comparison row: |value − ref| < k·σ_comb with σ_comb = √(σ_v² + σ_r²); prints the
# row and returns `true` when it passes.
function check_row(label, value, σv, ref, σr; k = 4.0)
    σ = sqrt(σv^2 + σr^2)
    dev = (value - ref) / σ
    ok = abs(dev) < k
    @printf("  %-34s cmc % .6e ± %.2e   ref % .6e ± %.2e   dev %+6.2fσ  %s\n",
            label, value, σv, ref, σr, dev, ok ? "ok" : "FAIL")
    return ok
end

# --- hand-written lattice terms -------------------------------------------------------
# Simple-cubic 1-atom cell with nearest-neighbour isotropic l = 1 pairs, E = J Σ_bonds
# e·e′ (each bond once): per axis a ∈ {x̂, ŷ, ẑ} the two directed members (0, +a) and
# (0, −a), each with raw coefficient J / (2 N1² 4π) — the `(4π)^(2/2)` scale applied
# by the TiledHamiltonian ctor and Σ_m Z_1m(a) Z_1m(b) = N1² a·b undo it. J < 0 is
# ferromagnetic. The scripts read the realized J back off energy differences.
function sc_pair_terms(J::Real)
    n1 = SCEFitting.Harmonics.N1
    raw = J / (2 * n1^2 * (4π))
    z = SVector(0, 0, 0)
    terms = MultipoleTerm[]
    for a in (SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1)), s in (1, -1)
        push!(terms, MultipoleTerm(raw, 2, [1, 1], [z, s * a], [1, 1],
                                   Matrix{Float64}(I, 3, 3)))
    end
    return terms
end

# A body-1 uniaxial l = 2 term (only the m = 0 entry of the l = 2 row, index 3), raw
# coefficient `c`. On the sphere Z_20 ∝ 3z² − 1, so the energy is −K Σ (e·ẑ)² + const
# for a K ∝ −c that the scripts read off energy differences.
uniaxial_term(c::Real) =
    MultipoleTerm(c, 1, [1], [SVector(0, 0, 0)], [2], [0.0, 0.0, 1.0, 0.0, 0.0])

# Energy of the uniform state `e` on every site.
uniform_energy(H, e) = total_energy(H, SpinConfig(fill(SVector{3,Float64}(e), n_sites(H))))

# Realized NN coupling of an sc ferromagnet: E(all ẑ) − E(one spin flipped) = 2·z·J,
# so J = (E_up − E_flip) / (2·6) with 6 neighbours (dims ≥ 3 along every axis).
function sc_read_J(H)
    up = SVector(0.0, 0.0, 1.0)
    c = SpinConfig(fill(up, n_sites(H)))
    E0 = total_energy(H, c)
    c[1] = -up
    return (E0 - total_energy(H, c)) / 12
end

banner(title) = (println("=" ^ 92); println(title);
                 println("julia $(VERSION)   threads=$(Threads.nthreads())   ",
                         "SCEMonteCarlo at $(pkgdir(SCEMonteCarlo))");
                 println("=" ^ 92))

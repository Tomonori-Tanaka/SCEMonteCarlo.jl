# Constrained Monte Carlo (CMC) for finite-temperature anisotropy: the
# Asselin-type pair move that holds the direction of a (weighted) magnetization
# fixed, plus the torque observables that read the free-energy gradient off the
# constrained ensemble. The standing contract — constrained measure, acceptance
# rule, null moves, partner rule, sweep and RNG consumption, torque conventions —
# is `docs/specs/constrained-mc.md`.
#
# This file only CALLS the energy kernels of energy.jl (`site_coeffs!`,
# `delta_energy`, `_zlm_row!`) and the public gradient (`energy_gradient!`, via
# `total_torque` in observables.jl); it changes none of them.

# --- the constraint ------------------------------------------------------------------

const _CMC_WEIGHT_RULES = (:auto, :uniform, :magmoms)

# Unit vector of `v`, with every component clamped into [-1, 1]. With correctly
# rounded arithmetic the division alone already keeps every component in [-1, 1]:
# the computed norm is ≥ every |component| (`fl(√fl(x²)) = |x|`, adding
# non-negative terms and `sqrt` round monotonically, and StaticArrays' scaled norm
# keeps the same bound), and a monotone rounded quotient |x|/n with |x| ≤ n is
# ≤ 1. The clamp is a guard that costs nothing and never moves a value in practice.
# Why the bound matters: the tesseral harmonics evaluate the associated Legendre
# functions at `u[3]` (`dnPl`), which throw a `DomainError` for |z| > 1, and a
# checkpoint restore refuses any component outside [-1, 1]
# (`_validate_unit_column`, state.jl).
@inline _unit_clamped(v::SVector{3,Float64})::SVector{3,Float64} =
    clamp.(v / norm(v), -1.0, 1.0)

"""
    MagnetizationConstraint(direction::AbstractVector{<:Real}; weights = :auto)
    MagnetizationConstraint(direction::NTuple{3,Real}; weights = :auto)
    MagnetizationConstraint(θ::Real, φ::Real; weights = :auto)

The constraint of a constrained Monte Carlo run: the direction `n̂` of the weighted
magnetization `M = Σ_{s∈𝒞} w_{a(s)} e_s` is held fixed (`docs/specs/constrained-mc.md`
D1). The sampled measure is the canonical one restricted to `M̂ = n̂` with the
factor `|M|²` of the solid-angle delta function, so that the mean torque is the
gradient of the free energy `F(n̂)` of the canonical distribution of `M̂`.

- `direction` — any finite nonzero 3-vector or 3-tuple; it is normalized
  (stored unit).
  The angle form uses the physics convention `n̂ = (sinθ cosφ, sinθ sinφ, cosθ)`:
  `θ` is the polar angle from `+z`, `φ` the azimuth from `+x` toward `+y`.
- `weights` — one of
  - `:auto` — the moments `magmoms` of the Hamiltonian when it carries them,
    otherwise `1` for every cell atom (always non-negative);
  - `:uniform` — `1` for every cell atom;
  - `:magmoms` — the moments of the Hamiltonian (refused when it has none);
  - a vector of finite real numbers, one per training-cell atom of the
    Hamiltonian in its atom order (the reduced cell's order after
    [`reduce_cell`](@ref)). Weights may be negative (an antiparallel sublattice
    enters `M` with its sign flipped) or zero (that atom's sites are not
    constrained and are updated by plain single-spin Metropolis).

The constrained set `𝒞` is the active sites whose atom has a nonzero weight;
inactive sites never enter `M`. Symbolic weights are resolved, and the length of a
weight vector checked, against the Hamiltonian of the run.

Two constraints compare equal (`==`, `isequal`, `hash`) when their stored
direction and weights are equal field by field (signed zeros are folded to `+0.0`
at construction, so the three agree). The weight vector is copied at construction;
do not mutate `c.weights` afterwards (it would change the hash; the finiteness is
re-checked when the constraint is resolved against a Hamiltonian).
"""
struct MagnetizationConstraint
    direction::SVector{3,Float64}
    weights::Union{Symbol,Vector{Float64}}

    function MagnetizationConstraint(direction::SVector{3,Float64},
                                     weights::Union{Symbol,Vector{Float64}})
        all(isfinite, direction) || throw(ArgumentError(
            "MagnetizationConstraint: direction $direction is not finite"))
        n = norm(direction)
        n > 0.0 || throw(ArgumentError(
            "MagnetizationConstraint: direction $direction is zero"))
        if weights isa Symbol
            weights in _CMC_WEIGHT_RULES || throw(ArgumentError(
                "MagnetizationConstraint: weights = :$weights; expected one of " *
                "$(_CMC_WEIGHT_RULES) or a vector of per-cell-atom weights"))
        else
            all(isfinite, weights) || throw(ArgumentError(
                "MagnetizationConstraint: weights must be finite; got $weights"))
            weights = weights .+ 0.0        # a copy, with −0.0 folded to +0.0
        end
        # `.+ 0.0` folds −0.0 to +0.0, so `==`, `isequal`, and `hash` agree on
        # directions that differ only in the sign of a zero component
        return new(_unit_clamped(direction) .+ 0.0, weights)
    end
end

function MagnetizationConstraint(direction::AbstractVector{<:Real}; weights = :auto)
    length(direction) == 3 || throw(ArgumentError(
        "MagnetizationConstraint: direction must have 3 components; got " *
        "$(length(direction))"))
    w = if weights isa Symbol
        weights
    elseif weights isa AbstractVector{<:Real}
        Vector{Float64}(weights)
    else
        throw(ArgumentError(
            "MagnetizationConstraint: weights must be a Symbol or a real vector; " *
            "got $(typeof(weights))"))
    end
    return MagnetizationConstraint(SVector{3,Float64}(direction), w)
end

MagnetizationConstraint(direction::NTuple{3,Real}; weights = :auto) =
    MagnetizationConstraint(SVector{3,Float64}(direction); weights = weights)

function MagnetizationConstraint(θ::Real, φ::Real; weights = :auto)
    (isfinite(θ) && isfinite(φ)) || throw(ArgumentError(
        "MagnetizationConstraint: angles must be finite; got θ = $θ, φ = $φ"))
    return MagnetizationConstraint(SVector{3,Float64}(sin(θ) * cos(φ),
                                                      sin(θ) * sin(φ), cos(θ));
                                   weights = weights)
end

Base.:(==)(a::MagnetizationConstraint, b::MagnetizationConstraint) =
    a.direction == b.direction && a.weights == b.weights
Base.isequal(a::MagnetizationConstraint, b::MagnetizationConstraint) =
    isequal(a.direction, b.direction) && isequal(a.weights, b.weights)
Base.hash(c::MagnetizationConstraint, h::UInt) =
    hash(c.weights, hash(c.direction, hash(:MagnetizationConstraint, h)))

# A `MagnetizationConstraint` resolved against one Hamiltonian — the type-stable
# form the hot path reads:
# - `nhat`   — the unit constraint direction n̂;
# - `cell_w` — the resolved per-cell-atom weights (what a checkpoint and the result
#              record);
# - `site_w` — the weight of every supercell site, `cell_w[site_atom(s)]` on the
#              constrained set and 0.0 elsewhere (inactive sites included), so
#              `site_w[s] != 0` exactly when `s ∈ cset` and `sum(abs, site_w)` is
#              `Σ_{s∈𝒞} |w_s|`;
# - `cset`   — the constrained sites 𝒞 (active, nonzero weight), ascending;
# - `wnorm`  — the weight norm `Σ_{s∈𝒞} |w_s|` (`sum(abs, site_w)`), the scale of the
#              drift and weak-constraint checks.
struct _ResolvedConstraint
    nhat::SVector{3,Float64}
    cell_w::Vector{Float64}
    site_w::Vector{Float64}
    cset::Vector{Int}
    wnorm::Float64
end

# Resolve `con` against `H` (docs/specs/constrained-mc.md D1): symbolic weights,
# the weight-vector length, and the door `|𝒞| ≥ 2` (a pair move needs a partner).
function _resolve_constraint(H::TiledHamiltonian,
                             con::MagnetizationConstraint)::_ResolvedConstraint
    n = H.n_cell_atoms
    w = con.weights
    mm = H.magmoms
    # branch on the type first, so each arm sees a concrete `w`
    cell_w = if w isa Vector{Float64}
        length(w) == n || throw(ArgumentError(
            "constraint has $(length(w)) weights but the Hamiltonian has $n cell " *
            "atoms (weights are per training-cell atom, in the Hamiltonian's order)"))
        all(isfinite, w) || throw(ArgumentError(
            "constraint weights are not finite: $w (mutated after construction?)"))
        copy(w)
    elseif w === :magmoms
        mm === nothing && throw(ArgumentError(
            "constraint weights = :magmoms, but the Hamiltonian carries no magmoms"))
        copy(mm)
    elseif w === :auto
        mm === nothing ? ones(n) : copy(mm)
    else                                    # :uniform (the inner ctor admits no other)
        ones(n)
    end
    site_w = zeros(H.n_sites)
    cset = Int[]
    for s = 1:H.n_sites
        ws = cell_w[site_atom(H, s)]
        (H.site_active[s] && ws != 0.0) || continue
        site_w[s] = ws
        push!(cset, s)
    end
    length(cset) >= 2 || throw(ArgumentError(
        "the constrained set (active sites with nonzero weight) has " *
        "$(length(cset)) site(s); a constrained run needs at least 2"))
    return _ResolvedConstraint(con.direction, cell_w, site_w, cset, sum(abs, site_w))
end

# The constrained magnetization `M = Σ_{s∈𝒞} w_s e_s`, from scratch, summed in
# `cset` order (deterministic).
function _cmc_magnetization(rc::_ResolvedConstraint,
                            config::SpinConfig)::SVector{3,Float64}
    M = zero(SVector{3,Float64})
    @inbounds for s in rc.cset
        M += rc.site_w[s] * config[s]
    end
    return M
end

"""
    _cmc_initial_config(H::TiledHamiltonian, rc::_ResolvedConstraint, init)
        -> SpinConfig

The deterministic, RNG-free start of a constrained chain
(`docs/specs/constrained-mc.md` D1), whose `M̂` equals `n̂` to rounding:

- `init === nothing` — the default pattern: `e_s = sign(w_s) n̂` on the
  constrained set (so every site adds `|w_s| n̂` to `M`), `n̂` on every other site
  (active zero-weight and inactive sites alike);
- otherwise `init` (a `3 × n_sites` matrix or a vector of 3-vectors, validated
  as the drivers' `init`) is rotated rigidly, every site included, by the minimal
  rotation taking its `M̂` to `n̂`. When `M̂ = −n̂` (within 1e-8, where the minimal
  rotation is ill-defined) the rotation is by `π` about `normalize(n̂ × x̂)` (`ŷ`
  when `n̂ ∥ x̂`), followed by the residual minimal rotation (the identity when
  `M̂ = −n̂` exactly).

Refuses (`ArgumentError`) an `init` with `|M| < 1e-8 Σ_{s∈𝒞} |w_s|`, whose
direction is undefined. Rotated spins are re-normalized (`_unit_clamped`), a
rounding-level change that keeps them in the domain of the harmonics.
"""
function _cmc_initial_config(H::TiledHamiltonian, rc::_ResolvedConstraint,
                             init)::SpinConfig
    n̂ = rc.nhat
    if init === nothing
        config = SpinConfig(undef, H.n_sites)
        for s = 1:H.n_sites
            w = rc.site_w[s]
            config[s] = w < 0.0 ? -n̂ : n̂
        end
        return config
    end
    config = _validated_init(H, init)
    M = _cmc_magnetization(rc, config)
    wnorm = sum(abs, rc.site_w)
    norm(M) >= 1e-8 * wnorm || throw(ArgumentError(
        "the weighted magnetization of init has |M| = $(norm(M)) < 1e-8·Σ|w| = " *
        "$(1e-8 * wnorm): its direction is undefined, so it cannot be rotated " *
        "onto the constraint direction"))
    R = _min_rotation(M / norm(M), n̂)
    return SpinConfig([_unit_clamped(R * e) for e in config])
end

# The minimal rotation taking the unit vector `a` to the unit vector `b`, as a 3×3
# matrix. In the plane of `a` and `b`, with `û` the unit vector along the component
# `u` of `b` orthogonal to `a` and `s = |u|`: `R a = (a·b) a + s û = b`,
# `R û = −s a + (a·b) û`, and the normal direction is fixed, i.e.
# `R = I + (a·b − 1)(a aᵀ + û ûᵀ) + s (û aᵀ − a ûᵀ)`. Written with `u` itself (not
# an axis `a × b / |a × b|`), and `û` made orthogonal to `a` by a second
# Gram–Schmidt pass, so `R` is orthogonal and `R a = b` to rounding.
#
# Near `a = −b` (`a·b < 0` and `s < 1e-8`) a separate branch is needed: for an
# exactly antiparallel pair `u = 0` and the formula has no plane at all (`s == 0`
# would return the identity), and just off it the plane is set by rounding noise.
# There the rotation is the half turn `P = 2 k kᵀ − I` about the conventional axis
# `k = normalize(b × x̂)` (`ŷ` when `b ∥ x̂`; the unnormalized `b × x̂` is exactly
# orthogonal to `b`, the normalized `k` to rounding), followed by the minimal
# rotation from `P a` (now within 1e-8 of `b`) to `b`. For `a = −b` exactly this is
# the half turn alone, to rounding.
function _min_rotation(a::SVector{3,Float64},
                       b::SVector{3,Float64})::SMatrix{3,3,Float64,9}
    Id = SMatrix{3,3,Float64,9}(I)
    c = dot(a, b)
    u = b - c * a
    u -= dot(u, a) * a
    s = norm(u)
    if c < 0.0 && s < 1e-8
        k = cross(b, SVector(1.0, 0.0, 0.0))
        kn = norm(k)
        axis = kn < floatmin(Float64) ? SVector(0.0, 1.0, 0.0) : k / kn
        P = 2 * axis * axis' - Id
        return _min_rotation(P * a, b) * P
    end
    s == 0.0 && return Id
    û = u / s
    return Id + (c - 1) * (a * a' + û * û') + s * (û * a' - a * û')
end

# --- the pair move -------------------------------------------------------------------

"""
    _CMCScratch(H::TiledHamiltonian)

Pre-allocated buffers of one CMC lane (never shared across tasks), sized like
[`SweepScratch`](@ref) from `H.nlm` / `H.lmax`:

- `lc` — the leave-one-out coefficient vector of the site being evaluated (the
  `c` of [`site_coeffs!`](@ref); named `lc` to keep it apart from the constraint);
- `zi_new` — the tesseral row `Z(e_i′)` of the source spin's proposal;
- `zi_old` — the saved column `i` of `zrows` while `Z(e_i′)` is written there
  tentatively (restored bitwise before [`_pair_delta_energy!`](@ref) returns);
- `zj_new` — the tesseral row `Z(e_j′)` of the compensating partner's proposal;
- `plm` — the associated-Legendre workspace of the internal `_zlm_row!`.

After a `_pair_delta_energy!` call `zi_new`/`zj_new` hold the proposed rows, so an
accepted move copies them into `zrows` without re-evaluating the harmonics.
"""
struct _CMCScratch
    lc::Vector{Float64}
    zi_new::Vector{Float64}
    zi_old::Vector{Float64}
    zj_new::Vector{Float64}
    plm::Vector{Float64}
end

_CMCScratch(H::TiledHamiltonian) =
    _CMCScratch(zeros(H.nlm), zeros(H.nlm), zeros(H.nlm), zeros(H.nlm),
                Vector{Float64}(undef, H.lmax + 1))

"""
    _pair_delta_energy!(scratch::_CMCScratch, zrows, H, i, j, e_i_new, e_j_new)
        -> Float64

Exact energy change of moving the two distinct sites `i ≠ j` jointly to
`e_i_new`, `e_j_new`, by sequential leave-one-out (`docs/specs/constrained-mc.md`):

    ΔE = [E(e_i′, e_j) − E(e_i, e_j)] + [E(e_i′, e_j′) − E(e_i′, e_j)]

1. `ΔE_i = c_i · (Z(e_i′) − Z(e_i))` with `c_i` from the current `zrows`;
2. column `i` of `zrows` is saved to `scratch.zi_old` and `Z(e_i′)` is written
   there tentatively;
3. `ΔE_j = c_j · (Z(e_j′) − Z(e_j))` with `c_j` built from the tentative `zrows`,
   so every instance that contains both sites sees `e_i′`;
4. column `i` is restored bitwise from `scratch.zi_old`.

Each bracket is a single-site move, exact at any body order under the
distinct-member-sites invariant of the constructor. When `i` and `j` share no
instance, `c_j` does not read column `i` and the result is bitwise the naive sum
of two independent single-site differences.

On return `zrows` is bitwise unchanged, `scratch.zi_new == Z(e_i′)` and
`scratch.zj_new == Z(e_j′)`; nothing else is written. Allocation-free. The spins
of sites `i`, `j` in the configuration are never read — `zrows` is the state.

Preconditions (the caller's): `i, j ∈ 1:H.n_sites`, `zrows` of size
`H.nlm × H.n_sites`, `scratch` built for this `H`, and `e_i_new`, `e_j_new` unit
vectors whose components lie in `[-1, 1]` (the harmonics throw otherwise — before
`zrows` is touched, so a throw leaves it intact).
"""
function _pair_delta_energy!(scratch::_CMCScratch, zrows::Matrix{Float64},
                             H::TiledHamiltonian, i::Int, j::Int,
                             e_i_new::SVector{3,Float64},
                             e_j_new::SVector{3,Float64})::Float64
    @assert i != j "pair move needs two distinct sites; got i = j = $i"
    lc = scratch.lc
    # Both proposed rows are tabulated BEFORE column i is touched: the harmonics
    # throw a DomainError on a component a few ulp outside [-1, 1], and a throw
    # inside the tentative-write window would leave `zrows` inconsistent.
    _zlm_row!(scratch.zi_new, e_i_new, H.lmax, scratch.plm)
    _zlm_row!(scratch.zj_new, e_j_new, H.lmax, scratch.plm)
    # (1) source site against the current rows
    fill!(lc, 0.0)
    site_coeffs!(lc, H, i, zrows)
    ΔE_i = delta_energy(lc, view(zrows, :, i), scratch.zi_new)
    # (2) tentatively place e_i′ — every instance shared with j must see it
    zcol_i = view(zrows, :, i)
    copyto!(scratch.zi_old, zcol_i)
    copyto!(zcol_i, scratch.zi_new)
    # (3) partner site against the tentative rows
    fill!(lc, 0.0)
    site_coeffs!(lc, H, j, zrows)
    ΔE_j = delta_energy(lc, view(zrows, :, j), scratch.zj_new)
    # (4) restore column i bitwise. The only calls inside the window are
    # `fill!`/`site_coeffs!`/`delta_energy` on the scratch and rows, which cannot
    # throw for in-range sites `i, j ∈ 1:H.n_sites` (the caller's precondition) and
    # a `zrows` of size `H.nlm × H.n_sites`; this is the only exit, so it always
    # runs.
    copyto!(zcol_i, scratch.zi_old)
    return ΔE_i + ΔE_j
end

"""
    _cmc_pair_proposal(config, zrows, H, rc::_ResolvedConstraint, M, i, j, e_i_new,
                       β, scratch::_CMCScratch) -> (e_j_new, ΔE, ℓ, M_new)

The deterministic core of one CMC pair attempt (`docs/specs/constrained-mc.md` D2):
given the proposed source spin `e_i_new` for the constrained site `i` and the
partner `j ∈ 𝒞 \\ {i}`, build the compensating partner spin, the exact joint
energy change, and the log acceptance ratio. Pure: `config` is not touched and
`zrows` is bitwise unchanged on return (column `i` is written only tentatively,
inside [`_pair_delta_energy!`](@ref)); only `scratch` is overwritten.

With `n̂ = rc.nhat`, weights `w_i = rc.site_w[i]`, `w_j = rc.site_w[j]`, and `x⊥ =
x − (x·n̂) n̂` (frame-free — valid for any `n̂`):

1. `e_j⊥′ = e_j⊥ − (w_i/w_j)(e_i′ − e_i)⊥`, which keeps `M⊥` fixed; `s2 =
   |e_j⊥′|²`; **null** unless `s2 < 1`;
2. `e_j′ = e_j⊥′ + σ_j √(1 − s2) n̂` with `σ_j = (e_j·n̂ ≥ 0 ? +1 : −1)` (the sign
   of the parallel part is preserved; `+1` when it is zero), re-normalized
   (see below);
3. `M′ = M + w_i (e_i′ − e_i) + w_j (e_j′ − e_j)` (one vector, formed once) and
   `M∥′ = M′·n̂`; **null** unless `M∥′ > 0`;
4. `ΔE` from [`_pair_delta_energy!`](@ref) (exact, any body order);
5. `ℓ = −βΔE + 2 ln(M∥′/M∥) + ln|e_j·n̂| − ln|e_j′·n̂|`.

Returns the partner spin, `ΔE`, `ℓ`, and `M′` — the magnetization an accepted move
stores, bitwise the vector whose `M′·n̂` entered the null test and the Jacobian.
A null move returns `(e_j, NaN, -Inf, M)` without evaluating any energy; the
comparisons are written `!(s2 < 1)` and `!(M∥′ > 0)` so that a NaN anywhere in the
inputs also yields a null move. A partner with `e_j·n̂ = 0` exactly gives
`ℓ = −Inf` with a finite `ΔE` (the move is then always rejected, which is
correct — the measure has zero density for its reverse).

`e_j′` is re-normalized with `_unit_clamped` because the constructed vector is
unit only to rounding, and its components overshoot 1 in a measurable fraction of
near-axis proposals — the harmonics throw for `|e_j′·ẑ| > 1` and a checkpoint
restore refuses any component above 1. This is a deterministic, rounding-level
change of the proposal map, far below any statistical resolution; the same `e_j′`
— hence the same `e_j′·n̂` — then enters `M∥′`, the Jacobian, and `ΔE`, so `ℓ` is
exactly the log ratio of the move actually proposed. A sign change of `e_j′·n̂`
against `σ_j` (not reachable with `s2 < 1`, kept as a guard) is a null move.

Preconditions (the caller's), checked before any unchecked read:
- `i ≠ j` — `AssertionError`;
- `i, j ∈ 1:H.n_sites` — `BoundsError`;
- `i, j ∈ rc.cset` (nonzero weights) — `AssertionError`;
- `M` the current weighted magnetization with `M·n̂ > 0` — `ArgumentError`
  (also for a NaN `M`);
- `e_i_new` a unit vector with components in `[-1, 1]` (not checked).
"""
function _cmc_pair_proposal(config::SpinConfig, zrows::Matrix{Float64},
                            H::TiledHamiltonian, rc::_ResolvedConstraint,
                            M::SVector{3,Float64}, i::Int, j::Int,
                            e_i_new::SVector{3,Float64}, β::Float64,
                            scratch::_CMCScratch
                            )::Tuple{SVector{3,Float64},Float64,Float64,
                                     SVector{3,Float64}}
    @assert i != j "the partner must differ from the source; got i = j = $i"
    n̂ = rc.nhat
    w_i, w_j = rc.site_w[i], rc.site_w[j]         # checked reads: BoundsError
    @assert w_i != 0.0 && w_j != 0.0 "sites $i and $j must both be constrained"
    Mpar = dot(M, n̂)
    Mpar > 0.0 || throw(ArgumentError(
        "current M·n̂ = $Mpar is not positive: the state violates the constraint"))
    e_i, e_j = config[i], config[j]
    z_j = dot(e_j, n̂)
    null = (e_j, NaN, -Inf, M)
    # (1) compensate the perpendicular part (frame-free projections onto n̂⊥);
    #     p = e_j⊥′
    d = e_i_new - e_i
    dperp = d - dot(d, n̂) * n̂
    p = (e_j - z_j * n̂) - (w_i / w_j) * dperp
    s2 = dot(p, p)
    !(s2 < 1.0) && return null
    # (2) restore the unit length, keeping the sign of the parallel part
    σ_j = z_j >= 0.0 ? 1.0 : -1.0
    e_j_new = _unit_clamped(p + (σ_j * sqrt(1.0 - s2)) * n̂)
    z_j′ = dot(e_j_new, n̂)
    (z_j′ >= 0.0) == (σ_j > 0.0) || return null  # guard; unreachable with s2 < 1
    # (3) the new parallel magnetization must stay positive. M′ is formed ONCE, as a
    #     vector: the same value decides the null test, enters the Jacobian, and is
    #     what an accepting caller stores, so the three agree bitwise.
    M_new = M + w_i * (e_i_new - e_i) + w_j * (e_j_new - e_j)
    Mpar_new = dot(M_new, n̂)
    !(Mpar_new > 0.0) && return null
    # (4)–(5) exact joint energy change and the log acceptance ratio
    ΔE = _pair_delta_energy!(scratch, zrows, H, i, j, e_i_new, e_j_new)
    ℓ = -β * ΔE + 2.0 * log(Mpar_new / Mpar) + log(abs(z_j)) - log(abs(z_j′))
    return (e_j_new, ΔE, ℓ, M_new)
end

# --- the sweep -----------------------------------------------------------------------

# Absolute tolerance of the transverse-magnetization drift check, in units of
# `Σ_{s∈𝒞} |w_s|` (D3). Absolute, not relative to `M∥`: a start near the
# `|M_init|` door (`|M| ≈ 1e-8 Σ|w|`) keeps `|M⊥|/M∥` at ~2e-8 after its rotation,
# which a relative 1e-10 would report as drift.
const _CMC_DRIFT_TOL = 1e-10

# Threshold of the weak-constraint check: `M∥ / Σ_{s∈𝒞} |w_s|` below it at the end
# of thermalization means the constrained direction carries little information.
const _CMC_WEAK_TOL = 0.05

# The drift check run at every sweep start on the from-scratch `M`: warns (once)
# when `|M⊥| > 1e-10 · Σ|w|`. The compensation never reads `M⊥`, so rounding
# leaves it as a random walk (a 1e10-attempt estimate of ~1e-11 rad); a value
# above the bound means the state was corrupted, not rounded. Returns whether it
# fired. `@noinline` keeps the logging machinery out of the sweep body.
@noinline function _cmc_check_drift(rc::_ResolvedConstraint, M::SVector{3,Float64})::Bool
    n̂ = rc.nhat
    perp = norm(M - dot(M, n̂) * n̂)
    tol = _CMC_DRIFT_TOL * rc.wnorm
    perp > tol || return false
    @warn "constrained MC: the transverse magnetization |M⊥| = $perp exceeds " *
          "$(_CMC_DRIFT_TOL)·Σ|w| = $tol at a sweep start; the pair moves keep M⊥ " *
          "fixed only to rounding, so the configuration was modified outside the " *
          "constrained sweep (or is corrupted)" maxlog = 1
    return true
end

"""
    _cmc_check_step_floor(st::ChainState) -> Bool

End-of-thermalization check of a constrained chain (the driver calls it once the
step adaptation stops): warns (once) when the adapted step sits at the lower clamp
`_STEP_MIN` of the adaptation (`_adapt_step!`), and returns whether it fired. In CMC the usual
cause is a high fraction of null moves — pairs with a large weight ratio, whose
compensation leaves the sphere for all but tiny rotations — which the acceptance
counters (null moves count as rejected attempts) read as a step that is too large.
"""
function _cmc_check_step_floor(st::ChainState)::Bool
    st.step <= _STEP_MIN || return false
    @warn "constrained MC: the adapted step is pinned at its lower clamp " *
          "($(_STEP_MIN) rad) at the end of thermalization. Null moves count as " *
          "rejected attempts, so a high null fraction (pairs with a large weight " *
          "ratio |w_i/w_j|) drives the step down; the measure stays correct but the " *
          "chain mixes slowly — consider weight-0 (hybrid) sites for the small " *
          "moments" maxlog = 1
    return true
end

"""
    _cmc_check_weak_constraint(rc::_ResolvedConstraint, config::SpinConfig) -> Bool

End-of-thermalization check of a constrained chain: warns (once) when
`M∥ / Σ_{s∈𝒞} |w_s| < 0.05` (`M` from scratch), and returns whether it fired. The
direction of a nearly vanishing `M` carries little information about the
anisotropy: above the ordering temperature (finite-N remnant,
`M∥/Σ|w| ~ 1/√|𝒞|`), or for an antiferromagnet / near a compensation point.
"""
function _cmc_check_weak_constraint(rc::_ResolvedConstraint,
                                    config::SpinConfig)::Bool
    ratio = dot(_cmc_magnetization(rc, config), rc.nhat) / rc.wnorm
    ratio < _CMC_WEAK_TOL || return false
    @warn "constrained MC: weak constraint — M∥/Σ|w| = $ratio < $(_CMC_WEAK_TOL) at " *
          "the end of thermalization. Either the run is above the ordering " *
          "temperature (a finite-size remnant M, M∥/Σ|w| ~ 1/√|𝒞|), or the weights " *
          "nearly cancel (an antiferromagnet or a compensation point: give the " *
          "antiparallel sublattice a negative weight, or weight 0 to make it a " *
          "hybrid single-spin sublattice). The direction of a vanishing M says " *
          "little about the anisotropy" maxlog = 1
    return true
end

# One CMC pair attempt with source `i ∈ 𝒞`, `pos` its position in `rc.cset`
# (`rc.cset[pos] == i`), all randomness from the site's own stream `rng`, in the
# fixed order (D3):
#   1. partner: one call `rand(rng, 1:|𝒞|-1)` (Random's range sampler, which may
#      redraw internally — deterministically for a given stream state), shifted
#      past `pos` — uniform over 𝒞 \ {i}, with no rejection loop of our own;
#   2. rotation axis (`_random_unit`), then angle `step · randn` — the source
#      proposal of `_attempt_metro!` without the antipodal flip;
#   3. the accept uniform, ONLY when the move is not null and `ℓ < 0`.
# A null move (`ΔE` NaN) is a rejected attempt that draws no uniform; it is
# counted by the caller like any other attempt (re-drawing instead would sample
# π(x)(1 − r(x)), not π). On accept: spins `i`, `j`, their `zrows` columns (the
# rows `_pair_delta_energy!` left in `scratch`), and the staged `dE[i] = ΔE` (the
# joint change of both spins; each source is visited once per sweep, so the slot is
# free) are written, and the returned `M` is the proposal's `M′` — bitwise the value
# its null test and Jacobian used. Returns `(accepted::Int, M)`.
@inline function _attempt_cmc_pair!(config::SpinConfig, zrows::Matrix{Float64},
                                    H::TiledHamiltonian, rc::_ResolvedConstraint,
                                    M::SVector{3,Float64}, β::Float64, step::Float64,
                                    i::Int, pos::Int, scratch::_CMCScratch,
                                    rng::Xoshiro, dE::Vector{Float64}
                                    )::Tuple{Int,SVector{3,Float64}}
    cset = rc.cset
    k = rand(rng, 1:(length(cset) - 1))
    k >= pos && (k += 1)
    j = cset[k]
    e_i_new = _rotate(config[i], _random_unit(rng), step * randn(rng))
    e_j_new, ΔE, ℓ, M_new = _cmc_pair_proposal(config, zrows, H, rc, M, i, j,
                                                e_i_new, β, scratch)
    isnan(ΔE) && return (0, M)                      # null move: no accept uniform
    if ℓ >= 0.0 || log(rand(rng)) < ℓ
        config[i] = e_i_new
        config[j] = e_j_new
        copyto!(view(zrows, :, i), scratch.zi_new)
        copyto!(view(zrows, :, j), scratch.zj_new)
        @inbounds dE[i] = ΔE
        return (1, M_new)
    end
    return (0, M)
end

"""
    cmc_sweep!(st::ChainState, H::TiledHamiltonian, β, sc::SweepScratch,
               rc::_ResolvedConstraint, scratch::_CMCScratch) -> (nacc, M)

One constrained Monte Carlo sweep (`docs/specs/constrained-mc.md` D3): one attempt
per **active** site, in site order (inactive sites are skipped and stay bitwise
frozen), serial only.

- `s ∈ 𝒞` (`rc.site_w[s] != 0`): one pair attempt with `s` as the source and a
  partner uniform over `𝒞 \\ {s}` (D2/D3);
- active with weight 0: one plain single-spin Metropolis attempt
  (`_attempt_metro!`, antipodal flip included) — these spins do not enter `M`, so
  they move freely under the constraint (hybrid CMC).

All randomness of the attempt at site `s` comes from `st.site_rngs[s]`; `st.rng`
is not used. `M` is recomputed from scratch (`_cmc_magnetization`) at the start —
nothing is carried across sweeps, so `ChainState` holds no constraint state — where
a transverse part above `1e-10 Σ|w|` triggers a one-time warning; within the sweep
it is updated with each accepted pair. Accepted energy changes are staged in
`sc.dE` and reduced in the fixed order of `_reduce_dE`. Both kinds of attempt are
counted in `acc_metro` / `att_metro` (null moves are rejected attempts), so
`_adapt_step!` applies unchanged. Allocation-free.

Returns the number of accepted moves and the incrementally maintained `M` at the
end of the sweep (equal to a from-scratch sum up to rounding). The returned `M` is
for diagnostics and tests only: a driver must not carry it into the next sweep —
the from-scratch recompute at every sweep start is what keeps the chain a function
of `(config, RNG streams)` alone, hence resume ≡ uninterrupted.

Preconditions: `rc` and `scratch` built for `H` (sizes asserted), and a state of
the constrained ensemble (`M·n̂ > 0`; the pair proposal throws otherwise).
"""
function cmc_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                    sc::SweepScratch, rc::_ResolvedConstraint,
                    scratch::_CMCScratch)::Tuple{Int,SVector{3,Float64}}
    @assert length(scratch.lc) == H.nlm && length(rc.site_w) == H.n_sites &&
            length(sc.dE) == H.n_sites "CMC buffers were built for another Hamiltonian"
    M = _cmc_magnetization(rc, st.config)
    _cmc_check_drift(rc, M)
    dE = sc.dE
    fill!(dE, 0.0)
    nacc = 0
    pos = 0                                  # position of the source in rc.cset
    @inbounds for s = 1:H.n_sites
        H.site_active[s] || continue
        rng = st.site_rngs[s]
        if rc.site_w[s] != 0.0
            pos += 1                         # cset is ascending: site order = cset order
            a, M = _attempt_cmc_pair!(st.config, st.zrows, H, rc, M, β, st.step, s,
                                      pos, scratch, rng, dE)
        else
            a = _attempt_metro!(st.config, st.zrows, H, β, st.step, s, sc, rng, dE)
        end
        nacc += a
    end
    st.energy += _reduce_dE(H, dE)
    st.acc_metro += nacc
    st.att_metro += H.n_active
    return (nacc, M)
end

# Analytic Zeeman torque of the full system, `MU_B_EV_T · (M_μ × B)` with
# `M_μ = Σ_{active s} m_{a(s)} e_s` (μ_B; no `1/n_cells`). Sign: the Zeeman energy
# is `E_Z = −MU_B_EV_T Σ m (e·B)`; a global rotation by `α` about the unit axis `w`
# moves `e → e + α w × e`, so `dE_Z/dα = −MU_B_EV_T Σ m (w × e)·B
# = −w · MU_B_EV_T (M_μ × B)`, and the torque convention `T·w = −dE/dα`
# (`docs/specs/constrained-mc.md`) gives `T_Z = +MU_B_EV_T (M_μ × B)` — the moment
# is pulled toward `B`. Zero without a field. This is the part of
# [`total_torque`](@ref) the external field contributes, which partial
# (field-free) Hamiltonians leave out.
function _zeeman_torque(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    mm = H.magmoms
    (mm === nothing || !has_field(H)) && return zero(SVector{3,Float64})
    M = zero(SVector{3,Float64})
    for s = 1:H.n_sites
        H.site_active[s] || continue
        M += mm[site_atom(H, s)] * config[s]
    end
    return MU_B_EV_T * cross(M, H.field)
end

# Value identity of two templates on every field that fixes their energy
# (atoms, shifts, ls, folded, coef). `ScaledTerm` holds arrays, so the default
# `==` (identity) would never match two separately constructed templates.
_term_key(t::ScaledTerm) = (t.coef, t.atoms, t.shifts, t.ls, t.folded)

"""
    torque_observable(name::Symbol, Hpart::TiledHamiltonian, H::TiledHamiltonian)
        -> Observable

A 3-component [`Observable`](@ref) named `name` that measures the torque of the
term-subset Hamiltonian `Hpart` on the configuration of a run on `H`:
`total_torque(Hpart, config) / n_cells` (Cartesian, per training cell, unweighted —
the normalization of `:torque`; see [`total_torque`](@ref) and
`docs/specs/constrained-mc.md`).

Build `Hpart` with the public constructor from the SAME term list that built `H`
(after [`reduce_cell`](@ref): `red.terms`), filtered to the subset of interest and
without a field — e.g. `TiledHamiltonian(n, filter(t -> t.body == 1, terms); dims)`
for the on-site part. Over a disjoint cover of the fitted terms the partial
torques plus the analytic Zeeman torque sum to the total torque, configuration by
configuration; an individual partial torque is a free-energy gradient only to
first order in the anisotropy.

Doors (`ArgumentError`): `Hpart` and `H` must agree on `n_cell_atoms`, `dims`, and
`n_sites`; `Hpart` must carry no field (the Zeeman part is measured analytically);
every template of `Hpart` must equal — in atoms, shifts, `ls`, `folded`, and the
scaled coefficient — one of the fitted templates `H.terms[1:H.n_fitted_terms]`, so
a subset of a different term list (another cell, another fit) is refused even when
the atom counts happen to agree. Disjointness and coverage are not enforced.

The observable is bound to the `H` object it was built with: measuring it in a run
on any other `TiledHamiltonian` (even an equal one) throws an `ArgumentError`, so
build it with the same `H` the run uses. `Hpart` and `H` are immutable and the
returned function captures no mutable state, so the observable may be shared across
lanes that sample the same `H`.
"""
function torque_observable(name::Symbol, Hpart::TiledHamiltonian,
                           H::TiledHamiltonian)::Observable
    Hpart.n_cell_atoms == H.n_cell_atoms || throw(ArgumentError(
        "torque_observable: Hpart has $(Hpart.n_cell_atoms) cell atoms, H has " *
        "$(H.n_cell_atoms)"))
    Hpart.dims == H.dims || throw(ArgumentError(
        "torque_observable: Hpart.dims = $(Tuple(Hpart.dims)) differs from " *
        "H.dims = $(Tuple(H.dims))"))
    Hpart.n_sites == H.n_sites || throw(ArgumentError(
        "torque_observable: Hpart has $(Hpart.n_sites) sites, H has $(H.n_sites)"))
    has_field(Hpart) && throw(ArgumentError(
        "torque_observable: Hpart carries an external field; build partial " *
        "Hamiltonians field-free (the Zeeman torque is measured analytically)"))
    fitted = Set(_term_key(t) for t in view(H.terms, 1:H.n_fitted_terms))
    for (k, t) in enumerate(Hpart.terms)
        _term_key(t) in fitted || throw(ArgumentError(
            "torque_observable: template $k of Hpart (atoms = $(t.atoms), " *
            "shifts = $(t.shifts), ls = $(t.ls)) matches no fitted template of " *
            "H; build Hpart from the same term list as H"))
    end
    ncells = prod(H.dims)
    return Observable(name, 3, (config, E, Hrun) -> begin
        Hrun === H || throw(ArgumentError(
            "torque_observable $name was built for a different TiledHamiltonian " *
            "than the one this run samples; build it with the run's own H"))
        total_torque(Hpart, config) / ncells
    end)
end

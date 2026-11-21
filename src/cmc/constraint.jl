# Constrained Monte Carlo (CMC), part 1 of 4: the constraint — its public type, its
# resolution against a Hamiltonian, the weighted magnetization it constrains, and the
# deterministic constrained start. The standing contract is
# `docs/specs/constrained-mc.md` (D1). The other parts: `pair_move.jl` (the pair move
# and the sweep), `driver.jl` (observables, result, single-chain `run_cmc`),
# `lanes.jl` (direction lanes over threads); the torque observables live next to
# `total_torque` in observables.jl.

const _CMC_WEIGHT_RULES = (:auto, :uniform, :magmoms)

# The `|M_init|` door of the constrained start, in units of `Σ_{s∈𝒞} |w_s|`: an
# `init` whose weighted magnetization is shorter has no direction to rotate (D1).
const _CMC_INIT_MIN_M = 1e-8

# Below this length of the component of `b` orthogonal to `a` (with `a·b < 0`) the
# plane of the minimal rotation `a → b` is rounding noise, and the rotation takes the
# half-turn branch (D1).
const _CMC_ANTIPARALLEL_TOL = 1e-8

# The read-side consistency door of a "cmc" checkpoint: a stored configuration must
# have `|M⊥| ≤ _CMC_RESTORE_TOL · Σ_{s∈𝒞} |w_s|` (D6). A legal file sits at the
# rounding level, far inside.
const _CMC_RESTORE_TOL = 1e-8

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

# Whether `n̂` is a finite unit vector to 4 eps — the test a stored or carried-over
# constraint direction must pass (it is never normalized again).
_is_unit_direction(n̂::SVector{3,Float64})::Bool =
    all(isfinite, n̂) && abs(norm(n̂) - 1.0) <= 4 * eps()

# Tag of the internal non-normalizing constructor of `MagnetizationConstraint`.
struct _Validated end

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

    # Internal: rebuild from fields that already passed the checks below, WITHOUT
    # re-normalizing — normalization is not idempotent at the ulp level (about a
    # third of normalized vectors move by an ulp when normalized again), so this is
    # how a resolved constraint carries its direction over bitwise (`CMCResult`).
    function MagnetizationConstraint(direction::SVector{3,Float64},
                                     weights::Vector{Float64}, ::_Validated)
        @assert(all(isfinite, weights) && _is_unit_direction(direction),
                "_Validated constraint fields must already be checked")
        return new(direction, weights)
    end

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
# - `direction`         — the unit constraint direction n̂;
# - `cell_weights`      — the resolved per-cell-atom weights (what a checkpoint and
#                         the result record);
# - `site_weights`      — the weight of every supercell site,
#                         `cell_weights[site_atom(s)]` on the constrained set and 0.0
#                         elsewhere (inactive sites included), so
#                         `site_weights[s] != 0` exactly when
#                         `s ∈ constrained_sites`;
# - `constrained_sites` — the constrained sites 𝒞 (active, nonzero weight),
#                         ascending;
# - `weight_norm`       — `Σ_{s∈𝒞} |w_s|` (`sum(abs, site_weights)`), the scale of the
#                         start, drift, weak-constraint, and restore checks.
struct _ResolvedConstraint
    direction::SVector{3,Float64}
    cell_weights::Vector{Float64}
    site_weights::Vector{Float64}
    constrained_sites::Vector{Int}
    weight_norm::Float64
end

# Resolve `con` against `H` (docs/specs/constrained-mc.md D1): symbolic weights,
# the weight-vector length, and the door `|𝒞| ≥ 2` (a pair move needs a partner).
function _resolve_constraint(H::TiledHamiltonian,
                             con::MagnetizationConstraint)::_ResolvedConstraint
    n = H.n_cell_atoms
    w = con.weights
    mm = H.magmoms
    # branch on the type first, so each arm sees a concrete `w`
    cell_weights = if w isa Vector{Float64}
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
    site_weights = zeros(H.n_sites)
    constrained_sites = Int[]
    for s = 1:H.n_sites
        ws = cell_weights[site_atom(H, s)]
        (H.site_active[s] && ws != 0.0) || continue
        site_weights[s] = ws
        push!(constrained_sites, s)
    end
    length(constrained_sites) >= 2 || throw(ArgumentError(
        "the constrained set (active sites with nonzero weight) has " *
        "$(length(constrained_sites)) site(s); a constrained run needs at least 2"))
    return _ResolvedConstraint(con.direction, cell_weights, site_weights,
                               constrained_sites, sum(abs, site_weights))
end

# The record of a resolved constraint — what `CMCResult.constraint` reports and a
# "cmc" checkpoint stores: the direction bitwise (no second normalization) and the
# resolved per-cell-atom weights. `.+ 0.0` copies and folds signed zeros (a `-0.0`
# moment passes the field door), so the record agrees in `==`, `isequal`, and
# `hash` with a constraint built from the same numbers.
_constraint_record(rc::_ResolvedConstraint)::MagnetizationConstraint =
    MagnetizationConstraint(rc.direction, rc.cell_weights .+ 0.0, _Validated())

# The constrained magnetization `M = Σ_{s∈𝒞} w_s e_s`, from scratch, summed in
# `constrained_sites` order (deterministic).
function _cmc_magnetization(rc::_ResolvedConstraint,
                            config::SpinConfig)::SVector{3,Float64}
    M = zero(SVector{3,Float64})
    @inbounds for s in rc.constrained_sites
        M += rc.site_weights[s] * config[s]
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
    n̂ = rc.direction
    if init === nothing
        config = SpinConfig(undef, H.n_sites)
        for s = 1:H.n_sites
            w = rc.site_weights[s]
            config[s] = w < 0.0 ? -n̂ : n̂
        end
        return config
    end
    config = _validated_init(H, init)
    M = _cmc_magnetization(rc, config)
    limit = _CMC_INIT_MIN_M * rc.weight_norm
    norm(M) >= limit || throw(ArgumentError(
        "the weighted magnetization of init has |M| = $(norm(M)) < 1e-8·Σ|w| = " *
        "$limit: its direction is undefined, so it cannot be rotated onto the " *
        "constraint direction"))
    R = _min_rotation(M / norm(M), n̂)
    return SpinConfig([_unit_clamped(R * e) for e in config])
end

# The resolved constraint and the constrained start of a run — the doors shared by
# the single-chain driver and the up-front validation of the lane method.
function _cmc_setup(H::TiledHamiltonian, constraint::MagnetizationConstraint,
                    init)::Tuple{_ResolvedConstraint,SpinConfig}
    rc = _resolve_constraint(H, constraint)
    return (rc, _cmc_initial_config(H, rc, init))
end

# The minimal rotation taking the unit vector `a` to the unit vector `b`, as a 3×3
# matrix. In the plane of `a` and `b`, with `û` the unit vector along the component
# `u` of `b` orthogonal to `a` and `s = |u|`: `R a = (a·b) a + s û = b`,
# `R û = −s a + (a·b) û`, and the normal direction is fixed, i.e.
# `R = I + (a·b − 1)(a aᵀ + û ûᵀ) + s (û aᵀ − a ûᵀ)`. Written with `u` itself (not
# an axis `a × b / |a × b|`), and `û` made orthogonal to `a` by a second
# Gram–Schmidt pass, so `R` is orthogonal and `R a = b` to rounding.
#
# Near `a = −b` (`a·b < 0` and `s < _CMC_ANTIPARALLEL_TOL`) a separate branch is
# needed: for an exactly antiparallel pair `u = 0` and the formula has no plane at
# all (`s == 0` would return the identity), and just off it the plane is set by
# rounding noise. There the rotation is the half turn `P = 2 k kᵀ − I` about the
# conventional axis `k = normalize(b × x̂)` (`ŷ` when `b ∥ x̂`; the unnormalized
# `b × x̂` is exactly orthogonal to `b`, the normalized `k` to rounding), followed by
# the minimal rotation from `P a` (now within the tolerance of `b`) to `b`. For
# `a = −b` exactly this is the half turn alone, to rounding.
function _min_rotation(a::SVector{3,Float64},
                       b::SVector{3,Float64})::SMatrix{3,3,Float64,9}
    Id = SMatrix{3,3,Float64,9}(I)
    c = dot(a, b)
    u = b - c * a
    u -= dot(u, a) * a
    s = norm(u)
    if c < 0.0 && s < _CMC_ANTIPARALLEL_TOL
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

# The read-side consistency door of a "cmc" checkpoint (D6), non-projecting: the
# stored configuration at `key` must satisfy the stored constraint `rc` — `M·n̂ > 0`
# and `|M⊥| ≤ 1e-8 Σ|w|` for `M = Σ_{s∈𝒞} w_s e_s`. A legal file is far inside both
# bounds (the start points along n̂ to rounding, and the pair moves keep `|M⊥|` at
# the rounding level, `docs/specs/constrained-mc.md` G1), so a refusal means the
# direction, the weights, or the configuration were rewritten; resuming such a
# file would sample another ensemble with no error. NaN-safe (both tests fail).
function _check_cmc_config(rc::_ResolvedConstraint, config::SpinConfig,
                           key::String)
    M = _cmc_magnetization(rc, config)
    Mpar = dot(M, rc.direction)
    Mperp = norm(M - Mpar * rc.direction)
    limit = _CMC_RESTORE_TOL * rc.weight_norm
    (Mpar > 0.0 && Mperp <= limit) || error(
        "checkpoint $key does not satisfy the stored constraint: M·n̂ = $Mpar, " *
        "|M⊥| = $Mperp (limit 1e-8·Σ|w| = $limit) — corrupted file?")
    return nothing
end

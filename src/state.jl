# Mutable chain state and thread-confined scratch — deliberately separated from the
# immutable `TiledHamiltonian` and from run configuration (no God-struct).

"""
    ChainState

The mutable state of one Markov chain: the spin `config` with its cached tesseral
rows `zrows` (column `s` = `Z_lm(e_s)`), the incrementally maintained total `energy`
(model units, `j0` excluded — kept exact by the ΔE bookkeeping and re-anchored at
every renormalization), the chain-owned `rng` (config resets) plus one independent
proposal/accept stream per site (`site_rngs`, derived from `rng` at construction —
what makes the sweeps deterministic regardless of how many tasks execute a color
class), the Metropolis proposal `step` (radians; adapted during thermalization,
frozen once `frozen` is set), windowed acceptance counters, and the worst
incremental-energy `max_drift` observed at renormalization points.
"""
mutable struct ChainState
    # config/zrows/energy are the swappable "payload" of a replica-exchange move
    # (`_swap_payload!` exchanges the references) — hence not `const`. The RNG
    # streams stay with the lane, like `rng`.
    config::SpinConfig
    zrows::Matrix{Float64}
    energy::Float64
    const rng::Xoshiro
    const site_rngs::Vector{Xoshiro}
    step::Float64
    frozen::Bool
    acc_metro::Int
    att_metro::Int
    acc_or::Int
    att_or::Int
    max_drift::Float64
end

function ChainState(H::TiledHamiltonian, config::SpinConfig, rng::Xoshiro,
                    step::Real)
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    zrows = _zrows(H, config)
    # One-word seeding is sound here: Julia ≥ 1.11 expands an integer seed into
    # the five-word Xoshiro state through a SHA-2-based hash, so sequentially
    # drawn seeds give effectively independent streams (a 64-bit birthday
    # collision — two sites sharing a proposal stream — has P ≈ n²/2⁶⁵).
    site_rngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:H.n_sites]
    return ChainState(config, zrows, _total_energy(H, zrows), rng, site_rngs,
                      Float64(step), false, 0, 0, 0, 0, 0.0)
end

Base.show(io::IO, st::ChainState) =
    print(io, "ChainState(", length(st.config), " sites, E=",
          @sprintf("%.6g", st.energy), ", step=", @sprintf("%.3g", st.step),
          st.frozen ? ", frozen" : "", ")")

"""
    SweepScratch(H::TiledHamiltonian)

Per-task scratch buffers for the sweep kernels (`c` — leave-one-out coefficients,
`znew` — the proposed spin's tesseral row, `plm` — the associated-Legendre
recursion workspace of the internal `_zlm_row!`, `dE` — the per-site accepted-ΔE
staging buffer the deterministic energy reduction reads back in color order; when
several tasks execute one sweep, only the **first** scratch's `dE` is used — its
writes are per-site disjoint). One per task, never shared across tasks.
"""
struct SweepScratch
    c::Vector{Float64}
    znew::Vector{Float64}
    plm::Vector{Float64}
    dE::Vector{Float64}
end

SweepScratch(H::TiledHamiltonian) =
    SweepScratch(zeros(H.nlm), zeros(H.nlm), Vector{Float64}(undef, H.lmax + 1),
                 zeros(H.n_sites))

# --- configuration helpers ----------------------------------------------------------

# Uniform random unit vector (Gaussian-normalized). An (astronomically improbable)
# near-zero draw would put a NaN spin into the chain; redraw instead. The extra
# branch never fires in practice, so RNG consumption — and bit-determinism — are
# unchanged on the no-retry path.
function _random_unit(rng::AbstractRNG)::SVector{3,Float64}
    while true
        v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
        n = norm(v)
        n > 1e-12 && return v / n
    end
end

# Resolve a chain start: `nothing` → uniform random from `rng`; a `3 × n_sites`
# matrix or a vector of 3-vectors → validated-then-projected copy (`_unit_column`).
function _initial_config(H::TiledHamiltonian, init, rng::AbstractRNG)::SpinConfig
    init === nothing &&
        return SpinConfig([_random_unit(rng) for _ = 1:H.n_sites])
    if init isa AbstractMatrix
        size(init) == (3, H.n_sites) || throw(DimensionMismatch(
            "init is $(size(init, 1))×$(size(init, 2)); expected 3×$(H.n_sites)"))
        return SpinConfig([_unit_column(SVector{3,Float64}(init[1, s], init[2, s],
                                                           init[3, s]), s)
                           for s = 1:H.n_sites])
    end
    length(init) == H.n_sites || throw(DimensionMismatch(
        "init has $(length(init)) sites; expected $(H.n_sites)"))
    return SpinConfig([_unit_column(SVector{3,Float64}(e), s)
                       for (s, e) in enumerate(init)])
end

# The family's unit-direction tolerance: the same 1e-6 band SCEFitting's config
# door (`predict_energy` et al.) enforces, so the two packages accept and refuse
# the same inputs.
const _UNIT_ATOL = 1.0e-6

# The family's projecting unit-direction door (ONE rule: finite,
# `|‖e‖ − 1| ≤ 1e-6`, component bound of the projected value), applied per column
# wherever a caller hands this package a spin state — `from_matrix` and the
# drivers' `init`. A wildly-scaled column (a moment vector, `‖e‖ = 1.7`) is
# REFUSED, never silently normalized: past the band the input is a different
# physical quantity wearing the wrong units, and the caller should normalize
# deliberately where the decision is visible. (Until the 3f71644 backport this
# door normalized anything nonzero — accepting here exactly what SCEFitting's
# doors refuse.) Every column is validated, spin-active or not: `_zlm_row!`
# evaluates the tesseral row at EVERY site's spin, so even a never-read
# placeholder must stay inside the Legendre domain.
# [Backported from SLCEMonteCarlo.jl 3f71644; upstream routes through
# SLCE.UnitVector3, which the revived SCEFitting does not carry — the rule is
# implemented locally instead.]
function _unit_column(e::SVector{3,Float64}, s::Int)::SVector{3,Float64}
    all(isfinite, e) || throw(ArgumentError(
        "spin column $s is not finite: $e"))
    n = norm(e)
    abs(n - 1.0) <= _UNIT_ATOL || throw(ArgumentError(
        "spin column $s has norm $n, more than $_UNIT_ATOL from unit: this is not " *
        "a direction (a moment-scaled vector?). Normalize deliberately in your " *
        "own code if that is what you mean."))
    u = e / n
    maximum(abs, u) <= 1.0 || throw(ArgumentError(
        "spin column $s leaves the Legendre domain even after projection: $u"))
    return u
end

# The non-projecting half of the same rule, for stored state that must restore
# bit-exactly (a projection here would ULP-perturb the chain): validate, keep bits.
function _validate_unit_column(e::SVector{3,Float64}, what::String)
    all(isfinite, e) || throw(ArgumentError("$what is not finite: $e"))
    abs(norm(e) - 1.0) <= _UNIT_ATOL || throw(ArgumentError(
        "$what has norm $(norm(e)), more than $_UNIT_ATOL from unit: the stored " *
        "state is corrupted (or was written by something other than this package)"))
    maximum(abs, e) <= 1.0 || throw(ArgumentError(
        "$what has a component outside the Legendre domain: $e"))
    return nothing
end

# Replace the chain's configuration in place (fresh restart): rebuild the tesseral
# rows and recompute the energy from scratch (no drift bookkeeping — this is not a
# renormalization of an evolved chain).
function _reset_config!(st::ChainState, H::TiledHamiltonian, config::SpinConfig)
    copyto!(st.config, config)
    plm = Vector{Float64}(undef, H.lmax + 1)
    for s = 1:H.n_sites
        _zlm_row!(view(st.zrows, :, s), st.config[s], H.lmax, plm)
    end
    st.energy = _total_energy(H, st.zrows)
    return st
end

# Renormalize every active spin, rebuild its tesseral rows, and re-anchor the
# incremental energy on a full recomputation. Records the observed drift; returns
# it. Inactive sites stay bitwise frozen (never updated, so no drift to fix; their
# zrows columns are never read).
function _renormalize!(st::ChainState, H::TiledHamiltonian,
                       plm::Vector{Float64})::Float64
    for s = 1:H.n_sites
        H.site_active[s] || continue
        e = normalize(st.config[s])
        st.config[s] = e
        _zlm_row!(view(st.zrows, :, s), e, H.lmax, plm)
    end
    E = _total_energy(H, st.zrows)
    drift = abs(st.energy - E)
    st.max_drift = max(st.max_drift, drift)
    if drift > 1e-8 * max(1.0, abs(E))
        @warn "incremental-energy drift $(drift) at renormalization (E = $E); " *
              "consider a smaller renorm_interval" maxlog = 1
    end
    st.energy = E
    return drift
end

# Convenience form (tests / scratch-less callers): allocates the workspace.
_renormalize!(st::ChainState, H::TiledHamiltonian)::Float64 =
    _renormalize!(st, H, Vector{Float64}(undef, H.lmax + 1))

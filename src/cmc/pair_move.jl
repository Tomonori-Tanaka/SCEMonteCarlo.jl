# Constrained Monte Carlo (CMC), part 2 of 4: the pair move — the exact joint
# energy difference, the pure pair proposal, one attempt with its fixed RNG order,
# the serial sweep, and the sweep-start / end-of-thermalization checks
# (`docs/specs/constrained-mc.md` D2, D3).
#
# This file only CALLS the energy kernels of energy.jl (`site_coeffs!`,
# `delta_energy`, `_zlm_row!`) and the single-site moves of updates.jl
# (`_rotate`, `_random_unit`, `_attempt_metro!`); it changes none of them.

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

With `n̂ = rc.direction`, weights `w_i = rc.site_weights[i]`,
`w_j = rc.site_weights[j]`, and `x⊥ = x − (x·n̂) n̂` (frame-free — valid for any
`n̂`):

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
- `i, j ∈ rc.constrained_sites` (nonzero weights) — `AssertionError`;
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
    n̂ = rc.direction
    w_i, w_j = rc.site_weights[i], rc.site_weights[j]         # checked reads: BoundsError
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
    n̂ = rc.direction
    perp = norm(M - dot(M, n̂) * n̂)
    tol = _CMC_DRIFT_TOL * rc.weight_norm
    perp > tol || return false
    @warn "constrained MC: the transverse magnetization |M⊥| = $perp exceeds " *
          "$(_CMC_DRIFT_TOL)·Σ|w| = $tol at a sweep start; the pair moves keep M⊥ " *
          "fixed only to rounding, so the configuration was modified outside the " *
          "constrained sweep (or is corrupted)" maxlog = 1
    return true
end

"""
    _cmc_check_step_floor(st::ChainState, kT::Float64, n̂::SVector{3,Float64}) -> Bool

End-of-thermalization check of a constrained chain (`run_cmc` calls it once per
temperature, when the step adaptation stops): warns when the adapted step sits at
the lower clamp `_STEP_MIN` of the adaptation (`_adapt_step!`), and returns whether
it fired. The warning names the temperature `kT` and the constraint direction `n̂`
(also as log keys) and is not rate-limited, so every affected temperature of a
ladder, and every run, reports. In CMC the usual cause is a high fraction of null
moves — pairs with a large weight ratio, whose compensation leaves the sphere for
all but tiny rotations — which the acceptance counters (null moves count as
rejected attempts) read as a step that is too large.
"""
function _cmc_check_step_floor(st::ChainState, kT::Float64,
                               n̂::SVector{3,Float64})::Bool
    st.step <= _STEP_MIN || return false
    @warn("constrained MC at kT = $kT, n̂ = $(Tuple(n̂)): the adapted step is " *
          "pinned at its lower clamp ($(_STEP_MIN) rad) at the end of " *
          "thermalization. Null moves count as rejected attempts, so a high null " *
          "fraction (pairs with a large weight ratio |w_i/w_j|) drives the step " *
          "down; the measure stays correct but the chain mixes slowly — consider " *
          "weight-0 (hybrid) sites for the small moments", kT, direction = Tuple(n̂))
    return true
end

"""
    _cmc_check_weak_constraint(rc::_ResolvedConstraint, config::SpinConfig,
                               kT::Float64) -> Bool

End-of-thermalization check of a constrained chain (`run_cmc` calls it once per
temperature): warns when `M∥ / Σ_{s∈𝒞} |w_s| < 0.05` (`M` from scratch), and
returns whether it fired. The warning names `kT` and `n̂` (also as log keys) and is
not rate-limited. The direction of a nearly vanishing `M` carries little
information about the anisotropy: above the ordering temperature (finite-N
remnant, `M∥/Σ|w| ~ 1/√|𝒞|`), or for an antiferromagnet / near a compensation
point.
"""
function _cmc_check_weak_constraint(rc::_ResolvedConstraint, config::SpinConfig,
                                    kT::Float64)::Bool
    ratio = dot(_cmc_magnetization(rc, config), rc.direction) / rc.weight_norm
    ratio < _CMC_WEAK_TOL || return false
    @warn("constrained MC at kT = $kT, n̂ = $(Tuple(rc.direction)): weak constraint — " *
          "M∥/Σ|w| = $ratio < $(_CMC_WEAK_TOL) at the end of thermalization. " *
          "Either the run is above the ordering temperature (a finite-size remnant " *
          "M, M∥/Σ|w| ~ 1/√|𝒞|), or the weights nearly cancel (an antiferromagnet " *
          "or a compensation point: give the antiparallel sublattice a negative " *
          "weight, or weight 0 to make it a hybrid single-spin sublattice). Weights " *
          "are per cell atom of H, so every sublattice that needs its own sign must " *
          "be its own cell atom: do not reduce_cell below the magnetic cell. The " *
          "direction of a vanishing M says little about the anisotropy",
          kT, ratio, direction = Tuple(rc.direction))
    return true
end

# One CMC pair attempt with source `i ∈ 𝒞`, `pos` its position in
# `rc.constrained_sites` (`rc.constrained_sites[pos] == i`), all randomness from the
# site's own stream `rng`, in the fixed order (D3):
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
    cset = rc.constrained_sites
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
    _cmc_sweep!(st::ChainState, H::TiledHamiltonian, β, sc::SweepScratch,
               rc::_ResolvedConstraint, scratch::_CMCScratch) -> (nacc, M)

One constrained Monte Carlo sweep (`docs/specs/constrained-mc.md` D3): one attempt
per **active** site, in site order (inactive sites are skipped and stay bitwise
frozen), serial only.

- `s ∈ 𝒞` (`rc.site_weights[s] != 0`): one pair attempt with `s` as the source and a
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
function _cmc_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                     sc::SweepScratch, rc::_ResolvedConstraint,
                     scratch::_CMCScratch)::Tuple{Int,SVector{3,Float64}}
    @assert length(scratch.lc) == H.nlm && length(rc.site_weights) == H.n_sites &&
            length(sc.dE) == H.n_sites "CMC buffers were built for another Hamiltonian"
    M = _cmc_magnetization(rc, st.config)
    _cmc_check_drift(rc, M)
    dE = sc.dE
    fill!(dE, 0.0)
    nacc = 0
    pos = 0                    # position of the source in rc.constrained_sites
    @inbounds for s = 1:H.n_sites
        H.site_active[s] || continue
        rng = st.site_rngs[s]
        if rc.site_weights[s] != 0.0
            pos += 1           # 𝒞 is ascending: site order = constrained_sites order
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

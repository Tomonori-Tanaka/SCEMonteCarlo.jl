# Update schemes (stationarity arguments: `docs/specs/updates-stationarity.md`).
#
# Sites are scanned in deterministic sequential order — a composition of per-site
# π-reversible kernels keeps the Boltzmann distribution stationary, consumes no RNG
# for site selection, and keeps runs bit-reproducible. β enters ONLY in the accept
# steps; coefficients and energies stay in the model's energy units.

# Antipodal-flip fraction of the Metropolis proposal: the Rodrigues rotation alone
# mixes slowly between the ±lobes of a strongly bimodal single-site potential.
const _FLIP_FRACTION = 0.2

# Rodrigues rotation of `e` about the unit `axis` by angle `θ` (an isometry; the
# proposal is symmetric because `θ ~ step·randn` is sign-symmetric and the axis is
# uniform).
@inline function _rotate(e::SVector{3,Float64}, axis::SVector{3,Float64},
                         θ::Float64)::SVector{3,Float64}
    c, s = cos(θ), sin(θ)
    return c * e + s * cross(axis, e) + (1 - c) * dot(axis, e) * axis
end

"""
    metropolis_sweep!(st::ChainState, H::TiledHamiltonian, β, sc::SweepScratch) -> Int

One single-spin Metropolis lattice sweep: `n_sites` sequential attempts with the
symmetric two-component proposal (antipodal flip with probability 0.2, else a
Rodrigues rotation by `st.step · randn` about a uniform axis) and the exact
`ΔE = c_s·ΔZ` from [`site_coeffs!`](@ref). Accepts with
`ΔE ≤ 0 || rand < exp(−β·ΔE)` (the uniform is drawn **only** when `ΔE > 0` — part
of the RNG-consumption contract that makes runs bit-reproducible). Mutates the
config/rows/energy in place; returns the number of accepted moves.
"""
function metropolis_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                           sc::SweepScratch)::Int
    nacc = 0
    rng = st.rng
    for s = 1:H.n_sites
        fill!(sc.c, 0.0)
        site_coeffs!(sc.c, H, s, st.zrows)
        e = st.config[s]
        e2 = if rand(rng) < _FLIP_FRACTION
            -e
        else
            _rotate(e, _random_unit(rng), st.step * randn(rng))
        end
        _zlm_row!(sc.znew, e2, H.lmax)
        ΔE = delta_energy(sc.c, view(st.zrows, :, s), sc.znew)
        if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
            st.config[s] = e2
            copyto!(view(st.zrows, :, s), sc.znew)
            st.energy += ΔE
            nacc += 1
        end
    end
    st.acc_metro += nacc
    st.att_metro += H.n_sites
    return nacc
end

"""
    overrelaxation_sweep!(st::ChainState, H::TiledHamiltonian, β, sc::SweepScratch)
        -> Int

One overrelaxation lattice sweep: at each site (sequentially), reflect the spin
about its local `l = 1` field axis — `e′ = 2(e·ĥ)ĥ − e`, with `ĥ` read off the
`l = 1` components of the leave-one-out coefficients (independent of `e` itself) —
and accept with the standard Metropolis rule on the **exact** ΔE.

The proposal is a deterministic involution (`S∘S = id`, an isometry of the sphere)
whose axis depends only on the other spins, so Metropolis acceptance gives detailed
balance outright; for a **pure-`l=1`** site channel the reflection conserves the
site energy exactly (`ΔE ≡ 0`, always accepted) and the move degenerates to
classical microcanonical overrelaxation — the accept step corrects the `l ≥ 2` /
multi-body remainder exactly (see `docs/specs/updates-stationarity.md`).

Not ergodic on its own (in the pure case it conserves `e·h` per move) — always mix
with Metropolis sweeps (`or_per_metropolis` in [`run_mc`](@ref)). Sites with no
`l = 1` channel (or a vanishing local field) are skipped and do not count as
attempts. Returns the number of accepted moves.
"""
function overrelaxation_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                               sc::SweepScratch)::Int
    nacc = 0
    natt = 0
    rng = st.rng
    for s = 1:H.n_sites
        H.site_has_l1[s] || continue
        fill!(sc.c, 0.0)
        site_coeffs!(sc.c, H, s, st.zrows)
        # Tesseral l = 1 row: Z_{1,-1} ∝ y, Z_{1,0} ∝ z, Z_{1,1} ∝ x (lm_index
        # slots 2, 3, 4) ⇒ the l=1 field direction is (c₄, c₂, c₃). A convention
        # drift upstream only rotates the axis (correctness is unaffected — the
        # axis is e-independent); the pure-l=1 ΔE ≡ 0 gate pins it.
        h = SVector(sc.c[4], sc.c[2], sc.c[3])
        hn = norm(h)
        hn < 1e-12 && continue
        axis = h / hn
        e = st.config[s]
        e2 = 2 * dot(e, axis) * axis - e
        natt += 1
        _zlm_row!(sc.znew, e2, H.lmax)
        ΔE = delta_energy(sc.c, view(st.zrows, :, s), sc.znew)
        if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
            st.config[s] = e2
            copyto!(view(st.zrows, :, s), sc.znew)
            st.energy += ΔE
            nacc += 1
        end
    end
    st.acc_or += nacc
    st.att_or += natt
    return nacc
end

# Multiplicative step adaptation toward the target acceptance, on the current
# counter window (then resets it). Thermalization only — once `st.frozen` the
# transition kernel must stay fixed (a history-dependent step is a bias source and
# breaks bit-reproducible resume), so this is a no-op.
function _adapt_step!(st::ChainState, target::Float64)::Float64
    st.frozen && return st.step
    if st.att_metro > 0
        a = st.acc_metro / st.att_metro
        st.step = clamp(st.step * exp(0.5 * (a - target)), 1e-3, Float64(π))
    end
    st.acc_metro = 0
    st.att_metro = 0
    return st.step
end

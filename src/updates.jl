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

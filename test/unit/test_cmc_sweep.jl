# Constrained MC, M4: the pair attempt and the CMC sweep
# (docs/specs/constrained-mc.md D2/D3), driven directly (the driver gates are in
# test_cmc_driver.jl).
#
# G1  — invariants over ≥ 10⁴ moves: the transverse weighted magnetization
#       recomputed here, the sweep's incremental M against the from-scratch sum,
#       the incremental energy against `total_energy`, the cached rows against a
#       rebuild; also right after `_renormalize!`.
# G1b — Fmix: the inactive site is bitwise frozen and outside M; the active
#       zero-weight site moves (hybrid single-spin Metropolis).
# G16 — `@allocated _cmc_sweep!(…) == 0` after warm-up.
# D3  — the RNG contract, replayed on copies of the streams: one pair attempt
#       (partner → axis → angle → accept uniform only when not null and ℓ < 0),
#       and whole sweeps (site order, the source's position in 𝒞, the hybrid
#       sites' streams, the chain stream untouched). A determinism check, not a
#       correctness oracle.
# G0  — the three warnings (sweep-start drift, step floor, weak constraint) fire
#       exactly when their conditions hold.
# G5  — theorem (I): for a constant energy, rotation-invariant quantities of the
#       constrained ensemble have their unconstrained distribution (closed forms).
# G6  — N = 2 dimers: the mean torque against a deterministic quadrature of the
#       constrained partition function Z₂(n̂).
# G7  — N = 3 linear response: K_eff = K ⟨P₂(e·M̂)⟩_iso, the average by quadrature.
# G5h — the hybrid w = 0 site: a spin coupled only to one constrained spin by an
#       isotropic pair follows the Langevin law against it, exactly.
#
# Statistical gates (G5–G7, G5h) run with a FIXED step (no adaptation), fixed seeds,
# and assert |mean − exact| < 5σ with σ the package's log-binning error, plus
# σ < εmax so a gate cannot pass by being noisy. The measured values, and the shift of the
# smallest mutation each gate must resolve, are recorded next to each gate and in
# the mutation table of docs/specs/constrained-mc.md.

using Logging: Warn

# Rodrigues rotation (test-side copy).
_sw_rot(e, w, α) = cos(α) * e + sin(α) * cross(w, e) + (1 - cos(α)) * dot(w, e) * w

_sw_perp(M, n̂) = norm(M - dot(M, n̂) * n̂)


# A constrained chain: resolved constraint, constrained start, fixed step.
function _sw_chain(H, n̂, w; init = nothing, step = 1.0, seed = 1)
    rc = MC._resolve_constraint(H, MC.MagnetizationConstraint(n̂; weights = w))
    st = MC.ChainState(H, MC._cmc_initial_config(H, rc, init), Xoshiro(seed), step)
    st.frozen = true                  # documentary: nothing here adapts the step
    return st, rc, MC.SweepScratch(H), MC._CMCScratch(H)
end

# Thermalize, then measure `f(st)` (a vector of `ncomp` reals) once per sweep into
# the package's log binner; returns (mean, σ) per component.
function _sw_sample(f, st, H, β, sc, rc, scratch; ntherm, nmeas, ncomp)
    for _ = 1:ntherm
        MC._cmc_sweep!(st, H, β, sc, rc, scratch)
    end
    b = MC.LogBinner(ncomp)
    for _ = 1:nmeas
        MC._cmc_sweep!(st, H, β, sc, rc, scratch)
        push!(b, f(st))
    end
    return mean(b), MC.std_error(b)
end


# ⟨P₂(e₁·M̂)⟩ for three iid uniform unit spins (the isotropic, unconstrained average;
# by theorem (I) also the constrained one at any n̂). With r = |e₂ + e₃| (density r/2
# on [0, 2]) and u = e₁·r̂ (uniform on [-1, 1]): R = |M| = √(1 + r² + 2ru) and
# e₁·M̂ = (1 + ru)/R = (R² + 1 − r²)/(2R). Substituting u → R (du = R dR / r) turns
# the weight (r/2)(du/2) into (R/4) dR on R ∈ [|1 − r|, 1 + r]:
#   ⟨P₂⟩ = ∫₀² dr ∫_{|1−r|}^{1+r} dR (R/4) P₂((R² + 1 − r²)/(2R)).
# In (r, u) the integrand jumps at M = 0 (r = 1, u = −1) and Gauss–Legendre converges
# only algebraically; in (r, R) the factor R removes the jump, and the remaining kink
# of the lower limit |1 − r| sits on the split of the r range at 1.
function _sw_p2_iso3(n)
    acc = 0.0
    for (a, b) in ((0.0, 1.0), (1.0, 2.0))
        rs, wr = _sw_gl(a, b, n)
        for (r, wrr) in zip(rs, wr)
            Rs, wR = _sw_gl(abs(1 - r), 1 + r, n)
            for (R, wRR) in zip(Rs, wR)
                x = (R^2 + 1 - r^2) / (2R)
                acc += wrr * wRR * (R / 4) * (3x^2 - 1) / 2
            end
        end
    end
    return acc
end

const _SW_N̂ = _sw_dir(0.6, 0.35)

# Per-site weights and active flags of a fixture, written from its definition (the
# atom weights; Fmix's one inactive atom) through the public `site_atom` — never
# from the package's resolution of the constraint.
_sw_site_w(H, w) = [w[MC.site_atom(H, s)] for s = 1:H.n_sites]
_sw_fmix_act(H) = [MC.site_atom(H, s) != _FMIX_INACTIVE for s = 1:H.n_sites]

# Replay one `_cmc_sweep!` on copies of the state and of every RNG stream, with the
# contract of docs/specs/constrained-mc.md D3 written out here: active sites in
# site order; a constrained source draws k = rand(1:|𝒞|−1) shifted past its
# position (counted here), axis = normalized 3 randn, angle = step·randn, and the
# accept uniform only when not null and ℓ < 0; a w = 0 site draws the flip uniform
# (flip with probability `_FLIP_FRACTION`), else axis and angle, and the accept
# uniform only when ΔE > 0 (ΔE from `total_energy` here). ℓ comes from the pure
# proposal: its value is G4's business, this replay pins the RNG consumption.
# Returns (config, streams, accepted count).
function _sw_replay_sweep(st, H, β, rc, w_site, act)
    cfg = copy(st.config)
    zr = copy(st.zrows)
    rs = [copy(r) for r in st.site_rngs]
    cset = [s for s = 1:H.n_sites if act[s] && w_site[s] != 0]
    aux = MC._CMCScratch(H)
    unit3(r) = (v = SVector(randn(r), randn(r), randn(r)); v / norm(v))
    pos = 0
    nacc = 0
    for s = 1:H.n_sites
        act[s] || continue
        r = rs[s]
        if w_site[s] != 0
            pos += 1
            k = rand(r, 1:(length(cset) - 1))
            j = cset[k >= pos ? k + 1 : k]
            e_new = _sw_rot(cfg[s], unit3(r), st.step * randn(r))
            e_j_new, ΔE, ℓ, _ = MC._cmc_pair_proposal(cfg, zr, H, rc,
                                                      _sw_M(cfg, w_site, act), s, j,
                                                      e_new, β, aux)
            if !isnan(ΔE) && (ℓ >= 0 || log(rand(r)) < ℓ)
                cfg[s], cfg[j] = e_new, e_j_new
                zr = MC._zrows(H, cfg)
                nacc += 1
            end
        else
            e = cfg[s]
            e_new = rand(r) < MC._FLIP_FRACTION ? -e :
                    _sw_rot(e, unit3(r), st.step * randn(r))
            y = copy(cfg)
            y[s] = e_new
            ΔE = total_energy(H, y) - total_energy(H, cfg)
            if ΔE <= 0 || rand(r) < exp(-β * ΔE)
                cfg = y
                zr = MC._zrows(H, cfg)
                nacc += 1
            end
        end
    end
    return cfg, rs, nacc
end

# --- statistical gates as functions of the seed --------------------------------------
# The testsets below call them with the pinned seeds.
#
# σ caps: εmax ≈ 1.7–2 × the largest binning σ of the row over 8 seeds (5501, 5601,
# 5701, 5801, 101–104; measured 2026-11-21, max σ in the comment). What 5·εmax still
# resolves — the smallest mutation each gate owns: G5 N=2 ⟨M∥⟩ 0.015 < 0.022 (null
# move re-drawn), weighted ⟨M∥²⟩ 0.11 < 0.17 (same); G6 F2u θ=0.3 T·ê_φ 8e-4 ≤ 8.9e-4
# (stale zrows column j), F2w− T·ê_φ 2.3e-3 < 4.6e-3 (σ_j forced +1); G7 T·ê_φ/K
# 0.038 < 0.050 (σ_j forced +1); G5h 0.025 < 0.15 (hybrid site at 0.5β).
const _SW_EPS = Dict(
    # G5 (max σ)
    "N=2 ⟨M∥⟩" => 3e-3,                     # 1.57e-3
    "N=2 ⟨M∥²⟩" => 7e-3,                    # 3.63e-3
    "N=2 P(we·n̂<0)" => 1e-3,               # identically 0
    "N=3 ⟨M∥⟩" => 5e-3,                     # 2.88e-3
    "N=3 ⟨M∥²⟩" => 1.6e-2,                  # 9.13e-3
    "N=3 P(we·n̂<0)" => 1.4e-3,             # 8.02e-4
    "w ⟨M∥⟩" => 5e-3,                       # 2.79e-3, both signs (bitwise equal)
    "w ⟨M∥²⟩" => 2.2e-2,                    # 1.13e-2
    # G6 (max σ)
    "F2u θ=0.3 T·ê_φ" => 1.6e-4,            # 8.99e-5
    "F2u θ=0.3 T·ê_θ" => 6e-5,              # 3.03e-5
    "F2u θ=0.3 T·n̂" => 1.8e-5,              # 9.37e-6
    "F2u θ=0.6 T·ê_φ" => 2.6e-4,            # 1.45e-4
    "F2u θ=0.6 T·ê_θ" => 9e-5,              # 4.81e-5
    "F2u θ=0.6 T·n̂" => 6e-5,                # 3.29e-5
    "F2u θ=1.2 T·ê_φ" => 1.7e-4,            # 9.21e-5
    "F2u θ=1.2 T·ê_θ" => 5e-5,              # 2.60e-5
    "F2u θ=1.2 T·n̂" => 1.2e-4,              # 6.69e-5
    "F2g T·ê_φ" => 6e-4,                    # 3.52e-4
    "F2g T·ê_θ" => 4.5e-4,                  # 2.68e-4
    "F2g T·n̂" => 4.5e-4,                    # 2.66e-4
    "F2w+ T·ê_φ" => 4.5e-4,                 # 2.58e-4
    "F2w+ T·ê_θ" => 4e-4,                   # 2.16e-4
    "F2w+ T·n̂" => 3.6e-4,                   # 2.02e-4
    "F2w− T·ê_φ" => 4.5e-4,                 # 2.41e-4
    "F2w− T·ê_θ" => 3.6e-4,                 # 1.93e-4
    "F2w− T·n̂" => 4.5e-4,                   # 2.48e-4
    # G7, both βK (max σ)
    "T·ê_φ/K" => 7.5e-3,                    # 4.31e-3
    "T·ê_θ/K" => 8e-3,                      # 4.56e-3
    "T·n̂/K" => 5.5e-3,                      # 3.12e-3
    # G5h (max σ)
    "⟨e₁·e₃⟩" => 5e-3)                      # 2.94e-3

# G5 — theorem (I) closed forms, constant energy, n̂ = (0.6, 0.35).
function _sw_g5_rows(seed)
    n̂ = _SW_N̂
    cases = (("N=2", 2, [1.0, 1.0], 4 / 3, 2.0, 0.0),
             ("N=3", 3, [1.0, 1.0, 1.0], 13 / 8, 3.0, 1 / 8),
             ("N=2 w=(1,2)", 2, [1.0, 2.0], 13 / 6, 5.0, nothing),
             ("N=2 w=(1,−2)", 2, [1.0, -2.0], 13 / 6, 5.0, nothing))
    rows = []
    for (name, n, w, R1, R2, Pneg) in cases
        H = MC.TiledHamiltonian(n, _sw_const_terms(n))
        # step fixed at 1.0 rad (no adaptation); β is irrelevant (ΔE ≡ 0)
        st, rc, sc, scratch = _sw_chain(H, n̂, w; step = 1.0, seed = seed)
        f(st) = begin
            M = _sw_M(st.config, w, trues(n))
            z = dot(M, n̂)
            SVector(z, z^2, count(s -> w[s] * dot(st.config[s], n̂) < 0, 1:n) / n)
        end
        m, σ = _sw_sample(f, st, H, 1.0, sc, rc, scratch; ntherm = 1_000,
                          nmeas = 400_000, ncomp = 3)
        key = w[2] != 1 ? "w" : name               # (1, ±2) share one cap
        push!(rows, (; name = "$name ⟨M∥⟩", value = m[1], σ = σ[1], exact = R1,
                     εmax = _SW_EPS["$key ⟨M∥⟩"]))
        push!(rows, (; name = "$name ⟨M∥²⟩", value = m[2], σ = σ[2], exact = R2,
                     εmax = _SW_EPS["$key ⟨M∥²⟩"]))
        # N = 2, equal weights: e·n̂ = c > 0 on the whole manifold, so the indicator
        # is identically 0 (σ = 0); the floor keeps `< 5σ` meaningful
        Pneg === nothing ||
            push!(rows, (; name = "$name P(we·n̂<0)", value = m[3],
                         σ = max(σ[3], 1e-12), exact = Pneg,
                         εmax = _SW_EPS["$name P(we·n̂<0)"]))
    end
    return rows
end

# G6 — N = 2 dimers against the quadrature of Z₂(n̂). Returns (rows, max quadrature
# self-convergence ΔZ/Z, max closed-form reconstruction error, βJ of each case).
function _sw_g6_rows(seed)
    β = 20.0
    f2u = MC.TiledHamiltonian(2, _f2u_terms())
    f2g = MC.TiledHamiltonian(2, _f2g_terms())
    rows = []
    conv = recon = 0.0
    βJ = Float64[]
    for (name, H, w, θ, φ) in (("F2u θ=0.3", f2u, [1.0, 1.0], 0.3, 0.35),
                               ("F2u θ=0.6", f2u, [1.0, 1.0], 0.6, 0.35),
                               ("F2u θ=1.2", f2u, [1.0, 1.0], 1.2, 0.35),
                               ("F2g", f2g, [1.0, 1.0], 0.6, 0.35),
                               ("F2w+", f2g, _f2w_weights(+1), 0.6, 0.35),
                               ("F2w−", f2g, _f2w_weights(-1), 0.6, 0.35))
        Ef, J, _, _, worst_E = _sw_dimer_model(H)
        recon = max(recon, worst_E)
        push!(βJ, β * J)
        ref, cv = _sw_torque_ref(Ef, β, w[1], w[2], θ, φ)
        conv = max(conv, cv)
        u = _sw_dir(θ, φ)
        eφ, eθ = _sw_eφ(θ, φ), _sw_eθ(θ, φ)
        st, rc, sc, scratch = _sw_chain(H, u, w; step = 0.8, seed = seed)
        f(st) = begin
            T = MC.total_torque(H, st.config)
            SVector(dot(T, eφ), dot(T, eθ), dot(T, u))
        end
        m, σ = _sw_sample(f, st, H, β, sc, rc, scratch; ntherm = 2_000,
                          nmeas = 200_000, ncomp = 3)
        for (c, lab) in enumerate(("T·ê_φ", "T·ê_θ", "T·n̂"))
            push!(rows, (; name = "$name $lab", value = m[c], σ = max(σ[c], 1e-15),
                         exact = ref[c], εmax = _SW_EPS["$name $lab"]))
        end
    end
    return rows, conv, recon, βJ
end

# The G7 Hamiltonian: a constant body-1 l = 0 term plus a uniaxial body-1 l = 2
# (Z₂₀ only) term on each of 3 atoms, dims (1,1,1). Returns (H, K) with K read off
# energies: per site E_K = −K (e·ẑ)² + const ⇒ K = E(x̂) − E(ẑ).
function _sw_g7_model()
    uniax = zeros(5)
    uniax[3] = 1.0
    terms = vcat(_sw_const_terms(3),
                 [MultipoleTerm(-0.01, 1, [a], [_CMC_Z], [2], copy(uniax)) for a = 1:3])
    H = MC.TiledHamiltonian(3, terms)
    x̂, ẑ = SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0)
    K = total_energy(H, MC.SpinConfig([x̂, ẑ, ẑ])) -
        total_energy(H, MC.SpinConfig([ẑ, ẑ, ẑ]))
    return H, K
end

# G7 — N = 3 linear response, one row triple per (βK, seed).
function _sw_g7_rows(cases, p2)
    θ, φ = 0.6, 0.35
    u = _sw_dir(θ, φ)
    eφ, eθ = _sw_eφ(θ, φ), _sw_eθ(θ, φ)
    H, K = _sw_g7_model()
    rows = []
    for (βK, seed) in cases
        β = βK / K
        st, rc, sc, scratch = _sw_chain(H, u, ones(3); step = 1.0, seed = seed)
        f(st) = begin
            T = MC.total_torque(H, st.config) / K
            SVector(dot(T, eφ), dot(T, eθ), dot(T, u))
        end
        m, σ = _sw_sample(f, st, H, β, sc, rc, scratch; ntherm = 1_000,
                          nmeas = 200_000, ncomp = 3)
        exact = (-3 * p2 * sin(2θ), 0.0, 0.0)
        for (c, lab) in enumerate(("T·ê_φ/K", "T·ê_θ/K", "T·n̂/K"))
            push!(rows, (; name = "βK=$βK $lab", value = m[c], σ = max(σ[c], 1e-15),
                         exact = exact[c], εmax = _SW_EPS[lab]))
        end
    end
    return rows
end

# G5h — the hybrid w = 0 site. Atoms 1, 2 constrained (w = 1), atom 3 hybrid
# (w = 0) and coupled ONLY to atom 1, by an isotropic l = 1 pair E = J e₁·e₃; atom 1
# also carries a random l = 2 anisotropy, atom 2 a constant l = 0 term (active).
# Integrating e₃ out, ∫dΩ₃ exp(−βJ e₁·e₃) = 4π sinh(βJ)/(βJ) does not depend on e₁,
# so the constrained marginal of (e₁, e₂) is untouched and, given e₁, e₃ has the
# Langevin law about e₁: ⟨e₁·e₃⟩ = L(−βJ) EXACTLY, whatever that marginal is.
# J is read off energies: (E(ẑ, ·, ẑ) − E(ẑ, ·, −ẑ))/2. Returns (rows, βJ).
function _sw_g5h_rows(seed)
    z = _CMC_Z
    terms = [MultipoleTerm(-0.02, 2, [1, 3], [z, z], [1, 1], _iso_folded(1)),
             MultipoleTerm(0.01, 1, [1], [z], [2], randn(MersenneTwister(9), 5)),
             MultipoleTerm(0.3, 1, [2], [z], [0], [1.0])]
    H = MC.TiledHamiltonian(3, terms)
    ẑ = SVector(0.0, 0.0, 1.0)
    J = (total_energy(H, MC.SpinConfig([ẑ, ẑ, ẑ])) -
         total_energy(H, MC.SpinConfig([ẑ, ẑ, -ẑ]))) / 2
    β = 20.0
    st, rc, sc, scratch = _sw_chain(H, _SW_N̂, [1.0, 1.0, 0.0]; step = 1.0,
                                    seed = seed)
    m, σ = _sw_sample(st -> SVector(dot(st.config[1], st.config[3])), st, H, β, sc,
                      rc, scratch; ntherm = 1_000, nmeas = 200_000, ncomp = 1)
    rows = [(; name = "⟨e₁·e₃⟩", value = m[1], σ = σ[1], exact = _langevin(-β * J),
             εmax = _SW_EPS["⟨e₁·e₃⟩"])]
    return rows, β * J
end

@testset "CMC sweep (M4)" begin
    n̂ = _SW_N̂

    @testset "G1: invariants over ≥ 10⁴ moves" begin
        fmix = MC.TiledHamiltonian(8, _fmix_terms())
        fmix2 = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        f2g = MC.TiledHamiltonian(2, _f2g_terms())
        rng = MersenneTwister(5101)
        rows = []
        # Fmix on dims (2,1,1): 𝒞 = {1..6, 9..14} is not a prefix of the sites, so
        # a wrong source position (e.g. `pos = s`) picks j = i or a wrong partner
        for (name, H, w, act, nsw) in (
                ("Fmix", fmix, _fmix_weights(), _sw_fmix_act(fmix), 2_000),
                ("Fmix(2,1,1)", fmix2, _sw_site_w(fmix2, _fmix_weights()),
                 _sw_fmix_act(fmix2), 1_000),
                ("F2w+", f2g, _f2w_weights(+1), trues(2), 5_000),
                ("F2w−", f2g, _f2w_weights(-1), trues(2), 5_000))
            wn = sum(abs(w[s]) for s in eachindex(w) if act[s])
            init = [_rand_spin(rng) for _ = 1:H.n_sites]
            st, rc, sc, scratch = _sw_chain(H, n̂, H === fmix2 ? _fmix_weights() : w;
                                            init = init, step = 0.7, seed = 5102)
            worst_perp = worst_Minc = worst_E = worst_unit = 0.0
            min_par = Inf
            rows_ok = true
            nacc = 0
            for _ = 1:nsw
                a, Minc = MC._cmc_sweep!(st, H, 20.0, sc, rc, scratch)
                nacc += a
                M = _sw_M(st.config, w, act)
                worst_perp = max(worst_perp, _sw_perp(M, n̂))
                worst_Minc = max(worst_Minc, norm(Minc - M))
                min_par = min(min_par, dot(M, n̂))
                worst_E = max(worst_E, abs(st.energy - total_energy(H, st.config)))
                worst_unit = max(worst_unit, maximum(e -> abs(norm(e) - 1), st.config))
                rows_ok &= st.zrows == MC._zrows(H, st.config)
            end
            nmoves = nsw * count(s -> act[s] && w[s] != 0, eachindex(w))
            MC._renormalize!(st, H)
            Mr = _sw_M(st.config, w, act)
            push!(rows, (; name, nmoves, nacc, worst_perp, worst_Minc, worst_E,
                         worst_unit, min_par, rows_ok, wn,
                         perp_renorm = _sw_perp(Mr, n̂),
                         rows_renorm = st.zrows == MC._zrows(H, st.config)))
        end
        # Measured (seeds 5101/5102, β = 20, step 0.7): 12 000 / 12 000 / 10 000 /
        # 10 000 pair attempts, 56–59 % accepted; max |M⊥|/Σ|w| ≤ 2.0e-14 (F2w−) —
        # a rounding random walk, ≥ 50× below the bound; max |M_inc − M| ≤ 3.8e-15;
        # max |E_inc − E| ≤ 2.2e-15 eV; max ||e| − 1| ≤ 2.9e-15. Mutants caught here:
        # stale zrows column j and the naive ΔE (`rows_ok`, `worst_E`); M not
        # updated after an accept (throws: the stale M∥ turns negative); a wrong
        # source position `pos = s` (Fmix(2,1,1): j = i, AssertionError).
        _SW_VERBOSE && foreach(r -> println("G1   ", r), rows)
        for r in rows
            @testset "$(r.name)" begin
                @test r.nmoves >= 10_000 && r.nacc >= 2_000      # non-vacuity
                @test r.worst_perp <= 1e-12 * r.wn
                @test r.worst_Minc <= 1e-12 * r.wn
                @test r.min_par > 0
                @test r.worst_E <= 1e-12
                @test r.worst_unit <= 1e-12
                @test r.rows_ok
                @test r.perp_renorm <= 1e-12 * r.wn
                @test r.rows_renorm
            end
        end
    end

    @testset "G1b: inactive site frozen and outside M; zero-weight site hybrid" begin
        H = MC.TiledHamiltonian(8, _fmix_terms())
        w = _fmix_weights()
        act = [s != _FMIX_INACTIVE for s = 1:8]
        @test !H.site_active[_FMIX_INACTIVE] && w[_FMIX_INACTIVE] != 0
        @test H.site_active[_FMIX_ZERO_W] && w[_FMIX_ZERO_W] == 0
        rng = MersenneTwister(5201)
        st, rc, sc, scratch = _sw_chain(H, n̂, w;
                                        init = [_rand_spin(rng) for _ = 1:8],
                                        step = 0.7, seed = 5202)
        e8 = st.config[_FMIX_INACTIVE]
        z8 = st.zrows[:, _FMIX_INACTIVE]
        # non-vacuity: with site 8 included, M would NOT be ∥ n̂
        @test _sw_perp(_sw_M(st.config, w, trues(8)), n̂) > 0.1
        n7moves = 0
        worst_Minc = 0.0
        frozen = true
        for _ = 1:2_000
            e7 = st.config[_FMIX_ZERO_W]
            _, Minc = MC._cmc_sweep!(st, H, 20.0, sc, rc, scratch)
            n7moves += st.config[_FMIX_ZERO_W] != e7
            worst_Minc = max(worst_Minc, norm(Minc - _sw_M(st.config, w, act)))
            frozen &= st.config[_FMIX_INACTIVE] === e8 &&
                      st.zrows[:, _FMIX_INACTIVE] == z8
        end
        _SW_VERBOSE && println("G1b  site-7 moves $n7moves, worst |M_inc − M| $worst_Minc")
        @test frozen
        # the sweep's M is the sum over 𝒞 only (site 8 would shift it by 0.7·e₈)
        @test worst_Minc <= 1e-12 * sum(abs, w[act])
        # Measured (seed 5202): site 7 moved in 1454 of 2000 sweeps; 0 when the
        # hybrid attempt is skipped (mutant)
        @test n7moves >= 500
        # the w = 0 site moves while the constrained part stays ∥ n̂
        @test _sw_perp(_sw_M(st.config, w, act), n̂) <= 1e-12 * sum(abs, w[act])
        # counters: one attempt per active site per sweep (null moves included)
        @test st.att_metro == 2_000 * H.n_active
    end

    @testset "G16: allocation-free sweeps" begin
        alloc(st, H, sc, rc, scratch) = @allocated MC._cmc_sweep!(st, H, 20.0, sc, rc,
                                                                  scratch)
        for (name, H, w) in (("F3a", MC.TiledHamiltonian(3, _f3a_terms()), ones(3)),
                             ("F4", MC.TiledHamiltonian(4, _f4_terms()), ones(4)),
                             ("Fmix", MC.TiledHamiltonian(8, _fmix_terms()),
                              _fmix_weights()))
            st, rc, sc, scratch = _sw_chain(H, n̂, w; step = 0.7, seed = 5301)
            for _ = 1:10
                MC._cmc_sweep!(st, H, 20.0, sc, rc, scratch)
            end
            alloc(st, H, sc, rc, scratch)
            @test alloc(st, H, sc, rc, scratch) == 0
        end
    end

    @testset "D3: RNG contract of the pair attempt" begin
        # A determinism / RNG-contract check, not a correctness oracle. Every
        # attempt is replayed on a COPY of the source's stream with the contract
        # written out here: k = rand(1:|𝒞|−1) shifted past the source's
        # position, axis = normalized 3 randn, angle = step·randn, and the accept
        # uniform only when the move is not null and ℓ < 0. ℓ itself comes from
        # the pure proposal (its value is G4's business, not this gate's). After the
        # attempt the real stream must equal the replayed copy, and the outcome must
        # match.
        H = MC.TiledHamiltonian(8, _fmix_terms())
        w = _fmix_weights()
        rng = MersenneTwister(5401)
        st, rc, sc, scratch = _sw_chain(H, n̂, w;
                                        init = [_rand_spin(rng) for _ = 1:8],
                                        step = 1.5, seed = 5402)
        act = [s != _FMIX_INACTIVE for s = 1:8]
        cset = [s for s = 1:8 if act[s] && w[s] != 0]
        @test rc.cset == cset
        β = 40.0
        aux = MC._CMCScratch(H)
        tally = Dict(:null => 0, :pos => 0, :neg_acc => 0, :neg_rej => 0)
        bad = Any[]
        for t = 1:3_000
            pos = mod1(t, length(cset))
            i = cset[pos]
            M = _sw_M(st.config, w, act)
            x0 = copy(st.config)
            r = copy(st.site_rngs[i])
            k = rand(r, 1:(length(cset) - 1))
            j = cset[k >= pos ? k + 1 : k]
            v = SVector(randn(r), randn(r), randn(r))
            e_i_new = _sw_rot(x0[i], v / norm(v), st.step * randn(r))
            e_j_new, ΔE, ℓ, _ = MC._cmc_pair_proposal(x0, st.zrows, H, rc, M, i, j,
                                                      e_i_new, β, aux)
            null = isnan(ΔE)
            acc = !null && (ℓ >= 0 || log(rand(r)) < ℓ)
            a, _ = MC._attempt_cmc_pair!(st.config, st.zrows, H, rc, M, β, st.step,
                                         i, pos, scratch, st.site_rngs[i], sc.dE)
            ok = r == st.site_rngs[i] && a == acc
            if acc
                ok &= norm(st.config[i] - e_i_new) < 1e-14 &&
                      norm(st.config[j] - e_j_new) < 1e-12 &&
                      all(s -> s in (i, j) || st.config[s] === x0[s], 1:8)
            else
                ok &= st.config == x0
            end
            ok || push!(bad, (; t, i, j, null, ℓ, acc, a))
            tally[null ? :null : ℓ >= 0 ? :pos : acc ? :neg_acc : :neg_rej] += 1
        end
        _SW_VERBOSE && println("D3   ", tally, " offenders ", length(bad))
        isempty(bad) || @error "RNG contract: first offender" first(bad) length(bad)
        @test isempty(bad)
        # non-vacuity: every branch of the accept rule was exercised. Measured (seed
        # 5402): 1247 null, 352 with ℓ ≥ 0, 307 accepted / 1094 rejected with ℓ < 0.
        # Mutants caught only here (they leave the measure intact): an accept
        # uniform drawn also when ℓ ≥ 0 (352 offenders), a partner rejection loop
        # (843), a partner drawn from all of 𝒞 with j = i as a no-op (948); the
        # null re-draw (owned by G5) is seen here too (1227).
        @test all(>=(100), values(tally))
    end

    @testset "D3: sweep-level RNG replay (Fmix on dims (2,1,1))" begin
        # Whole sweeps replayed on copies (`_sw_replay_sweep`): site order, the
        # source's position in 𝒞 counted independently, the w = 0 sites' own
        # streams (sites 7, 15), inactive sites (8, 16) drawing nothing, and the
        # chain stream `st.rng` never consumed. Re-synchronized at every sweep (the
        # replay uses `total_energy` for the hybrid ΔE, so configurations agree to
        # rounding, not bitwise).
        H = MC.TiledHamiltonian(8, _fmix_terms(); dims = (2, 1, 1))
        w_site = _sw_site_w(H, _fmix_weights())
        act = _sw_fmix_act(H)
        rng = MersenneTwister(5403)
        st, rc, sc, scratch = _sw_chain(H, n̂, _fmix_weights();
                                        init = [_rand_spin(rng) for _ = 1:16],
                                        step = 1.0, seed = 5404)
        @test rc.cset == [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
        chain_rng = copy(st.rng)
        bad = Any[]
        worst_cfg = 0.0
        nacc_total = 0
        for t = 1:300
            cfg, rs, nacc = _sw_replay_sweep(st, H, 20.0, rc, w_site, act)
            a, _ = MC._cmc_sweep!(st, H, 20.0, sc, rc, scratch)
            nacc_total += a
            d = maximum(s -> norm(cfg[s] - st.config[s]), 1:16)
            worst_cfg = max(worst_cfg, d)
            ok = a == nacc && d < 1e-12 && all(s -> rs[s] == st.site_rngs[s], 1:16)
            ok || push!(bad, (; t, a, nacc, d))
        end
        _SW_VERBOSE && println("D3s  accepted $nacc_total, worst |Δe| $worst_cfg, " *
                               "offenders $(length(bad))")
        isempty(bad) || @error "sweep replay: first offender" first(bad) length(bad)
        @test isempty(bad)
        @test st.rng == chain_rng                       # the chain stream is unused
        # non-vacuity. Measured (seed 5404): 1821 accepted moves over 300 sweeps
        # (14 attempts each); the replayed configurations agree bitwise. Mutants
        # caught: `pos = s` (j = i), a hybrid site drawing from `st.rng` (300
        # offending sweeps), the hybrid site at 0.5β (48), the null re-draw (299).
        @test nacc_total >= 1_000
    end

    @testset "G0: warnings fire exactly when their conditions hold" begin
        H = MC.TiledHamiltonian(2, _sw_const_terms(2))
        t1, t2 = _sw_frame(n̂)
        rc = MC._resolve_constraint(H, MC.MagnetizationConstraint(n̂; weights = :uniform))
        # a two-spin state with M∥ = 2c and an injected transverse part δ t₂
        state(c, δ) = MC.SpinConfig([c * n̂ + sqrt(1 - c^2) * t1,
                                     normalize(c * n̂ - sqrt(1 - c^2) * t1 + δ * t2)])
        # Σ|w| = 2, so the absolute bound is 2e-10
        @test (@test_logs (:warn, r"transverse magnetization") MC._cmc_check_drift(
            rc, _sw_M(state(0.6, 4e-10), ones(2), trues(2))))
        @test !(@test_logs min_level = Warn MC._cmc_check_drift(
            rc, _sw_M(state(0.6, 1e-10), ones(2), trues(2))))
        # absolute, not relative: M∥ = 2e-3 makes |M⊥|/M∥ = 5e-8 ≫ 1e-10, no warning
        @test !(@test_logs min_level = Warn MC._cmc_check_drift(
            rc, _sw_M(state(1e-3, 1e-10), ones(2), trues(2))))
        # through the sweep: the check runs at every sweep start
        sc, scratch = MC.SweepScratch(H), MC._CMCScratch(H)
        st = MC.ChainState(H, state(0.6, 4e-10), Xoshiro(1), 0.5)
        @test_logs (:warn, r"transverse magnetization") match_mode = :any MC._cmc_sweep!(
            st, H, 1.0, sc, rc, scratch)
        st = MC.ChainState(H, state(0.6, 1e-10), Xoshiro(1), 0.5)
        @test_logs min_level = Warn MC._cmc_sweep!(st, H, 1.0, sc, rc, scratch)

        # step floor: driven onto the clamp by the real adaptation (zero acceptance)
        @test (@test_logs min_level = Warn MC._cmc_check_step_floor(st, 0.1, n̂)) == false
        for _ = 1:200
            st.acc_metro, st.att_metro = 0, 100
            MC._adapt_step!(st, 0.5)
        end
        @test st.step == MC._STEP_MIN
        @test (@test_logs (:warn, r"lower clamp") MC._cmc_check_step_floor(st, 0.1, n̂))
        st.step = 1.001e-3
        @test !(@test_logs min_level = Warn MC._cmc_check_step_floor(st, 0.1, n̂))

        # weak constraint: M∥/Σ|w| = c against 0.05
        @test (@test_logs (:warn, r"weak constraint") MC._cmc_check_weak_constraint(
            rc, state(0.049, 0.0), 0.1))
        @test !(@test_logs min_level = Warn MC._cmc_check_weak_constraint(
            rc, state(0.051, 0.0), 0.1))
        # weighted (1, −2): the norm is Σ|w| = 3 — not |𝒞| = 2, not Σw = −1. With
        # e₁ = n̂ and e₂·n̂ = z, M∥ = 1 − 2z: M∥ = 0.12 → 0.04 (warn; 0.06 under
        # |𝒞|), M∥ = 0.18 → 0.06 (no warning; negative, hence a warning, under Σw).
        rcw = MC._resolve_constraint(H, MC.MagnetizationConstraint(n̂;
                                                                   weights = [1.0, -2.0]))
        xw(Mpar) = (z = (1 - Mpar) / 2; MC.SpinConfig([n̂, z * n̂ + sqrt(1 - z^2) * t1]))
        @test dot(_sw_M(xw(0.12), [1.0, -2.0], trues(2)), n̂) ≈ 0.12 atol = 1e-15
        @test (@test_logs (:warn, r"weak constraint") MC._cmc_check_weak_constraint(
            rcw, xw(0.12), 0.1))
        @test !(@test_logs min_level = Warn MC._cmc_check_weak_constraint(
            rcw, xw(0.18), 0.1))
    end

    @testset "G5: theorem (I) closed forms (constant energy)" begin
        # Closed forms (iid uniform spins, M = Σ w e; by theorem (I) — decision record
        # D1 — the constrained ensemble at any n̂ has the same distribution of
        # rotation-invariant quantities, and M·n̂ = |M| there):
        # - N = 2: cos γ = e₁·e₂ is uniform on [−1, 1] and |M|² = 2 + 2cos γ, so
        #   ⟨|M|²⟩ = 2 and ⟨|M|⟩ = ½∫√(2+2u) du = (1/6)·[(2+2u)^{3/2}]₋₁¹ = 8/6 = 4/3.
        # - N = 3: R = |M| has density R²/2 on [0, 1], R(3 − R)/4 on [1, 3], so
        #   ⟨R⟩ = 1/8 + ¼[R³ − R⁴/4]₁³ = 1/8 + 3/2 = 13/8; ⟨R²⟩ = Σ|e|² = 3; and
        #   P(e₁·M̂ < 0) = P(1 + ru < 0) = ∫₁² (r/2)·(1 − 1/r)/2 dr = 1/8 (r = |e₂+e₃|
        #   with density r/2, u uniform).
        # - N = 2 with w = (1, ±2): R² = 5 ± 4u, u uniform: ⟨R²⟩ = 5 and
        #   ⟨R⟩ = ½∫√(5+4u) du = (1/12)(27 − 1) = 13/6 (the gauge map e₂ → −e₂ makes
        #   (1, −2) identical).
        # Measured (seed 5501, step FIXED at 1.0 rad, no adaptation, 4·10⁵ sweeps):
        # σ = 1.6e-3 / 2.9e-3 / 2.3e-3 for ⟨M∥⟩ (N=2 / N=3 / weighted), 3.6e-3 …
        # 9.4e-3 for ⟨M∥²⟩, 7.6e-4 for P; worst |dev| = 1.6σ; atol = 5σ. The (1, −2)
        # chain reproduces the (1, 2) numbers bitwise: every step commutes exactly
        # with the gauge map e₂ → −e₂.
        # Smallest shift of each mutant (multiple of atol): null move re-drawn
        # (this gate owns it: 15.8–21.8σ across seeds) +0.022 on N=2 ⟨M∥⟩ (16σ,
        # 3.2×) and +0.17 on the weighted ⟨M∥²⟩ (19σ, 3.8×); Jacobian exponent
        # 2 → 1 −0.33 (220σ); |z| factor missing +0.17 (126σ); exponent 1 AND |z|
        # missing: invisible at N=2 equal weights, as designed, N=3 P +0.061 (70σ);
        # σ_j forced +1: N=3 P −0.076 (240σ).
        _sw_check_rows("G5", _sw_g5_rows(5501))
    end

    @testset "G6: N = 2 dimer torque vs quadrature" begin
        # β = 20 with J = −0.06 (βJ = −1.2) and single-ion eigenvalue spreads
        # 0.034 (F2u, β·spread = 0.67) and 0.081 (F2g, 1.6), all read off energies.
        # Measured (seed 5601, step FIXED at 0.8 rad, 2·10⁵ sweeps per case): σ =
        # 2.6e-5 … 1.2e-4 (F2u), 1.7e-4 … 3.0e-4 (F2g, F2w±); worst |dev| = 1.9σ;
        # atol = 5σ; quadrature ΔZ/Z under grid doubling ≤ 1.1e-14.
        # Smallest shift of each mutant (multiple of atol): naive ΔE (c_j from the
        # old rows) −6.7e-3 on F2u θ=0.6 T·ê_φ (62σ, 12×); Jacobian exponent 1
        # −8.3e-3 (97σ); |z| missing +4.5e-3 (77σ); stale zrows column j +8.9e-4
        # (12σ, 2.4×); σ_j forced +1 +4.6e-3 on F2w− (9.3σ, 1.9×). The null re-draw
        # is NOT owned here: 7.6σ with this seed, 4.3σ with another (G5 owns it).
        rows, conv, recon, βJ = _sw_g6_rows(5601)
        @test recon < 1e-14                      # closed-form energy reconstruction
        @test all(x -> -2 <= x <= -1, βJ)         # ferro, |βJ| ~ 1–2
        _SW_VERBOSE && println("G6   quadrature self-convergence max ΔZ/Z = $conv")
        @test conv < 1e-10                       # quadrature self-convergence
        _sw_check_rows("G6", rows)
    end

    @testset "G7: N = 3 linear response K_eff = K⟨P₂(e·M̂)⟩_iso" begin
        p2a, p2b = _sw_p2_iso3(48), _sw_p2_iso3(96)
        @test abs(p2a - p2b) < 1e-6                     # self-convergence
        # independent value: the inner R integral in closed form,
        # ∫ R P₂ dR = (3/8)(R⁴/4 + aR² + a² ln R) − R²/4 with a = 1 − r², then
        # 1-D Gauss–Legendre in r (split at 1): 0.2207816474254870
        @test p2b ≈ 0.2207816474254870 atol = 1e-9
        H, K = _sw_g7_model()
        @test K > 0
        # reconstruction: the G7 energy is exactly −K Σ (e·ẑ)² + C on the sphere
        x̂ = SVector(1.0, 0.0, 0.0)
        C = total_energy(H, MC.SpinConfig([x̂, x̂, x̂]))
        rng = MersenneTwister(5703)
        recon = maximum(1:200) do _
            x = _rand_config(rng, H)
            abs(total_energy(H, x) - (C - K * sum(e -> e[3]^2, x)))
        end
        @test recon < 1e-14
        # Measured (seeds 5701/5702, step FIXED at 1.0 rad, 2·10⁵ sweeps): σ(T·ê_φ/K)
        # = 4.1e-3 / 4.2e-3; dev −0.39σ / +0.02σ; the two K values differ by 0.28
        # combined σ — the O(βK) correction is below resolution; the ⟨P₂⟩
        # quadrature changes by 1.7e-10 from n = 48 to 96 (bound 1e-6).
        # Smallest shift of each mutant (multiple of atol): Jacobian
        # exponent 1 +0.145 (42σ, 8×); |z| missing −0.29 (96σ); σ_j forced +1
        # −0.050 (15σ, 2.9×); null move re-drawn −0.035 (11σ, 2.2×).
        rows = _sw_g7_rows(((1e-3, 5701), (1e-2, 5702)), p2b)
        _SW_VERBOSE && println("G7   ⟨P₂⟩ quadrature $p2a (n = 48), $p2b (n = 96)")
        _sw_check_rows("G7", rows)
        # the O(βK) correction: the two K values agree within 5 combined σ
        a, b = rows[1], rows[4]
        @test abs(a.value - b.value) < 5 * hypot(a.σ, b.σ)
    end

    @testset "G5h: hybrid w = 0 site, Langevin law" begin
        # β = 20, βJ = −1.2 (read off energies), exact ⟨e₁·e₃⟩ = L(1.2) = 0.36620.
        # Measured (seed 5801, step FIXED at 1.0 rad, 2·10⁵ sweeps): σ = 2.4e-3,
        # dev +1.2σ; σ over 8 seeds 2.0e-3 … 2.9e-3. The only statistical gate that
        # sees the hybrid kernel: the w = 0 site attempted at 0.5β shifts −0.146
        # (−92σ, 18× atol); its stream drawn from `st.rng` is seen by D3 only.
        rows, βJ = _sw_g5h_rows(5801)
        @test -2 <= βJ <= -1
        _sw_check_rows("G5h", rows)
    end
end

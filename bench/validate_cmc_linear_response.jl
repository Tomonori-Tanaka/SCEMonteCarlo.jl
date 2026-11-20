# G9 — large-N linear response of constrained MC: the exact finite-temperature
# Callen–Callen identity (docs/specs/constrained-mc.md, "Manual validation"; gate
# table of the spec, G9).
#
#   julia --project=bench -t 4 bench/validate_cmc_linear_response.jl [sweeps]
#
# FIXTURE FL: simple-cubic 4×4×4 (N = 64, one atom per cell) isotropic ferromagnet,
# E_iso = J Σ_⟨ij⟩ e_i·e_j with J = −1, plus a tiny single-ion term
# E_K = −K Σ_i (e_i·ẑ)², K = 1e-3 |J| (βK ≈ 1e-3). J and K are READ OFF energy
# differences (`total_energy`), and E − E_iso = −K Σ z² + const is checked on random
# configurations, so no coupling constant is taken on faith.
#
# ORACLE (research/C §1.5, research/A §6 O7). The isotropic constrained measure is
# axially symmetric about n̂, so by the addition theorem
#     F(θ) = const − N K ⟨P₂(e·M̂)⟩_iso cos²θ + O(βK² N),
# i.e. with F ≡ const − N K_eff cos²θ: K_eff/K = ⟨P₂(e·M̂)⟩_iso [1 + O(βK)] and, per
# site, ⟨ê_φ·T⟩ = −K_eff sin 2θ. ⟨P₂(e·M̂)⟩ is a rotation invariant, so (theorem (I))
# it is measured by the EXISTING unconstrained `run_mc` on the isotropic model
# (same uniform weights, same active set = all sites) — the CMC code is never
# touched by the oracle. 4 independent `run_mc` replicas (threads), pooled.
#
# CMC side: `run_cmc` at θ = π/4 (sin 2θ = 1), φ = 0.35, 4 duplicate lanes per
# temperature; K_eff = −⟨ê_φ·T⟩ per cell. Temperatures kT/|J| = 0.7, 1.2, 1.4
# (T_c(sc) ≈ 1.443 |J|: about 0.49, 0.83, 0.97 T_c).
#
# TOLERANCE RULE (fixed before looking at the result), per temperature:
#   - |K_eff/K − ⟨P₂⟩_iso| < 4 σ_comb, σ_comb = √(σ_cmc² + σ_iso²);
#   - resolution: σ_comb ≤ 1 % of ⟨P₂⟩_iso (the target "1 % agreement"); a point that
#     misses it is "underpowered", not a pass;
#   - ⟨ê_θ·T⟩ and ⟨n̂·T⟩ (exactly 0 for this uniaxial model) within 4σ;
#   - the O(βK) remainder (≤ ~0.1 % × fluctuation factor) is below this resolution.

include(joinpath(@__DIR__, "validate_cmc_common.jl"))

sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500_000
const DIMS = (4, 4, 4)
const KTS = [1.4, 1.2, 0.7]                 # × |J|, high → low (the run_mc ladder)
const K_TARGET = 1e-3
const θC, φC = π / 4, 0.35

banner("G9 — FL sc 4×4×4: K_eff(T)/K vs ⟨P₂(e·M̂)⟩_iso, sweeps = $sweeps")

H_iso = TiledHamiltonian(1, sc_pair_terms(-1.0); dims = DIMS)
J = sc_read_J(H_iso)
N = n_sites(H_iso)

# K per unit raw coefficient of the uniaxial term, then the full model
Hu = TiledHamiltonian(1, [uniaxial_term(1.0)]; dims = DIMS)
x̂, ẑ = SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0)
k_per_c = (uniform_energy(Hu, x̂) - uniform_energy(Hu, ẑ)) / N
H = TiledHamiltonian(1, [sc_pair_terms(-1.0); uniaxial_term(K_TARGET / k_per_c)];
                     dims = DIMS)
K = (uniform_energy(H, x̂) - uniform_energy(H, ẑ)) / N
# closed-form check: E − E_iso + K Σ z² must be one constant
rng = Xoshiro(901)
offs = map(1:50) do _
    cfg = SpinConfig([rand_unit(rng) for _ = 1:N])
    total_energy(H, cfg) - total_energy(H_iso, cfg) + K * sum(e -> e[3]^2, cfg)
end
@printf("N = %d, J (read) = %.12f, K (read) = %.6e, closed form E_K = −KΣz² + C: spread %.2e\n",
        N, J, K, maximum(offs) - minimum(offs))

# --- oracle: ⟨P₂(e·M̂)⟩ in the unconstrained isotropic run_mc ----------------------------
P2obs = Observable(:P2, 1, (cfg, E, Hr) -> begin
    M = sum(cfg)
    m̂ = M / norm(M)
    s = 0.0
    for e in cfg
        c = dot(e, m̂)
        s += 1.5 * c^2 - 0.5
    end
    s / length(cfg)
end)
t_iso = @elapsed iso = let
    tasks = [Threads.@spawn run_mc(H_iso; kT = KTS .* abs(J), sweeps_therm = 10_000,
                                   sweeps_measure = sweeps, measure_interval = 1,
                                   or_per_metropolis = 1,
                                   observables = [standard_observables(H_iso); P2obs],
                                   seed = 9000 + r) for r = 1:4]
    fetch.(tasks)
end
@printf("isotropic run_mc: 4 replicas × %d temperatures × %d sweeps (OR 1), %.1f s\n",
        length(KTS), sweeps, t_iso)

# --- CMC ------------------------------------------------------------------------------
function frame_obs(H, c)
    n̂ = c.direction
    u = normalize(cross(SVector(0.0, 0.0, 1.0), n̂))   # ê_φ
    v = cross(u, n̂)                                   # ê_θ
    ncell = n_sites(H) ÷ H.n_cell_atoms
    return [cmc_observables(H, c);
            Observable(:Tframe, 3, (cfg, E, Hr) -> begin
                T = MC.total_torque(H, cfg) / ncell
                SVector(dot(u, T), dot(v, T), dot(n̂, T))
            end)]
end

con = MagnetizationConstraint(θC, φC; weights = :uniform)
allok, allres = true, true
println("\n  kT/|J|  βK        ⟨P₂⟩_iso (χ²)          K_eff/K (χ²)            dev      rel.dev   σ_comb/P₂  ⟨ê_θ·T⟩/σ, ⟨n̂·T⟩/σ     M∥/N    acc    t_cmc")
for (k, kT) in enumerate(KTS .* abs(J))
    p2 = pool([r.points[k].stats[:P2].mean[1] for r in iso],
              [r.points[k].stats[:P2].err[1] for r in iso])
    t = @elapsed rs = run_cmc(H, fill(con, 4); kT = kT, sweeps_therm = 10_000,
                              sweeps_measure = sweeps, measure_interval = 1,
                              observables = c -> frame_obs(H, c), seed = 9100 + k)
    uT = pool_stat(rs, :Tframe, 1)
    vT = pool_stat(rs, :Tframe, 2)
    nT = pool_stat(rs, :Tframe, 3)
    Mp = pool_stat(rs, :M_par, 1)
    keff = -uT.mean / (K * sin(2θC))
    σk = uT.σ / (K * sin(2θC))
    σc = sqrt(σk^2 + p2.σ^2)
    dev = (keff - p2.mean) / σc
    # uniaxial model: ∂F/∂φ = 0 and ⟨n̂·T⟩ = 0 exactly
    ok = abs(dev) < 4 && abs(vT.mean) < 4 * vT.σ && abs(nT.mean) < 4 * nT.σ
    res = σc <= 0.01 * p2.mean
    global allok &= ok
    global allres &= res
    @printf("  %5.2f  %.2e  %.5f ± %.5f (%.2f)  %.5f ± %.5f (%.2f)  %+5.2fσ  %+6.2f%%  %5.2f%%     %+5.2f, %+5.2f        %.4f  %.3f  %5.1f s  %s%s\n",
            kT / abs(J), K / kT, p2.mean, p2.σ, p2.chi2, keff, σk, uT.chi2, dev,
            100 * (keff - p2.mean) / p2.mean, 100 * σc / p2.mean, vT.mean / vT.σ,
            nT.mean / nT.σ, Mp.mean, mean(r -> r.points[1].acceptance_metropolis, rs), t,
            ok ? "ok" : "FAIL", res ? "" : " UNDERPOWERED")
end
println("\nG9 verdict: ", allok ? "PASS" : "FAIL", " (all temperatures within 4σ_comb)",
        allres ? ", resolution ≤ 1 % met" : "; some point UNDERPOWERED")

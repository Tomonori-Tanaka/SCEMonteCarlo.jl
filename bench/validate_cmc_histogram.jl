# G11 — constrained-MC torque integral vs the unconstrained histogram of M̂
# (docs/specs/constrained-mc.md, "Manual validation", row G11).
#
#   julia --project=bench -t 4 bench/validate_cmc_histogram.jl [sweeps_hist] [sweeps_cmc]
#
# FIXTURE: simple-cubic 3×3×3 (N = 27, one atom per cell), isotropic ferromagnet
# E_iso = J Σ_⟨ij⟩ e_i·e_j (J = −1, read back from energy differences) plus a uniaxial
# single-ion term E_K = −K Σ (e·ẑ)² (K read back from energy differences), at
# kT = 1.15 |J| (≈ 0.8 T_c of the bulk sc model). An intermediate N where the
# rotated-iid reference of G8 has lost its effective sample size.
#
# ORACLE. With uniform weights, the
# unconstrained canonical density of M̂ per solid angle is ∝ e^{−βF(n̂)} — the SAME F
# whose gradient the constrained mean torque is. For this uniaxial model F = F(cosθ)
# and the density in c = cosθ is ρ(c) ∝ e^{−βF}, so
#     ΔF(θ_a → θ) = −kT ln[ρ(cos θ)/ρ(cos θ_a)]          (histogram, plain `run_mc`)
#                 = −∫_{θ_a}^{θ} ⟨ê_φ·T⟩ dθ′             (CMC, ⟨ê_φ·T⟩ = −∂F/∂θ).
# Histogram: scalar indicator observables of |cos θ_M| ∈ [c_k − h, c_k + h] (h = 0.02,
# folded over ±c — F(c) = F(−c) for this time-reversal-even model at B = 0) on the
# even Simpson nodes; ΔF_k is an `Evaluable`, so the jackknife over bins propagates
# the ratio, the log, and the correlation between the two bins. Bin-width bias is
# second order, ~(βF)″ h²/6 ≲ 0.01 kT. 4 independent `run_mc` replicas (threads),
# pooled. CMC: 17 θ nodes equally spaced on [θ_a, π/2], θ_a = acos(0.95), φ = 0.35,
# one `run_cmc` lane each; cumulative composite Simpson on the even nodes, σ from the
# lane binning errors propagated through the Simpson weights (lanes independent).
#
# K RULE (fixed before any comparison): a short isotropic pilot measures
# ⟨P₂(e·M̂)⟩_iso, and K = 3 kT / (N ⟨P₂⟩) so that the O(K) estimate of βΔF(0 → π/2)
# is 3 (the regime βΔF ≲ 4 where the histogram still has counts at the equator).
#
# TOLERANCE RULE (fixed before looking at the result):
#   - at every even node θ_{2k}: |ΔF_cmc − ΔF_hist| < 4 σ_comb (8 correlated rows);
#   - resolution: at the endpoint θ = π/2, σ_comb ≤ 0.1 kT, and βΔF(θ_a → π/2) is in
#     [1.5, 4.5] (otherwise the regime is wrong and the run is inconclusive);
#   - the CMC must also give ⟨ê_θ·T⟩ = 0 and ⟨n̂·T⟩ = 0 (uniaxial model) within 4σ at
#     every node.

include(joinpath(@__DIR__, "validate_cmc_common.jl"))

sweeps_hist = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4_000_000
sweeps_cmc = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400_000
const DIMS = (3, 3, 3)
const KT_J = 1.15
const NNODE = 17
const HBIN = 0.02
const φC = 0.35

banner("G11 — sc 3×3×3 + uniaxial K: ∫⟨ê_φ·T⟩dθ (run_cmc) vs M̂ histogram (run_mc), " *
       "sweeps_hist = $sweeps_hist × 4, sweeps_cmc = $sweeps_cmc × $NNODE lanes")

H_iso = TiledHamiltonian(1, sc_pair_terms(-1.0); dims = DIMS)
J = sc_read_J(H_iso)
N = n_sites(H_iso)
kT = KT_J * abs(J)

P2obs = Observable(:P2, 1, (cfg, E, Hr) -> begin
    m̂ = normalize(sum(cfg))
    sum(e -> 1.5 * dot(e, m̂)^2 - 0.5, cfg) / length(cfg)
end)
pilot = run_mc(H_iso; kT = kT, sweeps_therm = 5_000, sweeps_measure = 100_000,
               observables = [standard_observables(H_iso); P2obs], seed = 1101)
p2 = pilot.points[1].stats[:P2].mean[1]

x̂, ẑ = SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0)
Hu = TiledHamiltonian(1, [uniaxial_term(1.0)]; dims = DIMS)
k_per_c = (uniform_energy(Hu, x̂) - uniform_energy(Hu, ẑ)) / N
K_target = 3 * kT / (N * p2)
H = TiledHamiltonian(1, [sc_pair_terms(-1.0); uniaxial_term(K_target / k_per_c)];
                     dims = DIMS)
K = (uniform_energy(H, x̂) - uniform_energy(H, ẑ)) / N
@printf("N = %d, J (read) = %.12f, kT = %.4f, pilot ⟨P₂⟩_iso = %.4f", N, J, kT, p2)
@printf(" → K (read) = %.6f (βK = %.4f)\n", K, K / kT)

# --- nodes ----------------------------------------------------------------------------
θa = acos(0.95)
Δθ = (π / 2 - θa) / (NNODE - 1)
θs = [θa + (i - 1) * Δθ for i = 1:NNODE]
even = 1:2:NNODE                          # Simpson partial sums land on these nodes
cs = [max(cos(θs[i]), 0.0) for i in even]
lo = max.(cs .- HBIN, 0.0)
hi = cs .+ HBIN
meas = 2 .* (hi .- lo)                    # c-measure of the folded bin (both signs)
all(hi[2:end] .<= lo[1:(end - 1)]) || error("histogram bins overlap")

# --- histogram (unconstrained run_mc) ---------------------------------------------------
nb = length(even)
hnames = [Symbol("h", k) for k = 1:nb]
hobs = [Observable(hnames[k], 1, (cfg, E, Hr) -> begin
            M = sum(cfg)
            c = abs(M[3]) / norm(M)
            (lo[k] <= c < hi[k]) ? 1.0 : 0.0
        end) for k = 1:nb]
devals = [Evaluable(Symbol("dF", k), [hnames[1], hnames[k]],
                    (m, kT_, n) -> -kT_ * log((getfield(m, hnames[k]) / meas[k]) /
                                              (getfield(m, hnames[1]) / meas[1])))
          for k = 2:nb]
t_hist = @elapsed hist = let
    tasks = [Threads.@spawn run_mc(H; kT = kT, sweeps_therm = 20_000,
                                   sweeps_measure = sweeps_hist, measure_interval = 1,
                                   or_per_metropolis = 1,
                                   observables = [standard_observables(H); hobs],
                                   evaluables = [standard_evaluables(); devals],
                                   seed = 1200 + r) for r = 1:4]
    fetch.(tasks)
end
@printf("histogram run_mc: 4 replicas × %d sweeps (OR 1), %.1f s; acc %.3f\n",
        sweeps_hist, t_hist, mean(r -> r.points[1].acceptance_metropolis, hist))
dFh = [k == 1 ? (; mean = 0.0, σ = 0.0, chi2 = NaN) :
       pool([r.points[1].stats[Symbol("dF", k)].mean[1] for r in hist],
            [r.points[1].stats[Symbol("dF", k)].err[1] for r in hist]) for k = 1:nb]
frac = [mean(r -> r.points[1].stats[hnames[k]].mean[1], hist) for k = 1:nb]

# --- CMC lanes --------------------------------------------------------------------------
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
cons = [MagnetizationConstraint(θ, φC; weights = :uniform) for θ in θs]
t_cmc = @elapsed rs = run_cmc(H, cons; kT = kT, sweeps_therm = 10_000,
                              sweeps_measure = sweeps_cmc, measure_interval = 1,
                              observables = c -> frame_obs(H, c), seed = 1300)
@printf("run_cmc: %d lanes × %d sweeps, %.1f s; acc %.3f\n", NNODE, sweeps_cmc, t_cmc,
        mean(r -> r.points[1].acceptance_metropolis, rs))
# total torque per cell = per site here (1 atom/cell); ΔF in the same units (×N for F)
uT = [r.points[1].stats[:Tframe].mean[1] for r in rs] .* N
σu = [r.points[1].stats[:Tframe].err[1] for r in rs] .* N
w = simpson_weights(3, Δθ)

println("\n  θ        ⟨ê_φ·T⟩ (total)       ⟨ê_θ·T⟩/σ  ⟨n̂·T⟩/σ  M∥ per site")
okperp = true
for (i, r) in enumerate(rs)
    st = r.points[1].stats[:Tframe]
    zv, zn = st.mean[2] / st.err[2], st.mean[3] / st.err[3]
    global okperp &= abs(zv) < 4 && abs(zn) < 4
    @printf("  %.4f  % .5e ± %.1e   %+5.2f     %+5.2f    %.4f\n", θs[i], uT[i], σu[i],
            zv, zn, r.points[1].stats[:M_par].mean[1])
end

println("\n  node θ     |c| bin         frac      βΔF_hist (χ²)           " *
        "βΔF_cmc              dev")
okrows = true
F = 0.0
for (k, i) in enumerate(even)
    if k > 1                               # Simpson panel [i−2, i]
        global F += -sum(w .* uT[(i - 2):i])
    end
    # variance of the cumulative Simpson sum: per-node weights (a node shared by two
    # panels carries both), lanes independent
    wn = zeros(NNODE)
    for kk = 2:k
        j = even[kk]
        wn[(j - 2):j] .+= w
    end
    σF = sqrt(sum(abs2, wn .* σu))
    h = dFh[k]
    if k == 1
        @printf("  %.4f  [%.3f, %.3f)  %.5f   (reference bin)\n", θs[i], lo[k], hi[k],
                frac[k])
        continue
    end
    σc = sqrt(σF^2 + h.σ^2)
    dev = (F - h.mean) / σc
    ok = abs(dev) < 4
    global okrows &= ok
    @printf("  %.4f  [%.3f, %.3f)  %.5f   %.4f ± %.4f (%.2f)   %.4f ± %.4f   %+5.2fσ %s\n",
            θs[i], lo[k], hi[k], frac[k], h.mean / kT, h.σ / kT, h.chi2, F / kT, σF / kT,
            dev, ok ? "ok" : "FAIL")
    if k == nb
        global endσ = σc / kT
        global endβΔF = h.mean / kT
    end
end
res_ok = endσ <= 0.1 && 1.5 <= endβΔF <= 4.5
@printf("\nendpoint: βΔF(θ_a → π/2) = %.3f, σ_comb = %.3f kT  → resolution/regime %s\n",
        endβΔF, endσ, res_ok ? "ok" : "INCONCLUSIVE")
println("G11 verdict: ", okrows && okperp ? "PASS" : "FAIL",
        " (ΔF rows within 4σ_comb; ê_θ·T, n̂·T zero within 4σ)",
        res_ok ? "" : "; resolution/regime NOT met")

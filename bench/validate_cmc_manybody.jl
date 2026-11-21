# G8 — many-body composition of constrained MC against the rotated-iid reference
# (docs/specs/constrained-mc.md, "Manual validation", row G8).
#
#   julia --project=bench -t 4 bench/validate_cmc_manybody.jl [nsamp_ref] [sweeps_cmc]
#
# ORACLE (independent of the CMC code). For iid uniform spins e, rotate the whole
# configuration rigidly by the minimal rotation R(M̂_w(e)) that takes M̂_w = M_w/|M_w|
# (M_w = Σ_i w_i e_i) onto n̂. Because the E = 0 constrained measure at n₀ is invariant
# under rotations about n₀, e′ = R e is an EXACT sample of the E = 0 constrained
# ensemble at n̂ — the |M|² Jacobian of δ_{S²}(M̂ − n̂) included. Reweighting by
# e^{−βE(e′)} (E from `total_energy`, the from-scratch energy path) gives
#     ⟨A⟩_n̂ = E_iid[A(e′) e^{−βE(e′)}] / E_iid[e^{−βE(e′)}].
# The torque components are measured on the reference side WITHOUT the package's
# gradient code: u·T = −dE(R_u(α) e′)/dα by a central difference of `total_energy`
# (h = 1e-4; truncation ~1e-9 relative). M∥ = |M_w(e)| (a rotation keeps the norm).
# Reference error: 64 batches, σ = std(batch ratio)/√64 (captures the reweighting
# ESS loss); the Kish ESS is printed.
#
# CMC side: `run_cmc` lanes (4 duplicate lanes per case = independent replicas), with
# one extra 3-component observable (ê_φ·T, ê_θ·T, n̂·T) of `total_torque`; pooled
# mean, σ = √(Σ σ_lane²)/4 from the binning errors, replica χ² printed as a check.
#
# FIXTURES (copied from test/unit/fixtures_cmc.jl, same seeds; N = 3 or 4 sites, dims
# (1,1,1)):
#   F3  — isotropic pairs, a 3-body ls = [1,1,1] dense random term (ODD under time
#         reversal: Σl = 3), a body-1 l = 2 term on every atom.
#   F3w — F3 under the weights (1, −0.6, 1.4) (mixed signs, unequal).
#   F4  — a 4-body ls = [1,1,1,1] term + an isotropic pair (EVEN under time reversal).
#   F4o — 4 atoms, a 4-body ls = [1,1,1,2] dense random term (ODD: Σl = 5), isotropic
#         pairs (1,2), (3,4), a body-1 l = 2 term on atom 4 (added here for G8).
# Each odd model is also run at −n̂: a chain that leaked through M∥′ ≤ 0 into the −n̂
# hemisphere would mix the two ensembles, which differ ONLY for a time-reversal-odd
# model (for an even one ⟨T⟩(−n̂) = ⟨T⟩(n̂) exactly — printed for F4 as the control).
#
# β RULE (fixed before any comparison): β = 1.5 / std_iid(E) from a 10⁴-sample pilot
# (keeps the reweighting ESS ≳ 10 % while the Boltzmann factor is far from flat).
#
# TOLERANCE RULE (fixed before looking at the result):
#   - every compared quantity (ê_φ·T, ê_θ·T, M∥; n̂·T against the exact 0) passes when
#     |cmc − ref| < 4 σ_comb, σ_comb = √(σ_cmc² + σ_ref²) (≈ 20 comparisons ⇒ a
#     false alarm probability ~1e-3);
#   - resolution: σ_comb of both transverse components ≤ 3 % of the reference
#     transverse torque |⟨T⊥⟩| = √(⟨ê_φ·T⟩² + ⟨ê_θ·T⟩²) (so the 25 % shift of a naive
#     pair ΔE seen in the numpy prototype is ≥ 8σ); a case that misses it is
#     "underpowered", not a pass. (Scale fixed after the small-size dry run, before
#     the production run: the first wording used |⟨ê_φ·T⟩| alone, which vanishes by
#     accident for F4o at +n̂.)
#   - leak visibility (odd models): |⟨T⟩_ref(n̂) − ⟨T⟩_ref(−n̂)| projected on ê_φ(n̂) or
#     ê_θ(n̂) ≥ 5 σ (reported; it says the gate can see a leak, it does not pass/fail
#     the code).

include(joinpath(@__DIR__, "validate_cmc_common.jl"))

nsamp_ref = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000_000
sweeps_cmc = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2_000_000

banner("G8 — rotated-iid reference vs run_cmc (many-body, N = 3/4), " *
       "nsamp_ref = $nsamp_ref, sweeps_cmc = $sweeps_cmc × 4 lanes")

const Z0 = SVector(0, 0, 0)
iso_folded(l) = Matrix{Float64}(I, 2l + 1, 2l + 1)

function f3_terms()                   # = _f3_terms() of test/unit/fixtures_cmc.jl
    rng = MersenneTwister(3101)
    terms = [MultipoleTerm(-0.02, 2, [a, b], [Z0, Z0], [1, 1], iso_folded(1))
             for (a, b) in ((1, 2), (1, 3), (2, 3))]
    push!(terms, MultipoleTerm(0.01, 3, [1, 2, 3], [Z0, Z0, Z0], [1, 1, 1],
                               randn(rng, 3, 3, 3)))
    onsite = randn(rng, 5)
    for a = 1:3
        push!(terms, MultipoleTerm(0.01, 1, [a], [Z0], [2], copy(onsite)))
    end
    return terms
end

function f4_terms()                   # = _f4_terms() of test/unit/fixtures_cmc.jl
    folded = zeros(3, 3, 3, 3)
    folded[2, 2, 2, 2] = 1.0
    folded[1, 3, 2, 1] = 0.6
    folded[3, 2, 1, 3] = -0.3
    folded[1, 1, 3, 2] = 0.45
    return [MultipoleTerm(0.02, 4, [1, 2, 3, 4], [Z0, Z0, Z0, Z0], [1, 1, 1, 1], folded),
            MultipoleTerm(-0.02, 2, [1, 2], [Z0, Z0], [1, 1], iso_folded(1))]
end

function f4o_terms()                      # F4o: odd 4-body, defined here
    rng = MersenneTwister(4102)
    return [MultipoleTerm(0.004, 4, [1, 2, 3, 4], [Z0, Z0, Z0, Z0], [1, 1, 1, 2],
                          randn(rng, 3, 3, 3, 5)),
            MultipoleTerm(-0.02, 2, [1, 2], [Z0, Z0], [1, 1], iso_folded(1)),
            MultipoleTerm(-0.02, 2, [3, 4], [Z0, Z0], [1, 1], iso_folded(1)),
            MultipoleTerm(0.01, 1, [4], [Z0], [2], randn(rng, 5))]
end

# --- reference ------------------------------------------------------------------------

function pilot_beta(H, N; n = 10_000, seed = 77)
    rng = Xoshiro(seed)
    cfg = SpinConfig(undef, N)
    Es = map(1:n) do _
        for i = 1:N
            cfg[i] = rand_unit(rng)
        end
        total_energy(H, cfg)
    end
    return 1.5 / std(Es), minimum(Es)
end

# Rotated-iid reference at n̂(θ, φ) with weights w: batch sums of the Boltzmann weight
# and of weight × (ê_φ·T, ê_θ·T, M∥).
function reference(H, N, w, θ, φ, β, Eshift, nsamp; seed = 1, nbatch = 64, h = 1e-4)
    n̂, u, v = vdir(θ, φ), veφ(θ, φ), veθ(θ, φ)
    per = cld(nsamp, nbatch)
    S = zeros(nbatch, 5)                  # Σwt, Σwt·uT, Σwt·vT, Σwt·M∥, Σwt²
    Threads.@threads for b = 1:nbatch
        rng = Xoshiro(UInt64(seed) * 1_000_003 + UInt64(b))
        e = Vector{SVector{3,Float64}}(undef, N)
        cfg = SpinConfig(undef, N)
        acc = zeros(5)
        for _ = 1:per
            M = zero(SVector{3,Float64})
            for i = 1:N
                e[i] = rand_unit(rng)
                M += w[i] * e[i]
            end
            R = min_rot(normalize(M), n̂)
            for i = 1:N
                cfg[i] = R * e[i]
            end
            E = total_energy(H, cfg)
            Ed(axis, α) = total_energy(H, SpinConfig([rot_about(x, axis, α) for x in cfg]))
            uT = -(Ed(u, h) - Ed(u, -h)) / (2h)
            vT = -(Ed(v, h) - Ed(v, -h)) / (2h)
            wt = exp(-β * (E - Eshift))
            acc[1] += wt
            acc[2] += wt * uT
            acc[3] += wt * vT
            acc[4] += wt * norm(M)
            acc[5] += wt^2
        end
        S[b, :] .= acc
    end
    tot = vec(sum(S; dims = 1))
    ess = tot[1]^2 / tot[5]
    out = map(2:4) do k
        batch = S[:, k] ./ S[:, 1]
        (; mean = tot[k] / tot[1], σ = std(batch) / sqrt(nbatch))
    end
    return (; uT = out[1], vT = out[2], Mpar = out[3], ess = ess,
            n = per * nbatch)
end

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

function run_case(name, terms, N, weights, θ, φ, β, Eshift; seed)
    H = TiledHamiltonian(N, terms; dims = (1, 1, 1))
    wv = weights === :uniform ? ones(N) : weights
    t_ref = @elapsed ref = reference(H, N, wv, θ, φ, β, Eshift, nsamp_ref; seed = seed)
    con = MagnetizationConstraint(θ, φ; weights = weights === :uniform ? :uniform :
                                                 weights)
    t_cmc = @elapsed rs = run_cmc(H, fill(con, 4); kT = 1 / β, sweeps_therm = 10_000,
                                  sweeps_measure = sweeps_cmc, measure_interval = 1,
                                  observables = c -> frame_obs(H, c), seed = seed)
    uT = pool_stat(rs, :Tframe, 1)
    vT = pool_stat(rs, :Tframe, 2)
    nT = pool_stat(rs, :Tframe, 3)
    Mp = pool_stat(rs, :M_par, 1)
    resid = maximum(r -> r.points[1].stats[:constraint_residual].mean[1], rs)
    acc = mean(r -> r.points[1].acceptance_metropolis, rs)
    @printf("\n[%s] θ = %.4f, φ = %.4f, weights = %s, β = %.4g\n", name, θ, φ,
            string(weights), β)
    @printf("  ref: n = %d, Kish ESS = %.3g (%.1f %%), %.1f s\n", ref.n,
            ref.ess, 100 * ref.ess / ref.n, t_ref)
    @printf("  cmc: acc %.3f, step %.3g, max ⟨residual⟩ %.1e, ",
            acc, rs[1].points[1].final_step, resid)
    @printf("χ²/dof (uT, vT, M∥) = %.2f, %.2f, %.2f, %.1f s\n",
            uT.chi2, vT.chi2, Mp.chi2, t_cmc)
    ok = true
    ok &= check_row("ê_φ·T (= −∂F/∂θ)", uT.mean, uT.σ, ref.uT.mean, ref.uT.σ)
    ok &= check_row("ê_θ·T (= ∂F/∂φ / sinθ)", vT.mean, vT.σ, ref.vT.mean, ref.vT.σ)
    ok &= check_row("M∥", Mp.mean, Mp.σ, ref.Mpar.mean, ref.Mpar.σ)
    ok &= check_row("n̂·T (exact 0)", nT.mean, nT.σ, 0.0, 0.0)
    σc = max(sqrt(uT.σ^2 + ref.uT.σ^2), sqrt(vT.σ^2 + ref.vT.σ^2))
    Tperp = hypot(ref.uT.mean, ref.vT.mean)
    res_ok = σc <= 0.03 * Tperp
    @printf("  resolution: max σ_comb(T⊥ comps)/|⟨T⊥⟩_ref| = %.2f %%  %s\n",
            100 * σc / Tperp, res_ok ? "ok" : "UNDERPOWERED")
    # Cartesian reference torque, for the ±n̂ comparison
    Tcart = ref.uT.mean * veφ(θ, φ) + ref.vT.mean * veθ(θ, φ)
    σcart = sqrt(ref.uT.σ^2 + ref.vT.σ^2)
    return (; ok, res_ok, Tcart, σcart, ref)
end

cases = Tuple{String,Bool}[]
results = Dict{String,Any}()
θ0, φ0 = 0.6, 0.35
θm, φm = π - θ0, φ0 + π                    # −n̂(θ0, φ0)

for (fname, terms, N) in (("F3", f3_terms(), 3), ("F4", f4_terms(), 4),
                          ("F4o", f4o_terms(), 4))
    H = TiledHamiltonian(N, terms; dims = (1, 1, 1))
    β, Emin = pilot_beta(H, N)
    @printf("\n%s: N = %d, n_active = %d, terms = %d; ",
            fname, N, H.n_active, length(H.terms))
    @printf("pilot β = %.4g (1.5/std E), min pilot E = %.4g\n", β, Emin)
    results[fname * " +n"] = run_case(fname * " +n̂", terms, N, :uniform, θ0, φ0, β, Emin;
                                      seed = 11)
    results[fname * " -n"] = run_case(fname * " −n̂", terms, N, :uniform, θm, φm, β, Emin;
                                      seed = 12)
    if fname == "F3"
        results["F3w +n"] = run_case("F3w +n̂", terms, N, [1.0, -0.6, 1.4], θ0, φ0, β,
                                     Emin; seed = 13)
    end
end

println("\n--- leak visibility: reference ⟨T⟩(n̂) − ⟨T⟩(−n̂), " *
        "projected on ê_φ(n̂), ê_θ(n̂)")
for f in ("F3", "F4o", "F4")
    a, b = results[f * " +n"], results[f * " -n"]
    d = a.Tcart - b.Tcart
    σ = sqrt(a.σcart^2 + b.σcart^2)
    du, dv = dot(d, veφ(θ0, φ0)), dot(d, veθ(θ0, φ0))
    @printf("  %-4s Δ(ê_φ·T) = % .4e, Δ(ê_θ·T) = % .4e, σ ≈ %.2e  → max %.1fσ  %s\n",
            f, du, dv, σ, max(abs(du), abs(dv)) / σ,
            f == "F4" ? "(even model: expected 0)" :
            (max(abs(du), abs(dv)) / σ >= 5 ? "leak visible" : "leak NOT resolved"))
end

allok = all(r -> r.ok, values(results))
allres = all(r -> r.res_ok, values(results))
println("\nG8 verdict: ", allok ? "PASS" : "FAIL", " (all rows within 4σ_comb)",
        allres ? "" : "; some case UNDERPOWERED")

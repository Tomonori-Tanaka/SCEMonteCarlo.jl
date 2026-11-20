# G12 — production smoke of constrained MC on a Nd₂Fe₁₄B model
# (docs/specs/constrained-mc.md, "Manual validation"; gate table of the spec, G12).
#
#   julia --project=bench -t 4 bench/validate_cmc_smoke.jl [n] [sweeps_measure] [L02_DIR] [seed]
#
# MODEL. The real exchange of the l02 production model is rebuilt from its training
# data with the recipe of `<L02_DIR>/test/fit_mfa.jl` (68-atom cell, nbody = 2, every
# resolvable pair, lmax 1 on Nd/Fe and 0 on B, isotropic, OLS with torque_weight = 1;
# default L02_DIR = ~/jijs/magesty/2-14-1/nd2fe14b/1x1x1/magesty/l02). That model is
# ISOTROPIC — its torque vanishes configuration by configuration — so on its own the
# smoke would be blind. A SYNTHETIC, symmetry-adapted anisotropy is added, built by
# SCEFitting on the same crystal with the spglib backend (so every term respects the
# P4₂/mnm space group and time reversal):
#   A1 — body-1 l = 2 SALCs on Nd (single-ion), seeded random coefficients scaled so
#        that the uniform ferrimagnetic state costs +30 meV per cell along a vs c
#        (easy c axis; the order of K₁V of Nd₂Fe₁₄B at room temperature);
#   A2 — anisotropic l = 1 ⊗ l = 1 Fe–Fe pairs within 2.6 Å, seeded random
#        coefficients scaled to a uniform-state anisotropy of magnitude 10 meV/cell.
# The physics of the numbers is therefore NOT Nd₂Fe₁₄B's; the smoke checks the
# pipeline (normalization, symmetry, closed path, partial torques, cost) on a
# production-size, production-symmetry Hamiltonian. Moments (:auto weights and the
# Fe-only hybrid weights) are the mean |m| per atom of the training set.
#
# RUN. Supercell n×n×n (default 2: 544 sites, 512 active), constraint directions in
# the a–c plane, n̂(θ, φ = 0), θ = 0, π/8, …, π (9 lanes), two weight choices
# (`:auto` = moments; Fe-only = Nd weight 0, Nd sites updated by single-spin
# Metropolis), one annealing ladder 600 K → 300 K per lane (both below T_c of l02,
# ≈ 835 K), start = ferrimagnetic pattern (Fe ∥ +ẑ, Nd ∥ −ẑ) rotated onto n̂.
# Measured: (ê_φ·T, ê_θ·T, n̂·T) per cell from one `total_torque` per measurement,
# and the body-1 / body-2 partial torques (`torque_observable`).
#
# SYMMETRY PRECONDITION (deterministic, before any MC): the symmetry the checks below
# rely on is verified on the TILED Hamiltonian, not assumed: for the space-group
# operations whose proper part is C4z, C2x, and C2y (the axial-vector action of the
# mirror m_y), every spin is moved to the image site (atom map of the space group +
# the image cell) and rotated by the proper part, and `total_energy` must not change
# (1e-10 relative, 5 random configurations each).
#
# CHECKS (preconditions from the model's symmetry; decided before the run). Let
# τ(θ) = ⟨ê_φ·T⟩ per cell (= −∂f/∂θ), σ its binning error, σ_comb for two lanes
# √(σ₁² + σ₂²):
#   (a) ⟨n̂·T⟩ = 0 within 4σ at every lane (exact for any H);
#   (b) ⟨ê_θ·T⟩ = 0 within 4σ at every lane: the proper part of the mirror m_y acts on
#       spins as C2y, n̂(θ, φ) → −n̂(θ, −φ); with time-reversal evenness F(θ, φ) =
#       F(θ, −φ), so ∂F/∂φ = 0 on the a–c plane;
#   (c) τ(0) = τ(π) = 0 within 4σ (C4z ∥ c) and τ(π/2) = 0 within 4σ (C2x: n̂ = â is
#       on a 2-fold axis);
#   (d) τ(θ) + τ(π − θ) = 0 within 4σ_comb (C2x maps n̂(θ, 0) to n̂(π − θ, 0));
#   (e) closed path ∫₀^π τ dθ = 0 within 4σ (Simpson on the 9 nodes; F(n̂) = F(−n̂)
#       for a time-reversal-even H at B = 0);
#   (f) partial torques: ⟨T_body1⟩ + ⟨T_body2⟩ = ⟨T⟩ to 1e-10 relative (a
#       configuration-by-configuration identity: the means must agree to rounding);
#   (g) health: mean :constraint_residual ≤ 1e-10; acceptance in [0.05, 0.95]; final
#       step above 10× the clamp floor (1e-3); M∥ per cell ≥ 0.3 Σ|w| per cell (no
#       weak constraint); no SCEMonteCarlo warning in the log (read by eye; the
#       SCEFitting alias-group notes of the l02 refit are expected).
# Reported, not gated: τ(θ) itself (and the T = 0 rigid-rotation torque of the
# uniform ferrimagnetic state for scale), timing per sweep.

include(joinpath(@__DIR__, "validate_cmc_common.jl"))
using TOML: TOML
using Spglib: Spglib                      # activates SCEFitting's SpglibBackend

const NDIM = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const SWEEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30_000
const L02 = length(ARGS) >= 3 ? ARGS[3] :
            expanduser("~/jijs/magesty/2-14-1/nd2fe14b/1x1x1/magesty/l02")
const SEED = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1204
const TEMPS = [600.0, 300.0]
const NTHETA = 9
const MEAS_INT = 5

banner("G12 — Nd₂Fe₁₄B l02 (+ synthetic symmetric anisotropy) CMC smoke, " *
       "dims = $(NDIM)³, sweeps_measure = $SWEEPS, T = $TEMPS K, seed = $SEED")

# --- 1. crystal and the l02 exchange (fit_mfa.jl recipe) --------------------------------
inp = TOML.parsefile(joinpath(L02, "input.toml"))
labels = String.(inp["general"]["kd"])
st = inp["structure"]
A = reduce(hcat, [Float64.(v) for v in st["lattice"]])
species = Int.(st["kd_list"])
nat = length(species)
frac = mod.(reduce(hcat, [Float64.(p) for p in st["position"]]), 1.0)
crystal = Crystal(SCEFitting.Lattice(SMatrix{3,3,Float64}(A)), frac, species, labels)
isNd = [startswith(labels[s], "Nd") for s in species]
isFe = [startswith(labels[s], "Fe") for s in species]
@printf("crystal: %d atoms (%s); lattice columns |a| = %.4f, %.4f, %.4f Å\n", nat,
        join(labels, " "), norm(A[:, 1]), norm(A[:, 2]), norm(A[:, 3]))

t_fit = @elapsed begin
    lmax = [lab == "B" ? 0 : 1 for lab in labels]
    basis = SCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = Inf, lmax = lmax,
                                        isotropy = true);
                     backend = SpglibBackend(), tol = 1.0e-5)
    data = read_embset(joinpath(L02, "EMBSET"))
    fitted = fit(SCEFit, SCEDataset(basis, data), OLS(); torque_weight = 1.0)
    l02 = SCEPredictor(fitted)
end
@printf("l02 refit: %d SALCs, %d configs, R2(E) = %.6f, rmse(T) = %.4g eV (July log: 179 SALCs, R2(E) = 0.977563, rmse(T) = 0.0123) [%.1f s]\n",
        n_salcs(basis), length(data), r2_energy(fitted), rmse_torque(fitted), t_fit)
magmom = [mean(d -> d.magmoms[a], data) for a = 1:nat]
mNd = mean(magmom[isNd])
mFe = mean(magmom[isFe])
@printf("moments (mean |m| over the training set): Nd %.4f, Fe %.4f, B %.4f μ_B\n", mNd,
        mFe, mean(magmom[.!(isNd .| isFe)]))

# --- 2. synthetic symmetric anisotropy ----------------------------------------------------
ferri(n̂) = SpinConfig([isNd[a] ? -n̂ : n̂ for a = 1:nat])
cellE(terms, cfg) = total_energy(TiledHamiltonian(nat, terms; dims = (1, 1, 1)), cfg)
x̂, ẑ = SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0)
function scaled_terms(b::SCEBasis, seed, target; keep_sign = false)
    c = randn(MersenneTwister(seed), n_salcs(b))
    t = multipole_terms(SCEPredictor(b, 0.0, c))
    d = cellE(t, ferri(x̂)) - cellE(t, ferri(ẑ))
    s = keep_sign ? abs(target / d) : target / d
    return multipole_terms(SCEPredictor(b, 0.0, s .* c))
end
lmaxA1 = [startswith(lab, "Nd") ? 2 : 0 for lab in labels]
bA1 = SCEBasis(crystal, BasisSpec(; nbody = 1, cutoff = Inf, lmax = lmaxA1,
                                  isotropy = false);
               backend = SpglibBackend(), tol = 1.0e-5)
tA1 = scaled_terms(bA1, 1201, 0.030)                  # +30 meV/cell: easy c axis
lmaxA2 = [startswith(lab, "Fe") ? 1 : 0 for lab in labels]
bA2 = SCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = lmaxA2,
                                  isotropy = false);
               backend = SpglibBackend(), tol = 1.0e-5)
tA2 = scaled_terms(bA2, 1202, 0.010; keep_sign = true)
tA2 = filter(t -> t.body == 2, tA2)
tl02 = multipole_terms(l02)
terms = [tl02; tA1; tA2]
rng = Xoshiro(1203)
rms(t) = sqrt(mean(abs2, [cellE(t, SpinConfig([rand_unit(rng) for _ = 1:nat])) for _ = 1:200]) -
              mean([cellE(t, SpinConfig([rand_unit(rng) for _ = 1:nat])) for _ = 1:200])^2)
@printf("terms: l02 %d (body-2), A1 %d (body %s), A2 %d (body-2); A1/A2 SALCs %d/%d\n",
        length(tl02), length(tA1), string(unique(t.body for t in tA1)), length(tA2),
        n_salcs(bA1), n_salcs(bA2))
@printf("cell energies [meV]: E(ferri ∥ a) − E(ferri ∥ c): A1 %+.3f, A2 %+.3f, l02 %+.2e; random-config spread l02 %.1f, A1 %.2f, A2 %.2f\n",
        1e3 * (cellE(tA1, ferri(x̂)) - cellE(tA1, ferri(ẑ))),
        1e3 * (cellE(tA2, ferri(x̂)) - cellE(tA2, ferri(ẑ))),
        1e3 * (cellE(tl02, ferri(x̂)) - cellE(tl02, ferri(ẑ))),
        1e3 * rms(tl02), 1e3 * rms(tA1), 1e3 * rms(tA2))
ferro = SpinConfig(fill(ẑ, nat))
@printf("E(ferri ∥ c) − E(all ∥ c) = %+.1f meV/cell (ferrimagnetic start must be lower)\n",
        1e3 * (cellE(terms, ferri(ẑ)) - cellE(terms, ferro)))

# --- 3. tiled Hamiltonians ------------------------------------------------------------------
dims = (NDIM, NDIM, NDIM)
t_tile = @elapsed begin
    H = TiledHamiltonian(nat, terms; dims = dims, magmoms = magmom)
    H1 = TiledHamiltonian(nat, filter(t -> t.body == 1, terms); dims = dims)
    H2 = TiledHamiltonian(nat, filter(t -> t.body == 2, terms); dims = dims)
end
ncell = prod(dims)
@printf("H: %d sites (%d active), %d templates, nlm %d [%.1f s]; partial H1 (body 1) %d, H2 (body 2) %d templates\n",
        n_sites(H), H.n_active, length(H.terms), H.nlm, t_tile, length(H1.terms),
        length(H2.terms))

# --- symmetry precondition on the tiled H -------------------------------------------------
sg = SCEFitting.analyze_symmetry(SpglibBackend(), crystal; tol = 1.0e-5)
@printf("space group: %s (#%d), %d operations\n", sg.symbol, sg.number, length(sg.ops))
function site_image(op, k, s)               # image site of supercell site s under op k
    a = MC.site_atom(H, s)
    c = (s - 1) ÷ nat                       # cell index, site_index layout: atom fastest
    t = SVector(c % NDIM, (c ÷ NDIM) % NDIM, c ÷ NDIM^2)
    r = op.rotation_frac * (t + SVector{3}(frac[:, a])) + op.translation_frac
    a2 = sg.map_sym[a, k]
    t2 = round.(Int, r - SVector{3}(frac[:, a2]))
    @assert maximum(abs, r - SVector{3}(frac[:, a2]) - t2) < 1e-4
    return MC.site_index(H, a2, mod.(t2, NDIM))
end
for (name, Rwant) in (("C4z", SMatrix{3,3}(0.0, 1, 0, -1, 0, 0, 0, 0, 1)),
                      ("C2x", SMatrix{3,3}(1.0, 0, 0, 0, -1, 0, 0, 0, -1)),
                      ("C2y (m_y on spins)", SMatrix{3,3}(-1.0, 0, 0, 0, 1, 0, 0, 0, -1)))
    ks = findall(op -> maximum(abs, det(op.rotation_cart) * op.rotation_cart - Rwant) < 1e-8,
                 sg.ops)
    isempty(ks) && error("no operation with proper part $name")
    k = ks[1]
    op = sg.ops[k]
    Rp = det(op.rotation_cart) * op.rotation_cart
    perm = [site_image(op, k, s) for s = 1:n_sites(H)]
    isperm(perm) || error("$name: the site map is not a permutation")
    # control: the same site map WITHOUT the spin rotation must change the energy
    # (the check can fail), as long as the anisotropy is present
    worst, control = 0.0, Inf
    for _ = 1:5
        cfg = SpinConfig([rand_unit(rng) for _ = 1:n_sites(H)])
        img, ctl = similar(cfg), similar(cfg)
        for s = 1:n_sites(H)
            img[perm[s]] = Rp * cfg[s]
            ctl[perm[s]] = cfg[s]
        end
        E0 = total_energy(H, cfg)
        worst = max(worst, abs(total_energy(H, img) - E0) / abs(E0))
        control = min(control, abs(total_energy(H, ctl) - E0) / abs(E0))
    end
    @printf("  %-20s (op %d, first of %d matches; translation %s): max |ΔE|/|E| = %.1e  %s;  control without spin rotation: min |ΔE|/|E| = %.1e\n",
            name, k, length(ks), string(round.(op.translation_frac; digits = 3)),
            worst, worst < 1e-10 ? "ok" : "BROKEN", control)
    control > 1e-6 || error("$name: the control did not change the energy")
    worst < 1e-10 || error("the tiled Hamiltonian does not have the $name symmetry")
end

# rigid T = 0 torque of the uniform ferrimagnetic state, per cell: −dE/dθ (central diff)
function rigid_tau(θ; h = 1e-5)
    E(t) = total_energy(H, SpinConfig([isNd[MC.site_atom(H, s)] ? -vdir(t, 0.0) :
                                       vdir(t, 0.0) for s = 1:n_sites(H)]))
    return -(E(θ + h) - E(θ - h)) / (2h) / ncell
end

# --- 4. CMC lanes -----------------------------------------------------------------------------
θs = [(k - 1) * π / (NTHETA - 1) for k = 1:NTHETA]
w_fe = [isFe[a] ? magmom[a] : 0.0 for a = 1:nat]
init = SpinConfig([isNd[MC.site_atom(H, s)] ? -ẑ : ẑ for s = 1:n_sites(H)])

function smoke_obs(H, c)
    n̂ = c.direction
    # frame on the a–c plane (φ = 0): ê_φ = ŷ, ê_θ = (cosθ, 0, −sinθ); θ = 0, π included
    θ = atan(n̂[1], n̂[3])
    u, v = SVector(0.0, 1.0, 0.0), SVector(cos(θ), 0.0, -sin(θ))
    nc = n_sites(H) ÷ H.n_cell_atoms
    base = filter(o -> o.name !== :torque, cmc_observables(H, c))
    return [base;
            Observable(:Tframe, 3, (cfg, E, Hr) -> begin
                T = MC.total_torque(H, cfg) / nc
                SVector(dot(u, T), dot(v, T), dot(n̂, T))
            end);
            torque_observable(:T_b1, H1, H);
            torque_observable(:T_b2, H2, H)]
end

cons = [[MagnetizationConstraint(θ, 0.0; weights = :auto) for θ in θs];
        [MagnetizationConstraint(θ, 0.0; weights = w_fe) for θ in θs]]
t_run = @elapsed rs = run_cmc(H, cons; temperature = TEMPS, sweeps_therm = 3_000,
                              sweeps_measure = SWEEPS, measure_interval = MEAS_INT,
                              init = init, observables = c -> smoke_obs(H, c),
                              seed = SEED)
nsw = length(cons) * length(TEMPS) * (3_000 + SWEEPS)
@printf("run_cmc: %d lanes × %d temperatures × (%d + %d) sweeps, measure_interval %d: %.1f s wall on %d threads → %.3f ms per lane-sweep (incl. 3 torques / %d sweeps)\n",
        length(cons), length(TEMPS), 3_000, SWEEPS, MEAS_INT, t_run, Threads.nthreads(),
        1e3 * t_run * min(Threads.nthreads(), length(cons)) / nsw, MEAS_INT)

# --- 5. checks ------------------------------------------------------------------------------
allok = true
function gate(cond, msg)
    cond || println("    FAIL: ", msg)
    global allok &= cond
    return cond
end
sw = simpson_weights(NTHETA, θs[2] - θs[1])
for (wi, wname) in enumerate(("weights :auto (moments)", "weights Fe-only (Nd hybrid)"))
    lanes = rs[((wi - 1) * NTHETA + 1):(wi * NTHETA)]
    Σw = (wi == 1 ? sum(abs, magmom[isNd .| isFe]) : sum(abs, w_fe))   # per cell
    for (ti, T) in enumerate(TEMPS)
        println("\n[$wname, T = $T K]")
        println("   θ/π    τ = ⟨ê_φ·T⟩ [meV/cell]   τ_int  ⟨ê_θ·T⟩/σ  ⟨n̂·T⟩/σ   τ_b1 + τ_b2 − τ      M∥/Σ|w|  resid    acc    step   τ_rigid(T=0)")
        τ = zeros(NTHETA)
        στ = zeros(NTHETA)
        for (k, r) in enumerate(lanes)
            p = r.points[ti]
            fr = p.stats[:Tframe]
            τ[k], στ[k] = fr.mean[1], fr.err[1]
            zθ, zn = fr.mean[2] / fr.err[2], fr.mean[3] / fr.err[3]
            u = SVector(0.0, 1.0, 0.0)
            Tsum = p.stats[:T_b1].mean .+ p.stats[:T_b2].mean
            scale = norm(p.stats[:T_b1].mean) + norm(p.stats[:T_b2].mean)
            # frame comps of the partial sum, against the frame comps measured directly
            θk = θs[k]
            v = SVector(cos(θk), 0.0, -sin(θk))
            n̂ = vdir(θk, 0.0)
            dpart = maximum(abs, [dot(u, Tsum), dot(v, Tsum), dot(n̂, Tsum)] .- fr.mean)
            Mrel = p.stats[:M_par].mean[1] / Σw
            res = p.stats[:constraint_residual].mean[1]
            @printf("  %.3f  %+9.4f ± %.4f      %5.1f  %+5.2f     %+5.2f     %.1e (rel %.0e)   %.3f   %.0e  %.3f  %.3f  %+9.4f\n",
                    θk / π, 1e3 * τ[k], 1e3 * στ[k], fr.tau_int[1], zθ, zn, dpart, dpart / scale, Mrel,
                    res, p.acceptance_metropolis, p.final_step, 1e3 * rigid_tau(θk))
            gate(abs(zn) < 4, "(a) n̂·T at θ = $θk")
            gate(abs(zθ) < 4, "(b) ê_θ·T at θ = $θk")
            gate(dpart <= 1e-10 * scale, "(f) partial torques at θ = $θk")
            gate(res <= 1e-10, "(g) residual at θ = $θk")
            gate(0.05 <= p.acceptance_metropolis <= 0.95, "(g) acceptance at θ = $θk")
            gate(p.final_step > 1e-2, "(g) step at θ = $θk")
            gate(Mrel >= 0.3, "(g) weak constraint at θ = $θk")
        end
        for k in (1, NTHETA, (NTHETA + 1) ÷ 2)
            gate(abs(τ[k]) < 4 * στ[k], "(c) τ = 0 at θ = $(θs[k])")
        end
        sym = [(τ[k] + τ[NTHETA + 1 - k]) / hypot(στ[k], στ[NTHETA + 1 - k])
               for k = 2:((NTHETA - 1) ÷ 2)]
        foreach(z -> gate(abs(z) < 4, "(d) τ(θ) + τ(π − θ) = $z σ"), sym)
        I = sum(sw .* τ)
        σI = sqrt(sum(abs2, sw .* στ))
        gate(abs(I) < 4 * σI, "(e) closed path")
        @printf("  (c) τ(0), τ(π/2), τ(π) = %+.2fσ, %+.2fσ, %+.2fσ; (d) τ(θ)+τ(π−θ) = %s σ; (e) ∫₀^π τ dθ = %+.4f ± %.4f meV/cell (%+.2fσ)\n",
                τ[1] / στ[1], τ[(NTHETA + 1) ÷ 2] / στ[(NTHETA + 1) ÷ 2],
                τ[end] / στ[end], join((@sprintf("%+.2f", z) for z in sym), ", "),
                1e3 * I, 1e3 * σI, I / σI)
    end
end
println("\nG12 verdict: ", allok ? "PASS" : "FAIL", " (checks (a)–(g))")

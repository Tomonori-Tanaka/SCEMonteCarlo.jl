# External field on the device — the on-device validation of the body-1 Zeeman
# templates (docs/specs/zeeman-field.md; the CPU-backend gates live in
# test/unit/test_gpu.jl, this script re-claims them on a real GPU).
#
#   julia --project=bench/gpu bench/bench_gpu_zeeman.jl [n_bcc] [nsweeps]
#
# Backend selection as bench_gpu.jl: CUDA when functional, else the KA CPU backend
# (smoke only). Gates — every one backend-portable, i.e. it never asks a CUDA
# trajectory to equal a host trajectory (the sweep's accept test goes through the
# platform libm and its RNG, so bitwise identity is a within-backend contract;
# docs/specs/gpu-prototype.md G5 / G7):
#   A  repeat-run bitwise identity of device sweeps with a field
#   B  incremental-energy drift vs the host from-scratch total (field included)
#   C  device acceptance ≈ same-node CPU acceptance at one kT
#   D  zero-moment / SCE-inactive sites bitwise frozen under a field
#   E  device gradient ≡ host lane reference (bitwise claim, GR9; scaled-tolerance
#      fallback reported, never silently accepted) and ≈ host energy_gradient!
#      with tangency — the Zeeman part is in G
#   F  Langevin law ⟨e·B̂⟩ = L(β μ_B m B) of a free moment sampled on the device
#   G  GPU-PT with a field: repeat bitwise, checkpoint/resume bitwise, :M_B
#      present, the field-free twin refused
#   I  (informational) whether five device sweeps still equal the host keyed
#      reference bitwise on this backend — expected on KA-CPU, a bonus on CUDA
# plus the with/without-field sweep cost on the bench fixtures. Any gate failure
# raises at the end. Every line flushes (batch logs survive a walltime kill).

using SCEMonteCarlo
using SCEFitting
using KernelAbstractions: KernelAbstractions, CPU
using LinearAlgebra
using Printf
using Random
using StaticArrays

include(joinpath(@__DIR__, "fixtures.jl"))

const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

backend = HAVE_CUDA ? CUDABackend() : CPU()
backend_name = HAVE_CUDA ? "CUDA ($(CUDA.name(CUDA.device())))" : "KA-CPU (smoke)"

n_bcc = argn(1, 8)
nsweeps = argn(2, 100)
const WS = 32

bench_header("GPU external field (Zeeman) validation — backend: $backend_name, " *
             "ws=$WS, $nsweeps sweeps/point")
flush(stdout)

# --- gate bookkeeping ------------------------------------------------------------
const FAILURES = String[]
function gate(name::AbstractString, ok::Bool; detail::AbstractString = "")
    println(ok ? "PASS " : "FAIL ", name, isempty(detail) ? "" : "   [" * detail * "]")
    ok || push!(FAILURES, name)
    flush(stdout)
    return ok
end
info(msg) = (println("INFO ", msg); flush(stdout))

# --- fixtures ----------------------------------------------------------------------
# Field and moments shared with test_zeeman.jl / test_gpu.jl.
const B = (0.3, -1.2, 2.0)                         # tesla
const ZMM_BIQ = [1.5, 0.7]                         # μ_B, biquadratic 2-atom cell

# The dimer: 4 Fe in a column, only atoms 1–2 coupled (isotropic l = 1 pair, ferro);
# atoms 3–4 are SCE-inactive — atom 3 carries a moment (Zeeman-only site), atom 4
# none (frozen under a field). Verbatim the unit suite's fixtures (test/unit/
# fixtures.jl), DEFAULT symmetry backend included: under Spglib the equally spaced
# column is one primitive chain and the first SALC couples every neighbour pair —
# no free atom is left, and the frozen-site and Langevin gates lose their subject.
function dimer_model()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], isotropy = true))
    return SCEPredictor(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))
end
# Two-atom cell, anisotropic l ≤ 2 pair basis (the unit suite's biquadratic fixture).
function biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                               isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

rand_spin(rng) = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
rand_cfg(rng, H) = MC.SpinConfig([rand_spin(rng) for _ = 1:H.n_sites])
langevin(x) = coth(x) - 1 / x

# The case list: a Zeeman-only site inside a fitted model, fitted + Zeeman sites,
# an all-body-1 model (zero-length factor tables on the device), and the bench
# fixtures with a field (realistic sizes).
cases = [("dimer (Zeeman-only site + frozen zero-moment site)",
          TiledHamiltonian(dimer_model(); dims = (2, 1, 1),
                           magmoms = [2.2, 2.2, 1.0, 0.0], field = B)),
         ("biquadratic + field",
          TiledHamiltonian(biquadratic_model(3); dims = (2, 2, 2),
                           magmoms = ZMM_BIQ, field = B)),
         ("all-body-1 (no fitted terms)",
          MC.TiledHamiltonian(2, MultipoleTerm[]; dims = (2, 2, 1),
                              magmoms = [1.0, 2.0], field = B)),
         ("bcc Fe $(n_bcc)³ + field",
          TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc),
                           magmoms = [2.2, 2.2], field = B)),
         ("Nd2Fe14B nbody=2 2³ + field (boron 0)",
          TiledHamiltonian(nd2fe14b_model(); dims = (2, 2, 2),
                           magmoms = nd2fe14b_magmoms(), field = B))]

# --- gates A–E, I per case -------------------------------------------------------
for (name, H) in cases
    println("\n--- ", name, ": ", describe(H))
    flush(stdout)
    β = 1 / 0.05
    rng = Xoshiro(7)
    st = MC.ChainState(H, rand_cfg(rng, H), rng, 0.6)
    gH = MC.GPUTiledHamiltonian(backend, H)
    HAVE_CUDA && CUDA.synchronize()

    # A: repeat-run identity (two device replicas from one initial state, one seed)
    g1 = MC.GPUChainState(gH, st; seed = UInt64(0xc0ffee))
    g2 = MC.GPUChainState(gH, st; seed = UInt64(0xc0ffee))
    for _ = 1:20
        MC.gpu_metropolis_sweep!(g1, gH, β; workgroupsize = WS)
        MC.gpu_metropolis_sweep!(g2, gH, β; workgroupsize = WS)
    end
    gate("A repeat-run identity — $name",
         Array(g1.config) == Array(g2.config) && g1.energy == g2.energy &&
         g1.acc_metro == g2.acc_metro)

    # I: (informational) device sweeps vs the host keyed reference, bitwise
    let st2 = MC.ChainState(H, copy(st.config), Xoshiro(0), st.step)
        gst = MC.GPUChainState(gH, st2; seed = UInt64(0xc0ffee))
        cfg2 = copy(st2.config)
        zr2 = copy(st2.zrows)
        dE2 = zeros(H.n_sites)
        acc2 = zeros(Int32, H.n_sites)
        E = gst.energy
        naccs = Int[]
        naccs_ref = Int[]
        for sw = 1:5
            push!(naccs, MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS))
            push!(naccs_ref, MC._metropolis_sweep_keyed_ref!(cfg2, zr2, dE2, acc2, H, β,
                                                             st2.step, gst.seed,
                                                             Int32(sw), WS))
            E += MC._reduce_dE(H, dE2)
        end
        MC.to_host!(st2, gst)
        same = st2.config == cfg2 && gst.energy == E && naccs == naccs_ref
        info("I device sweeps ≡ host keyed reference (bitwise): $same" *
             (same ? "" : "  (accepted: $(sum(naccs)) device / $(sum(naccs_ref)) host)"))
    end

    # B + D: drift after 200 sweeps with renormalization off; frozen sites
    gst = MC.GPUChainState(gH, st; seed = UInt64(0xbe11c0de))
    frozen = [s for s = 1:H.n_sites if !H.site_active[s]]
    cfg0 = copy(st.config)
    MC.gpu_run_sweeps!(gst, gH, st, β, 200; renorm_interval = 0)
    E = total_energy(H, st.config)
    gate("B incremental-energy drift — $name",
         abs(gst.energy - E) <= 1e-8 * max(1.0, abs(E));
         detail = @sprintf("|ΔE| = %.2e, E = %.6f eV", abs(gst.energy - E), E))
    gate("D inactive / zero-moment sites frozen ($(length(frozen)) sites) — $name",
         all(st.config[s] === cfg0[s] for s in frozen))

    # C: acceptance, device vs the tuned CPU path at the same kT
    acc_dev = gst.acc_metro / max(gst.att_metro, 1)
    let st3 = MC.ChainState(H, copy(cfg0), Xoshiro(11), 0.6)
        ntasks = min(4, Threads.nthreads())
        scs = [MC.SweepScratch(H) for _ = 1:ntasks]
        for _ = 1:200
            metropolis_sweep!(st3, H, β, ntasks == 1 ? scs[1] : scs)
        end
        acc_cpu = st3.acc_metro / max(st3.att_metro, 1)
        gate("C acceptance device ≈ cpu — $name", abs(acc_dev - acc_cpu) < 0.1;
             detail = @sprintf("device %.3f, cpu %.3f", acc_dev, acc_cpu))
    end

    # E: gradient — bitwise vs the lane reference (GR9), tolerance vs the host
    let config = rand_cfg(Xoshiro(23), H)
        zrows = MC._zrows(H, config)
        ref = Vector{SVector{3,Float64}}(undef, H.n_sites)
        MC._gradient_lane_ref!(ref, H, config, zrows, WS)
        gsc = MC.GPUGradientScratch(gH)
        dconfig = KernelAbstractions.allocate(backend, SVector{3,Float64}, H.n_sites)
        copyto!(dconfig, config)
        dG = KernelAbstractions.allocate(backend, SVector{3,Float64}, H.n_sites)
        MC.gpu_energy_gradient!(dG, gH, dconfig, gsc; workgroupsize = WS)
        G = Vector(dG)
        Ghost = MC.energy_gradient(H, config)
        scale = max(1.0, maximum(norm, Ghost))
        dev_ref = maximum(norm.(G .- ref))
        bitwise = G == ref
        gate("E gradient ≡ lane reference — $name",
             bitwise || dev_ref <= 1e-12 * scale;
             detail = bitwise ? "bitwise" :
                      @sprintf("NOT bitwise — scaled deviation %.2e (fallback)",
                               dev_ref / scale))
        gate("E gradient ≈ host energy_gradient! + tangency — $name",
             maximum(norm.(G .- Ghost)) <= 1e-12 * scale &&
             maximum(abs(dot(config[s], G[s])) / max(1.0, norm(G[s]))
                     for s = 1:H.n_sites) <= 1e-13;
             detail = @sprintf("max|G − G_host|/scale = %.2e",
                               maximum(norm.(G .- Ghost)) / scale))
        # inactive sites exactly zero
        gate("E inactive-site gradient exactly zero — $name",
             all(G[s] === SVector(0.0, 0.0, 0.0) for s in frozen))
    end
end

# --- gate F: Langevin law for a free moment on the device ------------------------
let
    println("\n--- F Langevin law")
    m3 = 2.0
    Bz = 20.0
    h = MU_B_EV_T * m3 * Bz
    H = TiledHamiltonian(dimer_model(); magmoms = [0.0, 0.0, m3, 0.0],
                         field = (0.0, 0.0, Bz))
    βh = 2.5
    β = βh / h
    rng = Xoshiro(7)
    st = MC.ChainState(H, rand_cfg(rng, H), rng, 1.2)
    gH = MC.GPUTiledHamiltonian(backend, H)
    gst = MC.GPUChainState(gH, st; seed = UInt64(2027))
    MC.gpu_run_sweeps!(gst, gH, st, β, 500; renorm_interval = 0)
    e4 = st.config[4]
    nmeas = 20_000
    samples = Vector{Float64}(undef, nmeas)
    for k = 1:nmeas
        MC.gpu_metropolis_sweep!(gst, gH, β)
        MC.to_host!(st, gst)
        samples[k] = st.config[3][3]
    end
    mean_z = sum(samples) / nmeas
    # 20-block standard error (the unit suite's tolerance was set from σ ≈ 0.006–0.008)
    nb = 20
    blk = [sum(@view samples[((b - 1) * (nmeas ÷ nb) + 1):(b * (nmeas ÷ nb))]) /
           (nmeas ÷ nb) for b = 1:nb]
    σ = sqrt(sum((blk .- mean_z) .^ 2) / (nb - 1) / nb)
    L = langevin(βh)
    gate("F Langevin ⟨e₃z⟩ = L(βμ_B m B)", abs(mean_z - L) <= 0.04;
         detail = @sprintf("⟨e₃z⟩ = %.4f, L = %.4f, dev = %.4f (%.1fσ, σ = %.4f)",
                           mean_z, L, mean_z - L, abs(mean_z - L) / σ, σ))
    gate("F zero-moment site bitwise frozen", st.config[4] === e4)
end

# --- gate G: GPU-PT with a field, checkpoint / resume ------------------------------
let
    println("\n--- G GPU-PT with a field")
    dir = mktempdir()
    H = TiledHamiltonian(biquadratic_model(0); dims = (2, 1, 1),
                         magmoms = ZMM_BIQ, field = B)
    gH = MC.GPUTiledHamiltonian(backend, H)
    kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
          exchange_interval = 7, measure_interval = 3, renorm_interval = 40,
          nbins = 8, seed = 17, workgroupsize = WS)
    path = joinpath(dir, "gpu_pt_field.jld2")
    a = gpu_run_pt(gH; kw...)
    b = gpu_run_pt(gH; kw..., checkpoint = path, checkpoint_interval = 120)
    gate("G PT repeat bitwise (plain vs checkpointing run)",
         a.final_configs == b.final_configs && a.swap_acceptance == b.swap_acceptance)
    gate("G :M_B observable present", haskey(a.points[1].stats, :M_B))
    phase, done = MC.jldopen(path, "r") do f
        (f["progress/phase"], f["progress/done"])
    end
    gate("G checkpoint is mid-run", phase == "measure" && 0 < done < 300;
         detail = "phase = $phase, done = $done")
    c = resume(path, gH)
    same = c isa PTResult && a.final_configs == c.final_configs &&
           a.swap_acceptance == c.swap_acceptance &&
           all(pa.stats[k].mean == pc.stats[k].mean && pa.stats[k].err == pc.stats[k].err
               for (pa, pc) in zip(a.points, c.points) for k in keys(pa.stats))
    gate("G resume ≡ uninterrupted (bitwise, stats included)", same)
    gH0 = MC.GPUTiledHamiltonian(backend, TiledHamiltonian(biquadratic_model(0);
                                                            dims = (2, 1, 1)))
    refused = try
        resume(path, gH0)
        false
    catch err
        err isa ErrorException
    end
    gate("G field-free twin refused on resume", refused)
    println("swap acceptances: ",
            join(map(x -> @sprintf("%.3f", x), a.swap_acceptance), " "))
    flush(stdout)
end

# --- sweep cost with / without a field --------------------------------------------
println("\n--- sweep cost (device ms/sweep, $nsweeps sweeps, ws=$WS)")
function dev_ms(H; β = 1 / BENCH_KT)
    gH = MC.GPUTiledHamiltonian(backend, H)
    rng = Xoshiro(7)
    st = MC.ChainState(H, MC._initial_config(H, nothing, rng), rng, 0.6)
    gst = MC.GPUChainState(gH, st; seed = UInt64(0xbe11c0de))
    for _ = 1:10
        MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS)
    end
    t = @elapsed for _ = 1:nsweeps
        MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS)
    end
    return 1e3 * t / nsweeps, gst.acc_metro / max(gst.att_metro, 1)
end
for (label, H0, H1) in (("bcc Fe $(n_bcc)³",
                         TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc)),
                         TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc),
                                          magmoms = [2.2, 2.2], field = (0.0, 0.0, 2.0))),
                        ("Nd2Fe14B nbody=2 4³",
                         TiledHamiltonian(nd2fe14b_model(); dims = (4, 4, 4)),
                         TiledHamiltonian(nd2fe14b_model(); dims = (4, 4, 4),
                                          magmoms = nd2fe14b_magmoms(),
                                          field = (0.0, 0.0, 2.0))))
    ms0, acc0 = dev_ms(H0)
    ms1, acc1 = dev_ms(H1)
    println(@sprintf("%-22s  no field %8.3f ms (acc %.2f)", label, ms0, acc0),
            @sprintf("   field %8.3f ms (acc %.2f)   %+.1f %%", ms1, acc1,
                     100 * (ms1 / ms0 - 1)))
    flush(stdout)
end

println()
if isempty(FAILURES)
    println("=== ALL GATES PASS on $backend_name ===")
else
    println("=== $(length(FAILURES)) GATE(S) FAILED on $backend_name ===")
    foreach(f -> println("  ", f), FAILURES)
    flush(stdout)
    error("external-field device validation FAILED")
end
flush(stdout)

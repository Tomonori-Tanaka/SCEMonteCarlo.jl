# Replica exchange over device chains — the on-device smoke + perf readout of
# docs/specs/gpu-prototype.md G8.
#
#   julia --project=bench/gpu bench/bench_gpu_pt.jl [n] [nsweeps_measure]
#
# Backend selection as bench_gpu.jl: CUDA when functional, else the KA CPU
# backend (correctness smoke only). Runs the device-portable G8 gates on the
# nbody=3 Nd₂Fe₁₄B fixture at n³ — (1) repeat-run bitwise identity of
# `gpu_run_pt`, (2) the exchange-free composition gate (ladder ≡ independent
# device chains, bitwise), (3) an exchanging ladder's sanity readout (swap
# acceptances, rung-energy ordering) — then reports the ladder's wall cost
# against R × the single-chain sweep cost (shared tables; expected ratio ≈ 1).
# Any gate failure raises. Every line flushes (batch logs survive a kill).

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

n = argn(1, 8)
nmeas = argn(2, 500)

bench_header("GPU parallel tempering (G8) — backend: $backend_name, " *
             "fixture Nd2Fe14B nbody=3 $(n)³, kT ladder around $(BENCH_KT) eV")
flush(stdout)

H = TiledHamiltonian(nd2fe14b3_model(); dims = (n, n, n))
println(describe(H))
t_up = @elapsed begin
    gH = GPUTiledHamiltonian(backend, H)
    HAVE_CUDA && CUDA.synchronize()
end
@printf("tables uploaded once in %.2f s (shared by every rung)\n", t_up)
flush(stdout)

kts = BENCH_KT .* [2.0, 1.4, 1.0, 0.7]          # descending — hot rung first
R = length(kts)

# --- gate 1 + 3: an exchanging ladder, twice — repeat bitwise + sanity ------------
kw = (; kT = kts, sweeps_therm = 200, sweeps_measure = nmeas,
      measure_interval = 5, exchange_interval = 10, nbins = 8,
      seed = UInt64(0x5ce9))
t_pt = @elapsed r1 = gpu_run_pt(gH; kw...)
r2 = gpu_run_pt(gH; kw...)
show(stdout, MIME("text/plain"), r1)
println()

repeat_ok = r1.final_configs == r2.final_configs &&
            r1.swap_acceptance == r2.swap_acceptance &&
            all(pa.stats[:energy].mean == pb.stats[:energy].mean &&
                pa.stats[:energy].err == pb.stats[:energy].err
                for (pa, pb) in zip(r1.points, r2.points))
println("repeat-run bitwise identity: ", repeat_ok)
Es = [p.stats[:energy].mean[1] for p in r1.points]
order_ok = issorted(Es; rev = true)             # hotter rung ⇒ higher energy
println("rung energies ordered (descending kT ⇒ descending E): ", order_ok)
swaps_ok = all(a -> 0 <= a <= 1, r1.swap_acceptance)
@printf("swap acceptances: %s\n",
        join([@sprintf("%.2f", a) for a in r1.swap_acceptance], " "))
flush(stdout)

# --- gate 2: exchange-free composition (test_gpu_pt.jl's gate, on this backend) ---
# With no interior exchange boundary and adaptation / periodic renormalization
# pushed out of range, each rung must be EXACTLY an independent keyed device
# chain driven through the public sweep API, plus the one boundary
# renormalization — pins the lane bookkeeping and the G8 seed-derivation order.
function composition_gate(gH, H; nt::Int = 40, nm::Int = 60)
    ckts = kts[1:3]
    seed = UInt64(77)
    r = gpu_run_pt(gH; kT = ckts, sweeps_therm = nt, sweeps_measure = nm,
                   exchange_interval = nt + nm, adapt_interval = nt + nm,
                   renorm_interval = 10 * (nt + nm), nbins = 4, seed = seed)
    master = Xoshiro(seed)
    lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                         rand(master, UInt64), rand(master, UInt64))
                 for _ in ckts]
    Xoshiro(rand(master, UInt64), rand(master, UInt64),
            rand(master, UInt64), rand(master, UInt64))    # exchange-rng slot
    dev_seeds = [rand(master, UInt64) for _ in ckts]
    ok = true
    for (i, kt) in enumerate(ckts)
        st = MC.ChainState(H, MC._initial_config(H, nothing, lane_rngs[i]),
                           lane_rngs[i], 0.6)
        gst = GPUChainState(gH, st; seed = dev_seeds[i])
        β = 1 / kt
        for _ = 1:nt
            gpu_metropolis_sweep!(gst, gH, β)
        end
        to_host!(st, gst)
        MC._renormalize!(st, H)
        MC._from_host!(gst, st)
        for _ = 1:nm
            gpu_metropolis_sweep!(gst, gH, β)
        end
        to_host!(st, gst)
        ok &= r.final_configs[i] == st.config
    end
    return ok
end
comp_ok = composition_gate(gH, H)
println("exchange-free ladder ≡ independent device chains (bitwise): ", comp_ok)
flush(stdout)

(repeat_ok && order_ok && swaps_ok && comp_ok) ||
    error("G8 device gate FAILED — see the lines above")

# --- perf: ladder cost vs R × single-chain cost -----------------------------------
ns = 200
t_ladder = @elapsed gpu_run_pt(gH; kT = kts, sweeps_therm = 0,
                               sweeps_measure = ns, measure_interval = ns,
                               exchange_interval = 10, nbins = 2,
                               seed = UInt64(1))
st0, _ = chain_state(H)
gst0 = GPUChainState(gH, st0; seed = UInt64(2))
for _ = 1:10                                    # warmup
    gpu_metropolis_sweep!(gst0, gH, 1 / BENCH_KT)
end
t_single = @elapsed for _ = 1:ns
    gpu_metropolis_sweep!(gst0, gH, 1 / BENCH_KT)
end
@printf("ladder: %d rungs × %d sweeps in %.2f s (%.2f ms/sweep/rung)\n", R, ns,
        t_ladder, 1e3 * t_ladder / (R * ns))
@printf("single chain: %.2f ms/sweep → ladder/(R × single) = %.2f\n",
        1e3 * t_single / ns, t_ladder / (R * t_single))
@printf("full gated run above (R = %d, %d + %d sweeps, measure every 5): %.1f s\n",
        R, 200, nmeas, t_pt)
println("\n=== G8 device gates: ALL PASS (backend: $backend_name) ===")
flush(stdout)

# GPU parallel tempering (decision record docs/specs/gpu-prototype.md G8), all on
# the KernelAbstractions CPU backend: the device payload partition, the
# exchange-free composition gate (ladder ≡ independent device chains, bitwise),
# determinism, exchange-rate limits, and the fixed-temperature marginal oracle.

using KernelAbstractions: CPU

@testset "gpu parallel tempering" begin
    Hd = TiledHamiltonian(_dimer_model())
    J = _dimer_J(Hd)
    gHd = MC.GPUTiledHamiltonian(CPU(), Hd)

    @testset "marginals match the closed form (exact dimer, 3-rung ladder)" begin
        # Independent oracle: ⟨e₁·e₂⟩ = L(β|J|) for the classical Heisenberg
        # dimer, at every rung of a ladder that IS exchanging. R = 3 makes the
        # pair schedule itself load-bearing (pairs (1,2) and (2,3) attempt on
        # alternating parities): a mispaired or misattributed swap contaminates
        # the middle rung's marginal, which no 2-rung gate can see. The sharp
        # detector for the accept rule's SIGN is the exchange-rate-limits gate
        # below — the dimer decorrelates between exchanges, so this gate alone
        # would largely launder a flipped sign.
        kts = abs(J) .* [0.3, 0.6, 1.2]              # βJ ≈ 3.33, 1.67, 0.83
        obs = vcat(standard_observables(Hd), _corr12_obs())
        r = gpu_run_pt(gHd; kT = kts, sweeps_therm = 500, sweeps_measure = 20_000,
                       measure_interval = 5, exchange_interval = 10, seed = 31,
                       observables = obs)
        @test [p.kT for p in r.points] == kts
        for (p, kt) in zip(r.points, kts)
            exact = _langevin(abs(J) / kt)
            # measured-σ gate with 5σ headroom (binning err ≈ 0.006–0.011 for
            # 4000 correlated samples here), plus the family's proven absolute
            # backstop from the CPU PT gate
            @test abs(p.stats[:corr12].mean[1] - exact) <
                  5 * p.stats[:corr12].err[1]
            @test p.stats[:corr12].mean[1] ≈ exact atol = 0.04
        end
        @test length(r.swap_acceptance) == 2
        @test all(a -> 0 < a <= 1, r.swap_acceptance)
        @test length(r.final_configs) == 3
        # no overrelaxation on the device path
        @test all(isnan(p.acceptance_or) for p in r.points)
    end

    @testset "exchange-rate limits" begin
        # (near-)degenerate ladder → swaps ~always accepted
        r_eq = gpu_run_pt(gHd; kT = [0.049999, 0.05], sweeps_therm = 100,
                          sweeps_measure = 400, exchange_interval = 5, seed = 1)
        @test r_eq.swap_acceptance[1] > 0.95
        # Huge β gap on a coupled supercell → swaps rare, energies ordered.
        # This gate OWNS the accept rule's sign: |logw| ~ O(10²) here, so a
        # flipped sign in `_swap_accepts` drives the acceptance to ~1 and fails
        # the < 0.2 bound decisively (the marginal gate above cannot).
        H = TiledHamiltonian(1, _chain_terms(-0.05); dims = (4, 4, 2))
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        r_far = gpu_run_pt(gH; kT = [0.002, 0.2], sweeps_therm = 200,
                           sweeps_measure = 1_000, exchange_interval = 5, seed = 2)
        @test r_far.swap_acceptance[1] < 0.2
        Es = [p.stats[:energy].mean[1] for p in r_far.points]
        @test issorted(Es)
    end

    @testset "exchange-free ladder ≡ independent device chains (bitwise)" begin
        # Composition gate: with no exchange boundary inside either phase, and
        # adaptation/periodic renormalization pushed out of range, each rung must
        # be EXACTLY an independent keyed device chain driven through the public
        # sweep API — plus the one boundary renormalization. This pins the lane
        # bookkeeping (no cross-rung contamination) and the G8 seed-derivation
        # order, which is a documented contract.
        H = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        kts = [0.5, 0.2, 0.1]
        seed = UInt64(77)
        nt, nm = 40, 60
        r = gpu_run_pt(gH; kT = kts, sweeps_therm = nt, sweeps_measure = nm,
                       exchange_interval = nt + nm,       # no interior boundary
                       adapt_interval = nt + nm,          # adaptation never fires
                       renorm_interval = 10 * (nt + nm),  # periodic renorm neither
                       nbins = 4, seed = seed)
        @test all(isnan, r.swap_acceptance)               # nothing was attempted

        # the documented derivation: lane rngs (4 words each), exchange rng
        # (4 words, unused here), then one device seed per rung
        master = Xoshiro(seed)
        lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                             rand(master, UInt64), rand(master, UInt64))
                     for _ in kts]
        Xoshiro(rand(master, UInt64), rand(master, UInt64),
                rand(master, UInt64), rand(master, UInt64))
        dev_seeds = [rand(master, UInt64) for _ in kts]
        for (i, kt) in enumerate(kts)
            st = MC.ChainState(H, MC._initial_config(H, nothing, lane_rngs[i]),
                               lane_rngs[i], 0.6)
            gst = MC.GPUChainState(gH, st; seed = dev_seeds[i])
            β = 1 / kt
            for _ = 1:nt
                MC.gpu_metropolis_sweep!(gst, gH, β)
            end
            MC.to_host!(st, gst)                          # boundary renorm
            MC._renormalize!(st, H)
            MC._from_host!(gst, st)
            for _ = 1:nm
                MC.gpu_metropolis_sweep!(gst, gH, β)
            end
            MC.to_host!(st, gst)
            @test r.final_configs[i] == st.config
        end
    end

    @testset "determinism: repeat bitwise, seeds differ" begin
        H = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        kw = (; kT = [0.5, 0.3, 0.2, 0.1], sweeps_therm = 200,
              sweeps_measure = 600, exchange_interval = 7, renorm_interval = 100,
              nbins = 8, seed = 5)
        a = gpu_run_pt(gH; kw...)
        b = gpu_run_pt(gH; kw...)
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        for (pa, pb) in zip(a.points, b.points)
            @test pa.stats[:energy].mean == pb.stats[:energy].mean
            @test pa.stats[:energy].err == pb.stats[:energy].err
            @test pa.stats[:m].mean == pb.stats[:m].mean
            @test pa.acceptance_metropolis == pb.acceptance_metropolis
            @test pa.final_step == pb.final_step
        end
        # R = 4 physics on the exchanging ladder: descending kT ⇒ descending
        # mean energies (a pair-schedule error that only appears for R ≥ 3
        # scrambles this ordering; the bitwise a == b comparison cannot see it)
        Es = [p.stats[:energy].mean[1] for p in a.points]
        @test issorted(Es; rev = true)
        # a different seed genuinely differs
        c = gpu_run_pt(gH; kw..., seed = 6)
        @test c.final_configs != a.final_configs

        # the default seed is drawn fresh per call and recorded in the result
        kwd = (; kT = [0.5, 0.2], sweeps_therm = 50, sweeps_measure = 100,
               nbins = 4)
        d1 = gpu_run_pt(gH; kwd...)
        d2 = gpu_run_pt(gH; kwd...)
        @test d1.seed != d2.seed
        @test gpu_run_pt(gH; kwd..., seed = d1.seed).final_configs ==
              d1.final_configs
    end

    @testset "guards, printing, and the backend-convenience form" begin
        @test_throws ArgumentError gpu_run_pt(gHd; kT = [0.05])
        @test_throws ArgumentError gpu_run_pt(gHd; kT = [0.05, 0.02, 0.03])
        @test_throws ArgumentError gpu_run_pt(gHd; kT = [0.05, 0.02],
                                              exchange_interval = 0)
        @test_throws ArgumentError gpu_run_pt(gHd; kT = [0.05, 0.02],
                                              workgroupsize = 3)
        r = gpu_run_pt(gHd; kT = [0.05, 0.02], sweeps_therm = 20,
                       sweeps_measure = 50, nbins = 4, seed = 1)
        @test occursin("PTResult", sprint(show, r))
        # the convenience form uploads and runs the same deterministic driver
        r2 = gpu_run_pt(CPU(), Hd; kT = [0.05, 0.02], sweeps_therm = 20,
                        sweeps_measure = 50, nbins = 4, seed = 1)
        @test r2.final_configs == r.final_configs
    end

    # What a replica exchange moves between device lanes, over EVERY field of
    # `GPUChainState` rather than a hand-picked sample — the device sibling of
    # the exhaustive `ChainState` partition in test_pt.jl. A swap moves the
    # physical state (device config / rows references + the incremental energy);
    # the keyed-RNG bookkeeping (seed, sweep_index), the step, the counters, and
    # the per-sweep staging buffers stay with the lane. The union requirement
    # forces any new field to be classified rather than forgotten.
    @testset "replica exchange moves the device physical state and nothing else" begin
        H = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        payload = (:config, :zrows, :energy)

        # Every mutable scalar gets a value unique to its chain, so `===` can
        # tell a field that swapped from one that did not; the arrays are
        # distinct objects already.
        function mkgst(k)
            st = MC.ChainState(H, _rand_config(MersenneTwister(50 + k), H),
                               Xoshiro(50 + k), 0.3)
            gst = MC.GPUChainState(gH, st; seed = UInt64(1000k))
            gst.energy = 100.0k
            gst.sweep_index = 10k
            gst.step = 0.1k
            gst.acc_metro, gst.att_metro = 1k, 2k
            return gst
        end
        a, b = mkgst(1), mkgst(2)
        fields = fieldnames(typeof(a))
        before_a = Dict(f => getfield(a, f) for f in fields)
        before_b = Dict(f => getfield(b, f) for f in fields)
        # every field really is distinguishable, or the partition is vacuous
        @test all(f -> before_a[f] !== before_b[f], fields)

        MC._swap_payload!(a, b)
        moved = [f for f in fields if getfield(a, f) === before_b[f]]
        stayed = [f for f in fields if getfield(a, f) === before_a[f]]

        @test Set(moved) == Set(payload)
        @test Set(moved) ∪ Set(stayed) == Set(fields)      # classified
        @test isempty(Set(moved) ∩ Set(stayed))
        # and the other chain received the mirror image, not a copy of its own
        @test all(f -> getfield(b, f) === before_a[f], payload)
        @test all(f -> getfield(b, f) === before_b[f],
                  [f for f in fields if !(f in payload)])
    end
end

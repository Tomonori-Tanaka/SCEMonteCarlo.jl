# Parallel tempering: fixed-temperature marginals against independent run_mc,
# exchange sanity limits, and the bit-determinism gate (ntasks-independence).

@testset "parallel tempering" begin
    Hd = TiledHamiltonian(_dimer_model())
    J = _dimer_J(Hd)

    @testset "marginals match independent single-T runs (exact dimer)" begin
        kts = abs(J) .* [0.4, 1.0]                       # βJ = 2.5 and 1.0
        obs = vcat(standard_observables(Hd), _corr12_obs())
        r = run_pt(Hd; kT = kts, sweeps_therm = 500, sweeps_measure = 20_000,
                   measure_interval = 5, exchange_interval = 10, seed = 31,
                   observables = obs)
        @test [p.kT for p in r.points] == kts
        for (p, kt) in zip(r.points, kts)
            @test p.stats[:corr12].mean[1] ≈ _langevin(abs(J) / kt) atol = 0.04
        end
        @test length(r.swap_acceptance) == 1
        @test 0 < r.swap_acceptance[1] <= 1
        @test length(r.final_configs) == 2
    end

    @testset "exchange-rate limits" begin
        # (near-)degenerate ladder → swaps ~always accepted
        r_eq = run_pt(Hd; kT = [0.049999, 0.05], sweeps_therm = 100,
                      sweeps_measure = 400, exchange_interval = 5, seed = 1)
        @test r_eq.swap_acceptance[1] > 0.95
        # huge β gap on a coupled supercell → swaps rare
        H = TiledHamiltonian(1, _chain_terms(-0.05); dims = (4, 4, 2))
        r_far = run_pt(H; kT = [0.002, 0.2], sweeps_therm = 200,
                       sweeps_measure = 1_000, exchange_interval = 5, seed = 2)
        @test r_far.swap_acceptance[1] < 0.2
    end

    @testset "determinism: ntasks-independent bit-identical results" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        kw = (; kT = [0.5, 0.3, 0.2, 0.1], sweeps_therm = 200,
              sweeps_measure = 600, exchange_interval = 7, nbins = 8, seed = 5)
        a = run_pt(H; kw..., ntasks = 1)
        b = run_pt(H; kw..., ntasks = 4)
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        for (pa, pb) in zip(a.points, b.points)
            @test pa.stats[:energy].mean == pb.stats[:energy].mean
            @test pa.stats[:energy].err == pb.stats[:energy].err
            @test pa.stats[:m].mean == pb.stats[:m].mean
            @test pa.acceptance_metropolis == pb.acceptance_metropolis
            @test pa.final_step == pb.final_step
        end
        # and a different seed genuinely differs
        c = run_pt(H; kT = [0.5, 0.3, 0.2, 0.1], sweeps_therm = 200,
                   sweeps_measure = 600, exchange_interval = 7, nbins = 8,
                   seed = 6)
        @test c.final_configs != a.final_configs

        # the default seed is drawn fresh per call and recorded in the result
        kwd = (; kT = [0.5, 0.2], sweeps_therm = 50, sweeps_measure = 100,
               nbins = 4)
        d1 = run_pt(H; kwd...)
        d2 = run_pt(H; kwd...)
        @test d1.seed != d2.seed
        @test run_pt(H; kwd..., seed = d1.seed).final_configs == d1.final_configs

        # boundary-free phases (a phase shorter than one exchange segment) and a
        # maximal boundary rate (exchange_interval = 1) through the async schedule
        for ei in (1, 1_000)
            kwe = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 30, sweeps_measure = 60,
                   exchange_interval = ei, nbins = 4, seed = UInt64(9))
            e1 = run_pt(H; kwe..., ntasks = 1)
            e3 = run_pt(H; kwe..., ntasks = 3)
            @test e1.final_configs == e3.final_configs
            # isequal: an exchange-free phase leaves NaN acceptances (ei = 1000)
            @test isequal(e1.swap_acceptance, e3.swap_acceptance)
        end
    end

    @testset "async blocks with mid-phase checkpoints: bit-identity + resume" begin
        # forces ntasks ≥ 2 (async at any thread count) with periodic writes that
        # land inside both phases, so `_pt_block_sweeps` runs multi-block with
        # trailing exchange boundaries — the production configuration
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        dir = mktempdir()
        path = joinpath(dir, "pt_async.jld2")
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 60, sweeps_measure = 120,
              exchange_interval = 7, nbins = 4, seed = UInt64(21))
        a = run_pt(H; kw..., ntasks = 1)
        b = run_pt(H; kw..., ntasks = 3, checkpoint = path,
                   checkpoint_interval = 25)
        @test a.final_configs == b.final_configs
        @test isequal(a.swap_acceptance, b.swap_acceptance)
        for (pa, pb) in zip(a.points, b.points)
            @test pa.stats[:energy].mean == pb.stats[:energy].mean
            @test pa.stats[:energy].err == pb.stats[:energy].err
        end
        # the file's last periodic write is mid-measure; resume replays the tail
        c = resume(path, H)
        @test a.final_configs == c.final_configs
        @test isequal(a.swap_acceptance, c.swap_acceptance)
    end

    @testset "PT rescues the frozen anisotropic fixture" begin
        # At kT = 0.03 two independent chains freeze into different basins
        # (docs/specs/updates-stationarity.md U6); a ladder to hot temperatures
        # lets the cold rung tunnel between them. Weak gate: the cold rung's
        # energy is at least as low as any independent run's basin.
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        ladder = [0.03, 0.09, 0.2, 0.45]
        r = run_pt(H; kT = ladder, sweeps_therm = 2_000, sweeps_measure = 4_000,
                   exchange_interval = 5, seed = 8)
        E_cold_pt = r.points[1].stats[:energy].mean[1]
        E_indep = [run_mc(H; kT = 0.03, sweeps_therm = 2_000,
                          sweeps_measure = 4_000,
                          seed = s).points[1].stats[:energy].mean[1]
                   for s in (21, 22)]
        @test E_cold_pt <= maximum(E_indep) + 0.05
        # monotone ladder sanity: hotter rungs have higher energy
        Es = [p.stats[:energy].mean[1] for p in r.points]
        @test issorted(Es)
    end

    @testset "guards and printing" begin
        @test_throws ArgumentError run_pt(Hd; kT = [0.05])
        @test_throws ArgumentError run_pt(Hd; kT = [0.05, 0.02, 0.03])
        @test_throws ArgumentError run_pt(Hd; kT = [0.05, 0.02],
                                          exchange_interval = 0)
        @test_throws ArgumentError run_pt(Hd; kT = [0.05, 0.02], ntasks = 0)
        r = run_pt(Hd; kT = [0.05, 0.02], sweeps_therm = 20, sweeps_measure = 50,
                   nbins = 4, seed = 1)
        @test occursin("PTResult", sprint(show, r))
        long = sprint(show, MIME("text/plain"), r)
        @test occursin("swap acceptance", long)
    end

    # What a replica exchange moves between lanes, over EVERY field of `ChainState`
    # rather than a hand-picked sample. A swap moves a whole physical state — the
    # configuration, the cached rows and the energy — while the RNG streams, the
    # proposal width, the acceptance counters and the drift diagnostic stay with the
    # lane, because a lane is a fixed temperature and those describe it.
    #
    # An exhaustive partition is the point. The existing PT gates cannot see a payload
    # field's swap being dropped: `test_checkpoint.jl`'s resume gates compare a resumed
    # ladder against an uninterrupted one and BOTH sides run the same swap rule, so any
    # rule passes them, and the statistical rung marginals are too coarse to resolve
    # it. This asserts the two sets by name and requires their union to be every field,
    # so a new field must be classified rather than forgotten.
    # [Backported from SLCEMonteCarlo.jl 2b252da, re-partitioned over the spin-era
    # `ChainState` fields and placed here (upstream's version lives in its joint
    # suite).]
    @testset "replica exchange moves the physical state and nothing else" begin
        H = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        payload = (:config, :zrows, :energy)

        # Every mutable scalar gets a value unique to its chain, so `===` can tell a
        # field that swapped from one that did not; the arrays and RNG objects are
        # distinct objects already.
        function distinct!(st, k)
            st.energy = 100.0k
            st.step = 0.1k
            st.frozen = isodd(k)
            st.acc_metro, st.att_metro = 1k, 2k
            st.acc_or, st.att_or = 3k, 4k
            st.max_drift = 0.001k
            return st
        end

        mkst(k) = distinct!(MC.ChainState(H, _rand_config(MersenneTwister(40 + k), H),
                                          Xoshiro(40 + k), 0.3), k)
        a, b = mkst(1), mkst(2)
        before_a = Dict(f => getfield(a, f) for f in fieldnames(MC.ChainState))
        before_b = Dict(f => getfield(b, f) for f in fieldnames(MC.ChainState))
        # every field really is distinguishable, or the partition below is vacuous
        @test all(f -> before_a[f] !== before_b[f], fieldnames(MC.ChainState))

        MC._swap_payload!(a, b)
        moved = [f for f in fieldnames(MC.ChainState) if getfield(a, f) === before_b[f]]
        stayed = [f for f in fieldnames(MC.ChainState) if getfield(a, f) === before_a[f]]

        @test Set(moved) == Set(payload)
        @test Set(moved) ∪ Set(stayed) == Set(fieldnames(MC.ChainState))  # classified
        @test isempty(Set(moved) ∩ Set(stayed))
        # and the other chain received the mirror image, not a copy of its own state
        @test all(f -> getfield(b, f) === before_a[f], payload)
        @test all(f -> getfield(b, f) === before_b[f], stayed)
    end
end

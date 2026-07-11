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
end

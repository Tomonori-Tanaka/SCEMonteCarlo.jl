# Overrelaxation: the involutive reflection, the pure-l=1 exact microcanonical limit
# (which also pins the tesseral l=1 axis extraction), and Boltzmann stationarity of
# the Metropolis+OR mix against exact / pure-Metropolis references.

@testset "overrelaxation" begin
    @testset "reflection is an involution" begin
        rng = MersenneTwister(2)
        for _ = 1:5
            e = _rand_spin(rng)
            axis = _rand_spin(rng)
            S(x) = 2 * dot(x, axis) * axis - x
            @test S(S(e)) ≈ e atol = 1e-15
            @test norm(S(e)) ≈ 1.0 atol = 1e-15
        end
    end

    @testset "pure l=1: ΔE ≡ 0, always accepted, energy invariant" begin
        H = TiledHamiltonian(_dimer_model())            # l = 1 only
        rng = Xoshiro(9)
        st = MC.ChainState(H, MC._initial_config(H, nothing, rng), rng, 0.6)
        sc = MC.SweepScratch(H)
        E0 = st.energy
        e1_init = st.config[1]
        for _ = 1:20
            MC.overrelaxation_sweep!(st, H, 1 / 0.01, sc)
        end
        # only the coupled sites 1, 2 have an l=1 channel (free spins skipped)
        @test st.att_or == 20 * 2
        @test st.acc_or == st.att_or                     # every reflection accepted
        @test st.energy ≈ E0 atol = 1e-12                # microcanonical
        @test total_energy(H, st.config) ≈ E0 atol = 1e-12
        # and the chain actually moved
        @test norm(st.config[1] - e1_init) > 1e-6
    end

    @testset "Metropolis + OR mix reproduces the exact dimer curve" begin
        Hd = TiledHamiltonian(_dimer_model())
        J = _dimer_J(Hd)
        for (i, βJ) in enumerate([1.0, 2.5])
            obs = vcat(standard_observables(Hd), _corr12_obs())
            r = run_mc(Hd; kT = abs(J) / βJ, sweeps_therm = 500,
                       sweeps_measure = 10_000, measure_interval = 5,
                       or_per_metropolis = 3, seed = 300 + i, observables = obs)
            p = r.points[1]
            @test p.stats[:corr12].mean[1] ≈ _langevin(βJ) atol = 0.03
            @test p.acceptance_or ≈ 1.0 atol = 1e-12     # pure l=1 model
        end
    end

    @testset "multipole model: OR mix agrees with pure Metropolis" begin
        # kT = 0.5 — hot enough that the random anisotropic fixture equilibrates
        # (at kT ≲ 0.15 it freezes into seed-dependent basins and even two pure
        # Metropolis chains disagree far beyond their error bars).
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        base = (; kT = 0.5, sweeps_therm = 1_000, sweeps_measure = 8_000)
        a = run_mc(H; base..., seed = 21)
        b = run_mc(H; base..., or_per_metropolis = 2, seed = 22)
        Ea, Eb = a.points[1].stats[:energy], b.points[1].stats[:energy]
        tol = 4 * sqrt(Ea.err[1]^2 + Eb.err[1]^2)
        @test abs(Ea.mean[1] - Eb.mean[1]) < tol
        # the l≥2 remainder makes reflections cost energy: the accept step is
        # genuinely exercised, and some moves are rejected
        @test 0.1 < b.points[1].acceptance_or < 0.9
    end

    @testset "acceptance_or bookkeeping" begin
        H = TiledHamiltonian(_dimer_model())
        r0 = run_mc(H; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100, seed = 1)
        @test isnan(r0.points[1].acceptance_or)          # no OR sweeps requested
        r1 = run_mc(H; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100,
                    or_per_metropolis = 1, seed = 1)
        @test 0 < r1.points[1].acceptance_or <= 1
        @test_throws ArgumentError run_mc(H; kT = 0.05, or_per_metropolis = -1)
    end
end

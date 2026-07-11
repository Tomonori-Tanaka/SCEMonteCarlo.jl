# Checkpoint / resume: bit-identity gates (==, never ≈) for MC and PT, Xoshiro
# round-trips, and the schema guards.

# Everything result-shaped must be bit-equal between two runs.
function _assert_same_result(a, b)
    @test length(a.points) == length(b.points)
    for (pa, pb) in zip(a.points, b.points)
        @test pa.kT == pb.kT
        @test sort(collect(keys(pa.stats))) == sort(collect(keys(pb.stats)))
        for k in keys(pa.stats)
            @test pa.stats[k].mean == pb.stats[k].mean
            @test pa.stats[k].err == pb.stats[k].err
            @test isequal(pa.stats[k].tau_int, pb.stats[k].tau_int)
            @test pa.stats[k].count == pb.stats[k].count
        end
        @test pa.acceptance_metropolis == pb.acceptance_metropolis
        @test isequal(pa.acceptance_or, pb.acceptance_or)
        @test pa.final_step == pb.final_step
        @test pa.max_drift == pb.max_drift
    end
end

@testset "checkpoint / resume" begin
    dir = mktempdir()
    H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))

    @testset "Xoshiro word round-trip" begin
        rng = Xoshiro(1234)
        rand(rng, 17)
        words = MC._rng_words(rng)
        @test length(words) == fieldcount(Xoshiro)
        rng2 = MC._rng_from_words(words)
        @test all(rand(rng, UInt64) == rand(rng2, UInt64) for _ = 1:100)
        @test_throws ErrorException MC._rng_from_words(UInt64[1, 2])
    end

    @testset "MC: checkpointing does not perturb, resume is bit-identical" begin
        kw = (; kT = [0.5, 0.3], sweeps_therm = 200, sweeps_measure = 400,
              measure_interval = 2, nbins = 8, renorm_interval = 100, seed = 42)
        path = joinpath(dir, "mc.jld2")
        a = run_mc(H; kw...)                                # no checkpointing
        b = run_mc(H; kw..., checkpoint = path, checkpoint_interval = 150)
        _assert_same_result(a, b)                           # writing consumes no RNG
        @test a.final_config == b.final_config
        # the file's last periodic write is mid-run; resume must reproduce the
        # full run bit-exactly (last tick: temp 2, measure phase, sweep 250)
        @test isfile(path)
        c = resume(path, H)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
        @test c isa MCResult
    end

    @testset "MC: resume from a thermalization-phase checkpoint" begin
        # interval chosen so the last periodic write lands inside temp-2 therm
        kw = (; kT = [0.5, 0.3], sweeps_therm = 300, sweeps_measure = 100,
              measure_interval = 1, nbins = 8, seed = 7)
        path = joinpath(dir, "mc_therm.jld2")
        a = run_mc(H; kw...)
        run_mc(H; kw..., checkpoint = path, checkpoint_interval = 260)
        # ticks at global sweeps 260 (temp1 therm? 260 < 300 → therm sweep 260),
        # 520 (temp2 therm 120), 780 (temp2 measure 80 → but measure only 100)…
        c = resume(path, H)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "MC: boundary-only checkpoints (interval 0) and carryover=false" begin
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 100, sweeps_measure = 100,
              nbins = 4, carryover = false, seed = 3)
        path = joinpath(dir, "mc_boundary.jld2")
        a = run_mc(H; kw...)
        run_mc(H; kw..., checkpoint = path)                 # interval 0
        c = resume(path, H)                                 # from the last boundary
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "PT: resume is bit-identical" begin
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
              exchange_interval = 7, nbins = 8, seed = 11)
        path = joinpath(dir, "pt.jld2")
        a = run_pt(H; kw...)
        b = run_pt(H; kw..., checkpoint = path, checkpoint_interval = 120)
        _assert_same_result(a, b)
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        c = resume(path, H)
        @test c isa PTResult
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
        @test a.swap_acceptance == c.swap_acceptance
    end

    @testset "PT: resume from the phase-boundary checkpoint (interval 0)" begin
        kw = (; kT = [0.5, 0.2], sweeps_therm = 100, sweeps_measure = 200,
              exchange_interval = 9, nbins = 4, seed = 13)
        path = joinpath(dir, "pt_boundary.jld2")
        a = run_pt(H; kw...)
        run_pt(H; kw..., checkpoint = path)
        c = resume(path, H)
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
    end

    @testset "schema and mismatch guards" begin
        path = joinpath(dir, "guard.jld2")
        run_mc(H; kT = 0.5, sweeps_therm = 50, sweeps_measure = 60, nbins = 4,
               seed = 1, checkpoint = path, checkpoint_interval = 40)
        # fingerprint mismatch: different dims
        H2 = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        @test_throws ErrorException resume(path, H2)
        # observable mismatch
        @test_throws ErrorException resume(path, H; observables = [
            Observable(:energy, 1, (c, E, h) -> E)])
        # missing file
        @test_throws ArgumentError resume(joinpath(dir, "nope.jld2"), H)
        # negative interval guard
        @test_throws ArgumentError run_mc(H; kT = 0.5, checkpoint = path,
                                          checkpoint_interval = -1)
    end
end

# Constrained MC: the joint two-spin energy difference of the pair move
# (`_pair_delta_energy!`, sequential leave-one-out — docs/specs/constrained-mc.md
# D2). Gate G2. Oracle: `total_energy` of the configuration with both spins replaced,
# minus that of the original — a from-scratch contraction that shares no
# leave-one-out code with the kernel under test.

# Sum of two INDEPENDENT single-site differences, both against the unmodified rows
# — the naive ΔE the sequential form exists to correct. Exact only when the two
# sites share no instance.
function _en_naive_pair_dE(H, zrows, i, j, e_i, e_j)
    c = zeros(H.nlm)
    z = zeros(H.nlm)
    MC.site_coeffs!(c, H, i, zrows)
    MC._zlm_row!(z, e_i, H.lmax)
    ΔE_i = MC.delta_energy(c, view(zrows, :, i), z)
    fill!(c, 0.0)
    MC.site_coeffs!(c, H, j, zrows)
    MC._zlm_row!(z, e_j, H.lmax)
    return ΔE_i + MC.delta_energy(c, view(zrows, :, j), z)
end

# Allocation probe behind a function barrier (global-scope calls box their args).
function _en_pair_dE_allocs(scratch, zrows, H, i, j, e_i, e_j)
    MC._pair_delta_energy!(scratch, zrows, H, i, j, e_i, e_j)
    return @allocated MC._pair_delta_energy!(scratch, zrows, H, i, j, e_i, e_j)
end

@testset "CMC pair ΔE" begin
    rng = MersenneTwister(20261121)

    # (name, H, pairs, number of instances each pair shares). The counts are
    # asserted from the instance CSR first: a "sharing" case that shared nothing
    # would pass the naive sum and prove nothing.
    cases = [
        ("2-body instance (F2g)", MC.TiledHamiltonian(2, _f2g_terms()),
         [(1, 2)], 1),
        ("3-body instance (chain)",
         MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (7, 1, 1)),
         [(1, 3), (2, 4)], 1),
        ("4-body end pair (chain, slots 1 and 4)",
         MC.TiledHamiltonian(1, _fourbody_terms(0.05); dims = (7, 1, 1)),
         [(1, 4), (3, 6)], 1),
        ("4-body end pair (F4)", MC.TiledHamiltonian(4, _f4_terms()),
         [(1, 4), (4, 1)], 1),
        ("several instances (F4: 4-body + pair)", MC.TiledHamiltonian(4, _f4_terms()),
         [(1, 2), (2, 1)], 2),
        ("several instances (F3: pair + 3-body)", MC.TiledHamiltonian(3, _f3_terms()),
         [(1, 2), (3, 1), (2, 3)], 2),
        ("several instances (F3a: pairs + 3-body)", MC.TiledHamiltonian(3, _f3a_terms()),
         [(1, 2), (2, 3), (1, 3)], 3),
        ("no shared instance (F3, cross-cell)",
         MC.TiledHamiltonian(3, _f3_terms(); dims = (2, 1, 1)),
         [(1, 4), (2, 6), (3, 5)], 0),
        ("no shared instance (F4, cross-cell)",
         MC.TiledHamiltonian(4, _f4_terms(); dims = (2, 1, 1)),
         [(1, 5), (4, 8)], 0),
        ("no shared instance (3-body chain)",
         MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (7, 1, 1)),
         [(1, 5)], 0),
    ]

    for (name, H, pairs, nshared) in cases
        @testset "G2: $name" begin
            for (i, j) in pairs
                @test _n_shared_instances(H, i, j) == nshared
            end
            scratch = MC._CMCScratch(H)
            worst = 0.0
            naive_gap = 0.0          # largest |naive − exact| — the mutation's size
            restored = true
            rows_out = true
            for (i, j) in pairs, _ = 1:5
                config = _rand_config(rng, H)
                zrows = MC._zrows(H, config)
                zrows_before = copy(zrows)
                e_i, e_j = _rand_spin(rng), _rand_spin(rng)

                ΔE = MC._pair_delta_energy!(scratch, zrows, H, i, j, e_i, e_j)

                after = copy(config)
                after[i] = e_i
                after[j] = e_j
                exact = total_energy(H, after) - total_energy(H, config)
                @test ΔE ≈ exact atol = 1e-12
                worst = max(worst, abs(ΔE - exact))
                naive = _en_naive_pair_dE(H, zrows_before, i, j, e_i, e_j)
                naive_gap = max(naive_gap, abs(naive - exact))
                # rejection leaves the cached rows bitwise intact (column i was
                # written tentatively and must be restored) — compared as bit
                # patterns, so a signed zero or a NaN cannot hide behind `==`
                bits = reinterpret(UInt64, zrows)
                restored &= bits == reinterpret(UInt64, zrows_before)
                restored &= bits == reinterpret(UInt64, MC._zrows(H, config))
                # the proposed rows are left in the scratch for the accept path
                rows_out &= scratch.zi_new == MC._zlm_row!(zeros(H.nlm), e_i, H.lmax)
                rows_out &= scratch.zj_new == MC._zlm_row!(zeros(H.nlm), e_j, H.lmax)
                if nshared == 0
                    # no instance couples i and j ⇒ c_j never reads column i ⇒ the
                    # sequential form IS the naive sum, bit for bit
                    @test ΔE == naive
                end
            end
            @test restored
            @test rows_out
            @test worst < 1e-13
            if nshared > 0
                # the case discriminates: the naive sum is off by far more than the
                # tolerance, so dropping the tentative write cannot pass it
                @test naive_gap > 1e-5
            end
        end
    end

    @testset "G2: allocation-free, i ≠ j asserted" begin
        # pair/triplet fast paths (F3a), the general body ≥ 4 branch (F4), and the
        # appended Zeeman templates (FZ)
        for H in (MC.TiledHamiltonian(3, _f3a_terms()),
                  MC.TiledHamiltonian(4, _f4_terms()),
                  MC.TiledHamiltonian(2, _f2g_terms(); magmoms = _fz_magmoms(),
                                      field = _fz_field()))
            config = _rand_config(rng, H)
            zrows = MC._zrows(H, config)
            scratch = MC._CMCScratch(H)
            @test _en_pair_dE_allocs(scratch, zrows, H, 1, 2, _rand_spin(rng),
                                  _rand_spin(rng)) == 0
            @test_throws AssertionError MC._pair_delta_energy!(
                scratch, zrows, H, 2, 2, _rand_spin(rng), _rand_spin(rng))
        end
    end

    @testset "G2: a harmonics DomainError leaves zrows intact" begin
        # a proposal a few ulp off the unit sphere makes the tesseral rows throw;
        # both rows are tabulated before column i is written, so the cached rows
        # must come out bitwise unchanged
        H = MC.TiledHamiltonian(3, _f3a_terms())
        config = _rand_config(rng, H)
        zrows = MC._zrows(H, config)
        before = copy(zrows)
        scratch = MC._CMCScratch(H)
        bad = SVector(0.0, 0.0, 1.0 + 4e-16)
        @test_throws DomainError MC._pair_delta_energy!(scratch, zrows, H, 1, 2,
                                                        _rand_spin(rng), bad)
        @test reinterpret(UInt64, zrows) == reinterpret(UInt64, before)
    end

    # A body-1 l = 0 term makes every site active while leaving the energy constant
    # (Z₀₀ is a constant): an E = const ensemble without an all-site coupling, for
    # the constrained sampler's geometric gates. Record that the constructor
    # accepts it and that it is inert: same energy for every configuration, zero
    # gradient, zero torque, zero ΔE.
    @testset "G2: body-1 l = 0 term: all sites active, energy constant" begin
        terms = [MultipoleTerm(0.3, 1, [a], [_CMC_Z], [0], [1.0]) for a = 1:3]
        H = MC.TiledHamiltonian(3, terms; dims = (2, 1, 1))
        @test all(H.site_active) && H.n_active == 6
        @test H.lmax == 0 && H.nlm == 1
        energies = [total_energy(H, _rand_config(rng, H)) for _ = 1:5]
        @test all(==(energies[1]), energies)
        # the constant itself: 6 instances × coef·(4π)^(1/2)·Z₀₀, Z₀₀ = 1/√(4π)
        @test energies[1] ≈ 6 * 0.3 atol = 1e-14
        config = _rand_config(rng, H)
        @test all(iszero, MC.energy_gradient(H, config))
        @test MC.total_torque(H, config) == zero(SVector{3,Float64})
        scratch = MC._CMCScratch(H)
        @test MC._pair_delta_energy!(scratch, MC._zrows(H, config), H, 1, 4,
                                     _rand_spin(rng), _rand_spin(rng)) == 0.0
    end
end

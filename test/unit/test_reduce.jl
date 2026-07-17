# Cell reduction (`reduce_cell`): exact recovery of hand-built small-cell terms from
# their unfolded supercell form (diagonal and non-diagonal M), energy identities via
# the site permutation, fitted-model paths, and the verification error cases.

# Unfold small-cell terms onto a diagonal N₁×N₂×N₃ supercell: the training-cell term
# list a model fitted on that supercell would expose (atom ordering matches
# `supercell_crystal` / `site_index` — atom fastest, cells column-major).
function _unfold_diag(sub_terms, nsub_atoms, dims::NTuple{3,Int})
    d = SVector{3,Int}(dims)
    out = MultipoleTerm[]
    for c3 = 0:(d[3] - 1), c2 = 0:(d[2] - 1), c1 = 0:(d[1] - 1)
        t = SVector(c1, c2, c3)
        for mt in sub_terms
            atoms = Int[]
            shifts = SVector{3,Int}[]
            for (a, sh) in zip(mt.atoms, mt.shifts)
                σ = t + sh
                cw = mod.(σ, d)
                push!(atoms, a + nsub_atoms * (cw[1] + d[1] * (cw[2] + d[2] * cw[3])))
                push!(shifts, fld.(σ, d))
            end
            push!(out, MultipoleTerm(mt.coef, length(atoms), atoms, shifts,
                                     copy(mt.ls), copy(mt.folded)))
        end
    end
    return out
end

# For a *diagonal* reduction matrix M: the permutation taking training-tiled site s
# (of H_tr, dims D) to the equivalent reduced-tiled site (of H_red, dims |M|·D; the
# wrap makes negative-diagonal — left-handed — M work too).
function _reduce_perm(red, H_tr, H_red)
    md = SVector(red.M[1, 1], red.M[2, 2], red.M[3, 3])
    perm = zeros(Int, H_tr.n_sites)
    for s = 1:H_tr.n_sites
        a = MC.site_atom(H_tr, s)
        idx = (s - a) ÷ H_tr.n_cell_atoms
        c = SVector(idx % H_tr.dims[1], (idx ÷ H_tr.dims[1]) % H_tr.dims[2],
                    idx ÷ (H_tr.dims[1] * H_tr.dims[2]))
        b, o = red.atom_map[a]
        perm[s] = MC.site_index(H_red, b, mod.(o + md .* c, H_red.dims))
    end
    return perm
end

# General M: which training atom's coset the reduced cell `c` belongs to
# (c ≡ o_a mod M·ℤ³, decided exactly with the integer adjugate).
function _coset_atom(red, c::SVector{3,Int})
    m = red.M
    dt = MC._det3(m)
    adj = SMatrix{3,3,Int}(m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2],
                           m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3],
                           m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1],
                           m[1, 3] * m[3, 2] - m[1, 2] * m[3, 3],
                           m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1],
                           m[1, 2] * m[3, 1] - m[1, 1] * m[3, 2],
                           m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2],
                           m[1, 3] * m[2, 1] - m[1, 1] * m[2, 3],
                           m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
    for (a, (_, o)) in enumerate(red.atom_map)
        v = adj * (c - o)
        all(x -> x % dt == 0, v) && return a
    end
    error("reduced cell $c matches no coset of M = $(red.M)")
end

_permute_config(cfg, perm) = begin
    out = MC.SpinConfig(undef, length(cfg))
    for s = 1:length(cfg)
        out[perm[s]] = cfg[s]
    end
    out
end

@testset "cell reduction" begin
    sub_lat = Matrix(1.0 * I(3))
    sub_cr = Crystal(Lattice(sub_lat), reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
    sub_terms = _chain_terms(-0.03)

    @testset "diagonal supercell: exact recovery and energy identity" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3}(Diagonal([2, 2, 1]))
        @test red.parent_atoms == [1]
        # the ±x directed pair aligns onto one canonical key → two copies of the
        # (0, +x)-form representative
        @test length(red.terms) == 2
        for rt in red.terms
            st = sub_terms[1]
            @test rt.coef == st.coef
            @test rt.atoms == st.atoms
            @test rt.shifts == st.shifts
            @test rt.ls == st.ls
            @test rt.folded == st.folded
        end
        @test cartesian_positions(red.crystal) == zeros(3, 1)

        H_tr = MC.TiledHamiltonian(4, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        rng = MersenneTwister(11)
        cfg = _rand_config(rng, H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        @test total_energy(H_red, cfg_red) ≈ total_energy(H_tr, cfg) atol = 1e-13
        # equal to tiling the original small-cell terms directly (the reduced list
        # regroups the ±x pair, so equality is to summation order)
        H_sub = MC.TiledHamiltonian(1, sub_terms; dims = (2, 2, 1))
        @test total_energy(H_sub, cfg_red) ≈ total_energy(H_red, cfg_red) atol = 1e-13

        # non-training-multiple sizes are the point: 3×2×2 of the 1-atom cell
        H_tr2 = MC.TiledHamiltonian(4, tr_terms; dims = (3, 1, 2))
        H_red2 = TiledHamiltonian(red; dims = (6, 2, 2))
        cfg2 = _rand_config(rng, H_tr2)
        cfg_red2 = _permute_config(cfg2, _reduce_perm(red, H_tr2, H_red2))
        @test total_energy(H_red2, cfg_red2) ≈ total_energy(H_tr2, cfg2) atol = 1e-12
        H_odd = TiledHamiltonian(red; dims = (3, 3, 1))   # not a multiple of (2,2,1)
        @test MC.n_sites(H_odd) == 9
    end

    @testset "non-diagonal M (det 2): exact recovery" begin
        # A_train = A_sub · M with columns v₁ = (1,-1,0), v₂ = (1,1,0), v₃ = ẑ;
        # the two cosets sit at cart (0,0,0) and (1,0,0). Hand-unfolded ±x chain.
        mB = [1 1 0; -1 1 0; 0 0 1]
        tr_cr = Crystal(Lattice(Float64.(mB)), [0 0.5; 0 0.5; 0.0 0.0], [1, 1],
                        ["Fe"])
        raw = sub_terms[1].coef
        folded = sub_terms[1].folded
        z = SVector(0, 0, 0)
        tr_terms = [MultipoleTerm(raw, 2, [1, 2], [z, z], [1, 1], copy(folded)),
                    MultipoleTerm(raw, 2, [2, 1], [z, SVector(1, 1, 0)], [1, 1],
                                  copy(folded)),
                    MultipoleTerm(raw, 2, [1, 2], [z, SVector(-1, -1, 0)], [1, 1],
                                  copy(folded)),
                    MultipoleTerm(raw, 2, [2, 1], [z, z], [1, 1], copy(folded))]
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3,Int}(mB)
        @test length(red.terms) == 2
        for rt in red.terms                     # two copies of the +x-form rep
            st = sub_terms[1]
            @test rt.coef == st.coef
            @test rt.atoms == st.atoms
            @test rt.shifts == st.shifts
            @test rt.folded == st.folded
        end
        # per-site energy identity on a sub-periodic (here: uniform) configuration
        e = _rand_spin(MersenneTwister(3))
        H_tr = MC.TiledHamiltonian(2, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (3, 1, 1))
        E_tr = total_energy(H_tr, MC.SpinConfig([e, e]))
        E_red = total_energy(H_red, MC.SpinConfig([e, e, e]))
        @test E_red / 3 ≈ E_tr / 2 atol = 1e-14
    end

    @testset "fitted model, identity reduction (|det M| = 1)" begin
        model = _dimer_model()
        cr = _dimer_crystal()
        red = reduce_cell(model, cr, Matrix(cr.lattice.vectors))
        @test n_atoms(red) == 4
        @test length(red.terms) == length(multipole_terms(model))
        H_a = TiledHamiltonian(model; dims = (2, 1, 2))
        H_b = TiledHamiltonian(red; dims = (2, 1, 2))
        cfg = _rand_config(MersenneTwister(5), H_a)
        @test total_energy(H_b, cfg) ≈ total_energy(H_a, cfg) atol = 1e-13
    end

    @testset "fitted model, genuine 2× reduction + predict_energy gate" begin
        model, cr = _stacked_chain_model()
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test n_atoms(red) == 1
        @test length(red.terms) == length(multipole_terms(model)) ÷ 2
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (1, 1, 2))
        rng = MersenneTwister(17)
        cfg = _rand_config(rng, H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        E = total_energy(H_red, cfg_red)
        @test E ≈ total_energy(H_tr, cfg) atol = 1e-13
        @test E ≈ predict_energy(model, _config_matrix(cfg)) - intercept(model) atol =
            1e-12
        # a larger, non-commensurate tiling stays consistent with a training tiling
        H_tr2 = TiledHamiltonian(model; dims = (2, 2, 1))
        H_red2 = TiledHamiltonian(red; dims = (2, 2, 2))
        cfg2 = _rand_config(rng, H_tr2)
        cfg_red2 = _permute_config(cfg2, _reduce_perm(red, H_tr2, H_red2))
        @test total_energy(H_red2, cfg_red2) ≈ total_energy(H_tr2, cfg2) atol = 1e-12
    end

    @testset "two SALC channels on one cluster stay distinct" begin
        # same pair cluster, second channel with a different tensor and coefficient:
        # the (coef, folded) sub-partition must keep both, each with nc copies.
        ch1 = _chain_terms(-0.03)
        ising = zeros(3, 3)
        ising[3, 3] = 1.0                       # a second, distinct coupling tensor
        ch2 = [MultipoleTerm(0.007, 2, copy(t.atoms), copy(t.shifts), copy(t.ls),
                             copy(ising)) for t in ch1]
        sub2 = vcat(ch1, ch2)
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub2, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test length(red.terms) == 4
        # one canonical key; reps in encounter order, each emitted twice (the ±x
        # directed pair folds onto the +x form): [ch1, ch1, ch2, ch2]
        expected = [sub2[1], sub2[1], sub2[3], sub2[3]]
        for (rt, st) in zip(red.terms, expected)
            @test rt.coef == st.coef
            @test rt.shifts == st.shifts
            @test rt.folded == st.folded
        end
        H_sub2 = MC.TiledHamiltonian(1, sub2; dims = (2, 2, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        cfg = _rand_config(MersenneTwister(19), H_red)
        @test total_energy(H_red, cfg) ≈ total_energy(H_sub2, cfg) atol = 1e-13
    end

    @testset "fitted anisotropic model: channels survive a 2× reduction" begin
        model, cr = _stacked_anisotropic_model(SpglibBackend())
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test n_atoms(red) == 1
        @test length(red.terms) == length(multipole_terms(model)) ÷ 2
        # the sub-partition branch is genuinely exercised: some reduced cluster
        # carries several SALC channels (same anchored key, different folded)
        keys = [(t.atoms, t.shifts, t.ls) for t in red.terms]
        @test length(unique(keys)) < length(keys)
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (1, 1, 2))
        cfg = _rand_config(MersenneTwister(23), H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        E = total_energy(H_red, cfg_red)
        @test E ≈ total_energy(H_tr, cfg) atol = 1e-12
        @test E ≈ predict_energy(model, _config_matrix(cfg)) - intercept(model) atol =
            1e-12

        # NoSymmetry per-bond orbits do NOT align their SALC tensor bases across
        # translation partners, so even equal-fill coefficients genuinely break the
        # half-cell periodicity — reduce_cell must refuse (a physics refusal, not a
        # tolerance artifact).
        model_ns, cr_ns = _stacked_anisotropic_model(NoSymmetry(); fill_coefs = true)
        @test_throws ArgumentError reduce_cell(model_ns, cr_ns,
                                               [4.0 0 0; 0 4.0 0; 0 0 2.0])
    end

    @testset "fitted model, non-diagonal M: non-uniform energy identity" begin
        model, cr = _checkerboard_model()
        red = reduce_cell(model, cr, [1.0 0 0; 0 1.0 0; 0 0 4.0])
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3,Int}([1 1 0; -1 1 0; 0 0 1])
        @test length(red.terms) == length(multipole_terms(model)) ÷ 2
        # a training-periodic (not uniform!) configuration: paint each reduced cell
        # with its coset's spin — diag(2,2,1) = M·[1 -1 0; 1 1 0; 0 0 1] wraps a
        # sublattice of M·ℤ³ and covers two training cells.
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        rng = MersenneTwister(29)
        cfg_tr = MC.SpinConfig([_rand_spin(rng), _rand_spin(rng)])
        cfg_red = MC.SpinConfig(undef, MC.n_sites(H_red))
        for s = 1:MC.n_sites(H_red)
            idx = s - 1                          # one atom per reduced cell
            c = SVector(idx % 2, (idx ÷ 2) % 2, idx ÷ 4)
            cfg_red[s] = cfg_tr[_coset_atom(red, c)]
        end
        E_tr = total_energy(H_tr, cfg_tr)
        @test total_energy(H_red, cfg_red) ≈ 2 * E_tr atol = 1e-13
        @test E_tr ≈ predict_energy(model, _config_matrix(cfg_tr)) - intercept(model) atol =
            1e-12
    end

    @testset "left-handed reduced cell (det M < 0)" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, Matrix(Diagonal([1.0, -1.0, 1.0])))
        @test MC._det3(red.M) == -4              # M = diag(2, -2, 1)
        @test n_atoms(red) == 1
        @test length(red.terms) == 2
        for rt in red.terms                     # ±x untouched by the y flip;
            st = sub_terms[1]                   # two copies of the +x-form rep
            @test rt.coef == st.coef
            @test rt.shifts == st.shifts
        end
        H_tr = MC.TiledHamiltonian(4, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        cfg = _rand_config(MersenneTwister(31), H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        @test total_energy(H_red, cfg_red) ≈ total_energy(H_tr, cfg) atol = 1e-13
    end

    @testset "verification errors" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))

        # a coefficient that breaks the translation symmetry of the Hamiltonian
        bad = copy(tr_terms)
        bad[3] = MultipoleTerm(bad[3].coef * 1.001, 2, copy(bad[3].atoms),
                               copy(bad[3].shifts), copy(bad[3].ls),
                               copy(bad[3].folded))
        @test_throws ArgumentError reduce_cell(tr_cr, bad, sub_lat)

        # a distorted structure (atom off its translation image)
        frac = copy(tr_cr.frac_positions)
        frac[1, 2] += 0.02
        cr_bad = Crystal(tr_cr.lattice, frac, tr_cr.species, tr_cr.species_labels)
        @test_throws ArgumentError reduce_cell(cr_bad, tr_terms, sub_lat)

        # lattice not an integer relation
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms, Matrix(0.9 * I(3)))
        # |det M| does not divide n_atoms (A_sub = diag(2/3, 1, 1) ⇒ M = diag(3,2,1))
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms,
                                               Matrix(Diagonal([2 / 3, 1.0, 1.0])))
        # model/crystal mismatch
        @test_throws ArgumentError reduce_cell(_dimer_model(), sub_cr, sub_lat)
        # wrong sub_lattice shape
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms, ones(2, 2))
        # empty term list
        @test_throws ArgumentError reduce_cell(tr_cr, MultipoleTerm[], sub_lat)

        # geometrically periodic but chemically not: species differ across cosets
        tr2 = supercell_crystal(sub_cr, (2, 1, 1))
        cr_species = Crystal(tr2.lattice, tr2.frac_positions, [1, 2], ["Fe", "Co"])
        terms2 = _unfold_diag(sub_terms, 1, (2, 1, 1))
        @test_throws ArgumentError reduce_cell(cr_species, terms2, sub_lat)

        # coincident atoms folding onto one reduced site
        lat2 = Lattice(Matrix(Diagonal([2.0, 1.0, 1.0])))
        cr_dup = Crystal(lat2, [0.25 0.25; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        @test_throws ArgumentError reduce_cell(cr_dup, sub_terms, Matrix(1.0 * I(3)))
    end
end

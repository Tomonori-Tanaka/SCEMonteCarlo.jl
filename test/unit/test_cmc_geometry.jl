# Constrained MC: the constraint type, its resolution, the constrained
# initialization, and the pure pair proposal (docs/specs/constrained-mc.md D1/D2).
#
# G0 — doors and initialization: every refusal of the constraint / resolution /
#      init doors; the default pattern and a rotated `init` give M̂ = n̂ (M recomputed
#      here, not by the package); the exactly antiparallel `init`; sign(w) in the
#      default pattern; no RNG consumed.
# G1 — geometry of the compensation: after 10⁴ applied proposals the transverse
#      weighted magnetization, recomputed from scratch here, stays at rounding level;
#      every spin stays unit; i = j is refused.
# G4 — the log acceptance ratio against an independent formula: ΔE from
#      `total_energy`, M from scratch, the partner spin from a frame-based
#      reimplementation of the compensation; reversibility; null moves.
#
# Every oracle below is written in the test: weights and active sets come from the
# fixture definitions (all fixtures here are dims = (1,1,1), so site s = atom s), M
# is summed here, and the compensation is recomputed in an explicit orthonormal
# frame (t₁, t₂, n̂) — the package works frame-free with projections onto n̂⊥.

_geo_wnorm(w, active) = sum(abs(w[s]) for s in eachindex(w) if active[s])

# Frame-based reference of the compensation: in coordinates (x, y, z) along
# (t₁, t₂, n̂), the partner's transverse coordinates move by −(w_i/w_j) times the
# source's; its z keeps its sign (+ when zero) and restores unit length. Returns
# (e_j′, s2), with e_j′ = nothing when s2 ≥ 1.
function _geo_ref_partner(e_i, e_i_new, e_j, w_i, w_j, n̂)
    t1, t2, _ = _cmcref_frame(n̂)
    r = w_i / w_j
    x = dot(e_j, t1) - r * (dot(e_i_new, t1) - dot(e_i, t1))
    y = dot(e_j, t2) - r * (dot(e_i_new, t2) - dot(e_i, t2))
    s2 = x^2 + y^2
    s2 >= 1 && return (nothing, s2)
    zj = dot(e_j, n̂)
    z = (zj >= 0 ? 1.0 : -1.0) * sqrt(1 - s2)
    return (x * t1 + y * t2 + z * n̂, s2)
end

# A random state of the constrained ensemble: a random configuration rotated rigidly
# (test-side Rodrigues about M̂ × n̂) so that its weighted M̂ is n̂.
function _geo_valid_state(rng, nsites, w, active, n̂)
    config = [_rand_spin(rng) for _ = 1:nsites]
    M = _cmcref_M(config, w, active)
    Mh = normalize(M)
    ax = cross(Mh, n̂)
    α = atan(norm(ax), dot(Mh, n̂))
    ax = normalize(ax)
    return MC.SpinConfig([normalize(_cmcref_rot(e, ax, α)) for e in config])
end

# A random rotation of `e` (uniform axis, angle step·randn), re-normalized.
_geo_propose(rng, e, step) = normalize(_cmcref_rot(e, _rand_spin(rng), step * randn(rng)))

# The fixtures of G1/G4: (name, H, constraint weights, per-site weights, active).
function _geo_cases(n̂)
    f2g = MC.TiledHamiltonian(2, _f2g_terms())
    f3 = MC.TiledHamiltonian(3, _f3_terms())
    fmix = MC.TiledHamiltonian(8, _fmix_terms())
    act_fmix = [s != _FMIX_INACTIVE for s = 1:8]
    return [
        ("F2g (:auto ⇒ uniform)", f2g, :auto, [1.0, 1.0], trues(2)),
        ("F3 (:auto ⇒ uniform)", f3, :auto, ones(3), trues(3)),
        ("F2w+", f2g, _f2w_weights(+1), _f2w_weights(+1), trues(2)),
        ("F2w−", f2g, _f2w_weights(-1), _f2w_weights(-1), trues(2)),
        ("Fmix", fmix, _fmix_weights(), _fmix_weights(), act_fmix),
    ]
end


# Violation tally of a looped check: loops only record (count, first offender); one
# `@test` per quantity follows the loop, and its failure message shows the first
# offending case — `@test (name, n, first) == (name, 0, "")`.
mutable struct _GeoViolations
    n::Int
    first::String
end
_GeoViolations() = _GeoViolations(0, "")

function _geo_note!(v::_GeoViolations, ok::Bool, at)
    ok && return v
    v.n == 0 && (v.first = string(at))
    v.n += 1
    return v
end

_geo_tallies(keys) = Dict(k => _GeoViolations() for k in keys)

macro _geo_test_tallies(tallies)
    quote
        for (k, t) in sort!(collect($(esc(tallies))); by = first)
            @test (k, t.n, t.first) == (k, 0, "")
        end
    end
end

@testset "CMC geometry" begin
    θ0, φ0 = 0.6, 0.35
    n̂ = _cmcref_dir(θ0, φ0)
    # every Cartesian component of n̂ is far from 0 and 1: a lab-frame mix-up shows
    @test all(x -> 0.15 < abs(x) < 0.9, n̂)

    @testset "G0: MagnetizationConstraint" begin
        c = MC.MagnetizationConstraint([0.0, 0.0, 2.0])
        @test c.direction == SVector(0.0, 0.0, 1.0)
        @test c.weights === :auto
        ca = MC.MagnetizationConstraint(θ0, φ0; weights = [1, -2])
        @test ca.direction ≈ n̂ atol = 1e-15
        @test ca.weights == [1.0, -2.0] && ca.weights isa Vector{Float64}
        @test norm(MC.MagnetizationConstraint([3.0, -4.0, 12.0]).direction) ≈ 1 atol = 1e-15
        @test MC.MagnetizationConstraint([1.0, 1.0, 1.0]).direction ≈
              SVector(1, 1, 1) / sqrt(3) atol = 1e-15
        @test MC.MagnetizationConstraint((0, 0, 2)) ==
              MC.MagnetizationConstraint([0.0, 0.0, 1.0])
        @test all(r -> MC.MagnetizationConstraint(n̂; weights = r).weights === r,
                  (:auto, :uniform, :magmoms))
        # the weight vector is copied, not aliased
        wsrc = [1.0, 2.0]
        cw = MC.MagnetizationConstraint(n̂; weights = wsrc)
        wsrc[1] = 99.0
        @test cw.weights == [1.0, 2.0]
        # no positional (direction, weights) form: the inner constructors are
        # internal (tagged), so a raw direction cannot skip the public doors
        @test_throws MethodError MC.MagnetizationConstraint(SVector(0.0, 0.0, 2.0),
                                                             :auto)
        @test_throws MethodError MC.MagnetizationConstraint(SVector(0.0, 0.0, 2.0),
                                                             [1.0, 2.0])

        # equality, isequal, hash — field-wise, across separately built objects
        c1 = MC.MagnetizationConstraint(n̂; weights = [1.0, -0.5])
        c2 = MC.MagnetizationConstraint(Vector(n̂); weights = [1.0, -0.5])
        @test c1 == c2 && isequal(c1, c2) && hash(c1) == hash(c2)
        @test c1 != MC.MagnetizationConstraint(n̂; weights = [1.0, 0.5])
        @test c1 != MC.MagnetizationConstraint(n̂)
        @test MC.MagnetizationConstraint(n̂) == MC.MagnetizationConstraint(n̂)
        @test c1 != MC.MagnetizationConstraint(-n̂; weights = [1.0, -0.5])
        @test length(Set([c1, c2, MC.MagnetizationConstraint(n̂)])) == 2
        # signed zeros are folded, so ==, isequal and hash agree: θ = 0, φ = 2 gives
        # sin(0)·cos(2) = −0.0 in the x component
        cθ = MC.MagnetizationConstraint(0.0, 2.0; weights = [-0.0, 1.0])
        cz = MC.MagnetizationConstraint([0, 0, 1]; weights = [0.0, 1.0])
        @test cθ == cz && isequal(cθ, cz) && hash(cθ) == hash(cz)

        # doors: direction zero / non-finite / wrong length; weights rule / finite
        @test all(bad -> _cmcref_thrown(() -> MC.MagnetizationConstraint(bad)) isa
                         ArgumentError,
                  ([0.0, 0.0, 0.0], [NaN, 0.0, 1.0], [Inf, 0.0, 1.0], [0, -Inf, 0],
                   (0.0, 0.0, 0.0)))
        # a huge but finite direction is normalized, not refused
        @test MC.MagnetizationConstraint([1e308, 1e308, 1e308]).direction ≈
              SVector(1, 1, 1) / sqrt(3) atol = 1e-15
        @test_throws ArgumentError MC.MagnetizationConstraint([1.0, 0.0])
        @test_throws ArgumentError MC.MagnetizationConstraint(NaN, 0.3)
        @test_throws ArgumentError MC.MagnetizationConstraint(0.3, Inf)
        @test_throws ArgumentError MC.MagnetizationConstraint(n̂; weights = :foo)
        @test_throws ArgumentError MC.MagnetizationConstraint(n̂; weights = [1.0, NaN])
        @test_throws ArgumentError MC.MagnetizationConstraint(n̂; weights = [Inf, 1.0])
        @test_throws ArgumentError MC.MagnetizationConstraint(n̂; weights = "uniform")
    end

    @testset "G0: weight resolution" begin
        f2g = MC.TiledHamiltonian(2, _f2g_terms())
        fz = MC.TiledHamiltonian(2, _f2g_terms(); magmoms = _fz_magmoms(),
                                 field = _fz_field())
        resolve(H, w) =
            MC._resolve_constraint(H, MC.MagnetizationConstraint(n̂; weights = w))
        @test resolve(f2g, :auto).cell_weights == [1.0, 1.0]
        @test resolve(fz, :auto).cell_weights == _fz_magmoms()
        @test resolve(fz, :uniform).cell_weights == [1.0, 1.0]
        @test resolve(fz, :magmoms).cell_weights == _fz_magmoms()
        @test resolve(f2g, [1.0, -2.0]).cell_weights == [1.0, -2.0]
        @test resolve(f2g, :auto).direction == MC.MagnetizationConstraint(n̂).direction
        e = _cmcref_thrown(() -> resolve(f2g, :magmoms))
        @test e isa ArgumentError && occursin("magmoms", e.msg)
        @test_throws ArgumentError resolve(f2g, [1.0])
        @test_throws ArgumentError resolve(f2g, [1.0, 2.0, 3.0])
        # |𝒞| < 2: one nonzero weight
        @test_throws ArgumentError resolve(f2g, [1.0, 0.0])
        @test_throws ArgumentError resolve(f2g, [0.0, 0.0])
        # a weight vector mutated after construction is re-checked
        cm = MC.MagnetizationConstraint(n̂; weights = [1.0, 2.0])
        cm.weights[1] = NaN
        @test_throws ArgumentError MC._resolve_constraint(f2g, cm)

        # Fmix: the inactive site (nonzero weight) and the zero-weight active site
        # are both outside 𝒞; site_w is zero there
        fmix = MC.TiledHamiltonian(8, _fmix_terms())
        @test !fmix.site_active[_FMIX_INACTIVE] && fmix.site_active[_FMIX_ZERO_W]
        @test _fmix_weights()[_FMIX_INACTIVE] != 0
        rc = resolve(fmix, _fmix_weights())
        @test rc.constrained_sites == [1, 2, 3, 4, 5, 6]
        @test rc.site_weights == [1.0, -0.4, 1.5, -1.2, 0.8, 2.0, 0.0, 0.0]
        @test rc.cell_weights == _fmix_weights()
        # a weight vector whose only other nonzero entry sits on the inactive site
        # leaves |𝒞| = 1 — refused (an inactive site counted in 𝒞 would pass)
        wlone = zeros(8)
        wlone[1] = 1.0
        wlone[_FMIX_INACTIVE] = 3.0
        @test_throws ArgumentError resolve(fmix, wlone)

        # supercell sites take their atom's weight: F3 on dims = (2,1,1), site s ↦
        # atom mod1(s, 3) (the site_index ordering of `site_atom`, written out here)
        f3x2 = MC.TiledHamiltonian(3, _f3_terms(); dims = (2, 1, 1))
        rc3 = resolve(f3x2, [1.0, -2.0, 0.5])
        @test rc3.site_weights == [[1.0, -2.0, 0.5][mod1(s, 3)] for s = 1:6]
        @test rc3.constrained_sites == collect(1:6)
    end

    @testset "G0: constrained initialization" begin
        fmix = MC.TiledHamiltonian(8, _fmix_terms())
        w = _fmix_weights()
        act = [s != _FMIX_INACTIVE for s = 1:8]
        wn = _geo_wnorm(w, act)
        rc = MC._resolve_constraint(fmix, MC.MagnetizationConstraint(n̂; weights = w))
        n̂c = rc.direction

        # no RNG: the signature takes none, and the global stream is untouched
        g0 = copy(Random.default_rng())
        config = MC._cmc_initial_config(fmix, rc, nothing)
        @test copy(Random.default_rng()) == g0
        @test config == MC._cmc_initial_config(fmix, rc, nothing)

        # default pattern: sign(w) n̂ on 𝒞, n̂ on the zero-weight and inactive sites
        @test config == [(act[s] && w[s] < 0) ? -n̂c : n̂c for s = 1:8]
        @test config[2] == -n̂c && config[4] == -n̂c    # the negative weights
        M = _cmcref_M(config, w, act)
        @test norm(normalize(M) - n̂) < 1e-14
        @test dot(M, n̂) ≈ wn rtol = 1e-14             # every site adds |w| n̂

        # a rotated random init (Fmix, F2w±, F3): M̂ = n̂, and the rotation is rigid
        rng = MersenneTwister(31)
        f2g = MC.TiledHamiltonian(2, _f2g_terms())
        f3 = MC.TiledHamiltonian(3, _f3_terms())
        v = _geo_tallies((:rng, :Mhat, :unit, :rigid, :matrix_form))
        for (name, H, wv, av) in (("Fmix", fmix, w, act),
                                  ("F2w+", f2g, _f2w_weights(+1), trues(2)),
                                  ("F2w−", f2g, _f2w_weights(-1), trues(2)),
                                  ("F3", f3, ones(3), trues(3)))
            rcH = MC._resolve_constraint(H,
                                         MC.MagnetizationConstraint(n̂; weights = wv))
            for trial = 1:20
                at = (name, trial)
                init = _rand_config(rng, H)
                g = copy(Random.default_rng())
                out = MC._cmc_initial_config(H, rcH, init)
                _geo_note!(v[:rng], copy(Random.default_rng()) == g, at)
                _geo_note!(v[:Mhat],
                           norm(normalize(_cmcref_M(out, wv, av)) - n̂) < 1e-14, at)
                _geo_note!(v[:unit], all(e -> abs(norm(e) - 1) < 1e-15, out), at)
                _geo_note!(v[:rigid],
                       maximum(abs(dot(out[s], out[t]) - dot(init[s], init[t]))
                               for s in eachindex(out), t in eachindex(out)) < 1e-14,
                       at)
                # the init matrix form gives the same start
                _geo_note!(v[:matrix_form],
                       MC._cmc_initial_config(H, rcH, _config_matrix(init)) == out, at)
            end
        end
        @_geo_test_tallies v

        # minimality: the rotation axis k = M̂₀ × n̂ / |…| is fixed. Fmix sites 7
        # (w = 0) and 8 (inactive) do not enter M, so they can carry k and M̂₀ + k.
        # Acute and obtuse starts (M̂₀·n̂ of both signs) are both exercised.
        n_obtuse = 0
        vmin = _geo_tallies((:axis_fixed, :plane))
        for trial = 1:12
            init = _rand_config(rng, fmix)
            Mh0 = normalize(_cmcref_M(init, w, act))
            n_obtuse += dot(Mh0, n̂) < 0
            k = normalize(cross(Mh0, n̂))
            init[_FMIX_ZERO_W] = k
            init[_FMIX_INACTIVE] = normalize(Mh0 + k)
            out = MC._cmc_initial_config(fmix, rc, init)
            _geo_note!(vmin[:axis_fixed], norm(out[_FMIX_ZERO_W] - k) < 1e-14, trial)
            _geo_note!(vmin[:plane],
                   norm(out[_FMIX_INACTIVE] - normalize(n̂ + k)) < 1e-14, trial)
        end
        @_geo_test_tallies vmin
        @test 3 <= n_obtuse <= 9

        # exactly antiparallel M̂ = −n̂: rotation by π about normalize(n̂ × x̂)
        cz = MC._resolve_constraint(fmix,
                                    MC.MagnetizationConstraint([0.0, 0.0, 1.0];
                                                               weights = w))
        anti = [act[s] && w[s] != 0 ? -sign(w[s]) * SVector(0.0, 0.0, 1.0) :
                SVector(0.0, 0.0, 1.0) for s = 1:8]
        anti[_FMIX_ZERO_W] = SVector(0.6, 0.0, 0.8)
        anti[_FMIX_INACTIVE] = SVector(0.0, 1.0, 0.0)
        @test normalize(_cmcref_M(anti, w, act)) == SVector(0.0, 0.0, -1.0)
        out = MC._cmc_initial_config(fmix, cz, anti)
        # n̂ × x̂ = ẑ × x̂ = ŷ: (x, y, z) ↦ (−x, y, −z)
        @test norm(normalize(_cmcref_M(out, w, act)) - SVector(0.0, 0.0, 1.0)) < 1e-15
        @test norm(out[_FMIX_ZERO_W] - SVector(-0.6, 0.0, -0.8)) < 1e-15
        @test norm(out[_FMIX_INACTIVE] - SVector(0.0, 1.0, 0.0)) < 1e-15
        # n̂ ∥ x̂: n̂ × x̂ = 0, so the axis is ŷ
        cx = MC._resolve_constraint(fmix,
                                    MC.MagnetizationConstraint([1.0, 0.0, 0.0];
                                                               weights = w))
        antix = [act[s] && w[s] != 0 ? -sign(w[s]) * SVector(1.0, 0.0, 0.0) :
                 SVector(0.0, 0.0, 1.0) for s = 1:8]
        antix[_FMIX_INACTIVE] = SVector(0.0, 1.0, 0.0)
        @test normalize(_cmcref_M(antix, w, act)) == SVector(-1.0, 0.0, 0.0)
        out = MC._cmc_initial_config(fmix, cx, antix)
        @test norm(normalize(_cmcref_M(out, w, act)) - SVector(1.0, 0.0, 0.0)) < 1e-15
        @test norm(out[_FMIX_ZERO_W] - SVector(0.0, 0.0, -1.0)) < 1e-15
        @test norm(out[_FMIX_INACTIVE] - SVector(0.0, 1.0, 0.0)) < 1e-15
        # generic n̂, M̂ = −n̂ up to rounding (the negated default pattern)
        neg = [act[s] && w[s] != 0 ? -c : c for (s, c) in enumerate(config)]
        @test norm(normalize(_cmcref_M(neg, w, act)) + n̂) < 1e-14
        out = MC._cmc_initial_config(fmix, rc, neg)
        @test norm(normalize(_cmcref_M(out, w, act)) - n̂) < 1e-14
        @test maximum(abs(dot(out[s], out[t]) - dot(neg[s], neg[t]))
                      for s = 1:8, t = 1:8) < 1e-14
        # M̂ tilted from −n̂ by 1e-12 … 1e-4 rad (the ill-conditioned window of a
        # minimal rotation and both sides of its 1e-8 switch). Above the switch the
        # rotation is the minimal one: its axis k = M̂₀ × n̂ / |…| is fixed, to the
        # conditioning ~1e-16/tilt of the plane (checked with 1e-14/tilt).
        tax = normalize(cross(n̂, SVector(0.2, 0.9, -0.3)))
        vt = _geo_tallies((:Mhat, :rigid, :axis_fixed))
        for tilt in (1e-12, 1e-9, 1e-8, 1e-7, 1e-4)
            tl = [_cmcref_rot(e, tax, tilt) for e in neg]
            Mh0 = normalize(_cmcref_M(tl, w, act))
            k = normalize(cross(Mh0, n̂))
            tilt > 1e-8 && (tl[_FMIX_ZERO_W] = k)
            out = MC._cmc_initial_config(fmix, rc, tl)
            _geo_note!(vt[:Mhat],
                       norm(normalize(_cmcref_M(out, w, act)) - n̂) < 1e-14, tilt)
            _geo_note!(vt[:rigid], maximum(abs(dot(out[s], out[t]) - dot(tl[s], tl[t]))
                                       for s = 1:8, t = 1:8) < 1e-14, tilt)
            tilt > 1e-8 &&
                _geo_note!(vt[:axis_fixed],
                           norm(out[_FMIX_ZERO_W] - k) < 1e-14 / tilt, tilt)
        end
        @_geo_test_tallies vt

        # door: |M_init| < 1e-8 Σ_{𝒞}|w| (direction undefined); a small but
        # admissible M passes
        rc2 = MC._resolve_constraint(f2g, MC.MagnetizationConstraint(n̂))
        u = _rand_spin(rng)
        @test_throws ArgumentError MC._cmc_initial_config(f2g, rc2, [u, -u])
        perp = normalize(cross(u, SVector(0.3, -0.2, 0.9)))
        tiny = [u, -normalize(_cmcref_rot(u, perp, 1e-9))]     # |M| ≈ 1e-9 < 2e-8
        @test norm(tiny[1] + tiny[2]) < 2e-8
        @test_throws ArgumentError MC._cmc_initial_config(f2g, rc2, tiny)
        ok = [u, -normalize(_cmcref_rot(u, perp, 1e-6))]       # |M| ≈ 1e-6 > 2e-8
        @test MC._cmc_initial_config(f2g, rc2, ok) isa MC.SpinConfig
        # the scale is Σ|w| (here 4), not |𝒞| (3) or Σw² (6): F3 with weights
        # (2, 1, 1), M = 2u − u − R(α)u = u − R(α)u with |M| = 2 sin(α/2)
        rcw = MC._resolve_constraint(f3, MC.MagnetizationConstraint(n̂;
                                                                     weights = [2, 1, 1]))
        door(mag) = MC.SpinConfig([u, -u, -normalize(_cmcref_rot(u, perp,
                                                               2 * asin(mag / 2)))])
        @test norm(_cmcref_M(door(3.5e-8), [2, 1, 1], trues(3))) ≈ 3.5e-8 rtol = 1e-6
        @test_throws ArgumentError MC._cmc_initial_config(f3, rcw, door(3.5e-8))
        @test MC._cmc_initial_config(f3, rcw, door(4.5e-8)) isa MC.SpinConfig
        # init validation is the drivers' door
        @test_throws DimensionMismatch MC._cmc_initial_config(f2g, rc2, [u])
        @test_throws ArgumentError MC._cmc_initial_config(f2g, rc2, [2u, u])
    end

    @testset "G1: compensation keeps M⊥ fixed" begin
        rng = MersenneTwister(4401)
        fmix = MC.TiledHamiltonian(8, _fmix_terms())
        f2g = MC.TiledHamiltonian(2, _f2g_terms())
        act_fmix = [s != _FMIX_INACTIVE for s = 1:8]
        for (name, H, w, act) in (("Fmix", fmix, _fmix_weights(), act_fmix),
                                  ("F2w+", f2g, _f2w_weights(+1), trues(2)),
                                  ("F2w−", f2g, _f2w_weights(-1), trues(2)))
            @testset "G1: $name" begin
                rc = MC._resolve_constraint(H,
                                            MC.MagnetizationConstraint(n̂; weights = w))
                cset = [s for s in eachindex(w) if act[s] && w[s] != 0]
                wn = _geo_wnorm(w, act)
                scratch = MC._CMCScratch(H)
                config = _geo_valid_state(rng, H.n_sites, w, act, n̂)
                zrows = MC._zrows(H, config)
                applied = 0
                worst_perp = 0.0
                worst_unit = 0.0
                min_par = Inf
                for _ = 1:10_000
                    i = rand(rng, cset)
                    j = rand(rng, filter(!=(i), cset))
                    e_i_new = _geo_propose(rng, config[i], 0.4)
                    M = _cmcref_M(config, w, act)
                    e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(config, zrows, H, rc, M,
                                                           i, j, e_i_new, 7.0, scratch)
                    isfinite(ℓ) || continue
                    worst_unit = max(worst_unit, abs(norm(e_j_new) - 1))
                    config[i] = e_i_new
                    config[j] = e_j_new
                    MC._zlm_row!(view(zrows, :, i), e_i_new, H.lmax)
                    MC._zlm_row!(view(zrows, :, j), e_j_new, H.lmax)
                    applied += 1
                    M = _cmcref_M(config, w, act)
                    worst_perp = max(worst_perp, norm(M - dot(M, n̂) * n̂))
                    min_par = min(min_par, dot(M, n̂))
                end
                # Measured (seed 4401): 7.9k–8.5k applied moves per fixture,
                # max |M⊥|/Σ|w| = 1.4e-15 (Fmix) … 6.1e-15 (F2w±) — the rounding
                # random walk, ~160× below the bound; max ||e_j′| − 1| = 2.2e-16.
                @test applied >= 2_000                     # non-vacuity
                @test worst_perp <= 1e-12 * wn
                @test min_par > 0
                @test worst_unit <= 1e-15
                @test all(e -> abs(norm(e) - 1) <= 1e-15, config)
            end
        end

        # preconditions: i = j is refused even when the geometry alone would make it
        # a null move (AssertionError); an out-of-range site is a BoundsError; a
        # zero-weight partner is refused (AssertionError)
        f2g_rc = MC._resolve_constraint(f2g, MC.MagnetizationConstraint(n̂))
        config = _geo_valid_state(rng, 2, ones(2), trues(2), n̂)
        zr = MC._zrows(f2g, config)
        M = _cmcref_M(config, ones(2), trues(2))
        scratch = MC._CMCScratch(f2g)
        @test_throws AssertionError MC._cmc_pair_proposal(
            config, zr, f2g, f2g_rc, M, 1, 1, _geo_propose(rng, config[1], 0.1), 7.0,
            scratch)
        @test_throws AssertionError MC._cmc_pair_proposal(
            config, zr, f2g, f2g_rc, M, 2, 2, SVector(NaN, 0.0, 0.0), 7.0, scratch)
        @test_throws AssertionError MC._cmc_pair_proposal(
            config, zr, f2g, f2g_rc, M, 1, 1, -config[1], 7.0, scratch)
        @test_throws BoundsError MC._cmc_pair_proposal(
            config, zr, f2g, f2g_rc, M, 1, 3, config[1], 7.0, scratch)
        rcm = MC._resolve_constraint(fmix,
                                     MC.MagnetizationConstraint(n̂;
                                                                weights = _fmix_weights()))
        cm = _geo_valid_state(rng, 8, _fmix_weights(), act_fmix, n̂)
        @test_throws AssertionError MC._cmc_pair_proposal(
            cm, MC._zrows(fmix, cm), fmix, rcm, _cmcref_M(cm, _fmix_weights(), act_fmix),
            1, _FMIX_ZERO_W, cm[1], 7.0, MC._CMCScratch(fmix))
    end

    @testset "G4: log acceptance ratio" begin
        rng = MersenneTwister(4402)
        β = 23.0
        n_valid = n_null_s2 = n_null_M = n_negz = n_jac = n_zr = 0
        for (name, H, wcon, w, act) in _geo_cases(n̂)
            @testset "G4: $name" begin
                rc = MC._resolve_constraint(H,
                                            MC.MagnetizationConstraint(n̂;
                                                                       weights = wcon))
                cset = [s for s in eachindex(w) if act[s] && w[s] != 0]
                @test rc.constrained_sites == cset
                scratch = MC._CMCScratch(H)
                v = _geo_tallies((:pure, :null_s2, :null_M, :finite, :ref, :unit,
                                  :domain, :sign, :ΔE, :ℓ, :rev_finite, :rev_e,
                                  :rev_ℓ, :rev_ΔE))
                for trial = 1:1_000
                    step = (0.3, 1.0, 2.5)[mod1(trial, 3)]
                    x = _geo_valid_state(rng, H.n_sites, w, act, n̂)
                    zx = MC._zrows(H, x)
                    x0, zx0 = copy(x), copy(zx)
                    i = rand(rng, cset)
                    j = rand(rng, filter(!=(i), cset))
                    at = (name, trial, i, j)
                    e_i_new = _geo_propose(rng, x[i], step)
                    Mx = _cmcref_M(x, w, act)
                    e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(x, zx, H, rc, Mx, i, j,
                                                           e_i_new, β, scratch)
                    _geo_note!(v[:pure], x == x0 && zx == zx0, at)
                    isnull = ℓ == -Inf && isnan(ΔE) && e_j_new == x[j]
                    ref, s2 = _geo_ref_partner(x[i], e_i_new, x[j], w[i], w[j], n̂)
                    if ref === nothing
                        s2 > 1 + 1e-9 || continue           # ambiguous at the edge
                        _geo_note!(v[:null_s2], isnull, at)
                        n_null_s2 += 1
                        continue
                    end
                    s2 < 1 - 1e-9 || continue
                    yref = copy(x)
                    yref[i] = e_i_new
                    yref[j] = ref
                    Mpar_ref = dot(_cmcref_M(yref, w, act), n̂)
                    if Mpar_ref < -1e-9
                        _geo_note!(v[:null_M], isnull, at)
                        n_null_M += 1
                        continue
                    end
                    Mpar_ref > 1e-9 || continue
                    n_valid += 1
                    _geo_note!(v[:finite], isfinite(ℓ), at)
                    isfinite(ℓ) || continue
                    # the partner: frame reference, unit, in the domain, sign kept.
                    # e∥ = √(1 − s2) amplifies a rounding error of s2 by 1/(2|e∥|).
                    tol_e = 1e-13 + 1e-15 / abs(dot(ref, n̂))
                    _geo_note!(v[:ref], norm(e_j_new - ref) < tol_e, at)
                    _geo_note!(v[:unit], abs(norm(e_j_new) - 1) <= 1e-15, at)
                    _geo_note!(v[:domain], maximum(abs, e_j_new) <= 1, at)
                    zj, zj′ = dot(x[j], n̂), dot(e_j_new, n̂)
                    _geo_note!(v[:sign], (zj >= 0) == (zj′ >= 0), at)
                    n_negz += zj < 0
                    # ℓ against the independent formula
                    y = copy(x)
                    y[i] = e_i_new
                    y[j] = e_j_new
                    Mpar, Mpar′ = dot(Mx, n̂), dot(_cmcref_M(y, w, act), n̂)
                    dE = total_energy(H, y) - total_energy(H, x)
                    ℓref = -β * dE + 2 * log(Mpar′ / Mpar) + log(abs(zj)) -
                           log(abs(zj′))
                    _geo_note!(v[:ΔE], abs(ΔE - dE) <= 1e-12, at)
                    _geo_note!(v[:ℓ], abs(ℓ - ℓref) <= 1e-9, at)
                    n_jac += abs(log(Mpar′ / Mpar)) > 1e-3
                    n_zr += abs(log(abs(zj / zj′))) > 1e-3
                    # reversibility: from y, moving i back to e_i returns e_j, and
                    # ℓ(y→x) = −ℓ(x→y). (The antisymmetry alone cannot catch a wrong
                    # Jacobian exponent — any ℓ of the form f(y) − f(x) passes; the
                    # comparison with ℓref above does.)
                    zy = MC._zrows(H, y)
                    e_back, ΔE_back, ℓ_back = MC._cmc_pair_proposal(
                        y, zy, H, rc, _cmcref_M(y, w, act), i, j, x[i], β, scratch)
                    _geo_note!(v[:rev_finite], isfinite(ℓ_back), at)
                    _geo_note!(v[:rev_e], norm(e_back - x[j]) <
                                      1e-13 + 1e-15 / abs(zj) + 1e-15 / abs(zj′), at)
                    _geo_note!(v[:rev_ℓ], abs(ℓ_back + ℓ) <= 1e-9, at)
                    _geo_note!(v[:rev_ΔE], abs(ΔE_back + ΔE) <= 1e-12, at)
                end
                @_geo_test_tallies v
            end
        end
        # non-vacuity: every branch and every ℓ term was exercised. Measured (seed
        # 4402): 3698 valid, 1118 null by s2, 36 null by M∥′, 869 with e_j∥ < 0;
        # max |ℓ − ℓref| = 1.1e-13 (bound 1e-9), max |ΔE − dE| = 2.2e-16, max
        # |e_j′ − ref| = 1.3e-13, max reverse |e_j″ − e_j| = 2.8e-13.
        @test n_valid >= 1_000
        @test n_null_s2 >= 50
        @test n_null_M >= 20
        @test n_negz >= 100
        @test n_jac >= 500 && n_zr >= 500
    end

    @testset "G4: invalid states and edge cases" begin
        f2g = MC.TiledHamiltonian(2, _f2g_terms())
        f3 = MC.TiledHamiltonian(3, _f3_terms())
        rng = MersenneTwister(4403)
        rc = MC._resolve_constraint(f2g, MC.MagnetizationConstraint(n̂))
        scratch = MC._CMCScratch(f2g)
        x = _geo_valid_state(rng, 2, ones(2), trues(2), n̂)
        zx = MC._zrows(f2g, x)
        Mx = _cmcref_M(x, ones(2), trues(2))
        # NaN anywhere in the proposal or the partner → null, never a throw
        e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(x, zx, f2g, rc, Mx, 1, 2,
                                               SVector(NaN, 0.0, 0.0), 1.0, scratch)
        @test ℓ == -Inf && isnan(ΔE) && e_j_new == x[2]
        xn = copy(x)
        xn[2] = SVector(NaN, NaN, NaN)
        e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(xn, zx, f2g, rc, Mx, 1, 2,
                                               _geo_propose(rng, x[1], 0.1), 1.0,
                                               scratch)
        @test ℓ == -Inf && isnan(ΔE)
        # a current M with M·n̂ ≤ 0 (or NaN) is a caller error
        @test_throws ArgumentError MC._cmc_pair_proposal(
            x, zx, f2g, rc, -Mx, 1, 2, _geo_propose(rng, x[1], 0.1), 1.0, scratch)
        @test_throws ArgumentError MC._cmc_pair_proposal(
            x, zx, f2g, rc, SVector(NaN, 0.0, 0.0), 1, 2, x[1], 1.0, scratch)

        # a partner with e_j·n̂ = 0 exactly: ℓ = −Inf with a finite ΔE, σ_j = +1
        rz = MC._resolve_constraint(f3, MC.MagnetizationConstraint([0.0, 0.0, 1.0]))
        s3 = MC._CMCScratch(f3)
        x3 = MC.SpinConfig([SVector(1.0, 0.0, 0.0), SVector(-0.6, 0.0, 0.8),
                            SVector(-0.4, 0.0, sqrt(0.84))])
        M3 = _cmcref_M(x3, ones(3), trues(3))
        @test M3[1] == 0 && M3[2] == 0 && M3[3] > 0
        e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(x3, MC._zrows(f3, x3), f3, rz, M3, 2, 1,
                                               _geo_propose(rng, x3[2], 0.05), 1.0, s3)
        @test ℓ == -Inf && isfinite(ΔE) && e_j_new[3] > 0

        # |component| ≤ 1 for e_j′: proposals aimed at e_j′ = ê exactly, for ê = x̂
        # and ê = ẑ (the harmonics read only the z component; a checkpoint restore
        # checks all three). From x = (n̂, n̂) (uniform F2g), the source move
        # e_i′ = −ê⊥ + (ê·n̂) n̂ makes the partner e_j′ = ê⊥ + (ê·n̂) n̂ = ê. Over
        # many directions n̂ the UN-normalized construction overshoots 1 in some
        # cases; the proposal must neither throw nor return a component above 1.
        for (ax, ê) in ((1, SVector(1.0, 0.0, 0.0)), (3, SVector(0.0, 0.0, 1.0)))
            n_over = 0
            n_done = 0
            vz = _geo_tallies((:finite, :domain, :target))
            for trial = 1:2_000
                m = _rand_spin(rng)
                m = m[ax] < 0 ? -m : m
                rm = MC._resolve_constraint(f2g, MC.MagnetizationConstraint(m))
                mh = rm.direction
                xm = MC.SpinConfig([mh, mh])
                eperp = ê - mh[ax] * mh
                e_i_new = normalize(-eperp + mh[ax] * mh)
                p = -(e_i_new - dot(e_i_new, mh) * mh)          # test-side raw build
                raw = p + sqrt(1 - dot(p, p)) * mh
                n_over += maximum(abs, raw) > 1
                e_j_new, ΔE, ℓ = MC._cmc_pair_proposal(xm, MC._zrows(f2g, xm), f2g,
                                                       rm, 2 * mh, 1, 2, e_i_new, 1.0,
                                                       scratch)
                _geo_note!(vz[:finite], isfinite(ℓ), (ax, trial))
                isfinite(ℓ) || continue
                n_done += 1
                _geo_note!(vz[:domain], maximum(abs, e_j_new) <= 1, (ax, trial))
                _geo_note!(vz[:target], norm(e_j_new - ê) < 1e-12, (ax, trial))
            end
            @_geo_test_tallies vz
            # the regime is real: the raw construction overshoots 1 in ~13 % of the
            # cases (257 of 2000 for x̂ with this seed)
            @test n_over > 0 && n_done == 2_000
        end
    end
end

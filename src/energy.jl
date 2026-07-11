# The 4-function energy contract between `TiledHamiltonian` and everything else:
# `total_energy` / `site_coeffs!` / `delta_energy` / `site_gradient`. Updates and
# observables touch the energy only through these.
#
# The kernels are the site-generalized siblings of SCETools' `mc/metropolis.jl`
# (`_accumulate_site_term!` / `_term_energy`): the same `μ = idx − l − 1` index
# mapping, the same rank-specialized function barriers over the rank-erased `folded`
# tensors, contracted against concrete tesseral rows — here columns of a dense
# `nlm × n_sites` matrix, with the member slot precomputed in the CSR adjacency
# instead of a `findfirst`.

# Tabulate the full tesseral row Z_lm(e), l = 0:lmax, into `z` (ordered by
# `Harmonics.lm_index`, which is sequential in this loop order). `e` must be unit.
function _zlm_row!(z::AbstractVector{Float64}, e::SVector{3,Float64},
                   lmax::Int)::AbstractVector{Float64}
    i = 0
    @inbounds for l = 0:lmax, m = -l:l
        i += 1
        z[i] = Harmonics.Zlm_unsafe(l, m, e)
    end
    return z
end

# Fresh `nlm × n_sites` tesseral-row matrix of a configuration (column s = Z_lm(e_s)).
function _zrows(H::TiledHamiltonian, config::SpinConfig)::Matrix{Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    zrows = Matrix{Float64}(undef, H.nlm, H.n_sites)
    for s = 1:H.n_sites
        _zlm_row!(view(zrows, :, s), config[s], H.lmax)
    end
    return zrows
end

# One instance's full contraction against the concrete site columns of `zrows`
# (rank-specialized barrier; `sites` is that instance's slice of `inst_sites`).
@inline function _instance_energy(coef::Float64, sites::AbstractVector{Int32},
                                  ls::Vector{Int}, folded::Array{Float64,D},
                                  zrows::Matrix{Float64})::Float64 where {D}
    E = 0.0
    @inbounds for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            μk = idx[k] - ls[k] - 1
            p *= zrows[Harmonics.lm_index(ls[k], μk), sites[k]]
        end
        E += w * p
    end
    return coef * E
end

# Total energy from precomputed tesseral rows (the incremental-state entry point).
function _total_energy(H::TiledHamiltonian, zrows::Matrix{Float64})::Float64
    E = 0.0
    @inbounds for i in eachindex(H.inst_term)
        term = H.terms[H.inst_term[i]]
        sites = view(H.inst_sites, H.inst_ptr[i]:(H.inst_ptr[i + 1] - 1))
        E += _instance_energy(term.coef, sites, term.ls, term.folded, zrows)
    end
    return E
end

"""
    total_energy(H::TiledHamiltonian, config::SpinConfig) -> Float64

The SCE energy of `config` on the tiled supercell, in the model's energy units with
the intercept `j0` excluded: the sum of every cluster instance's contraction
`coef · Σ_μ folded[μ] ∏ᵢ Z_{lᵢμᵢ}(e_{siteᵢ})`. On the training cell
(`dims = (1,1,1)`) this equals `predict_energy(model, config) − intercept(model)`.
"""
total_energy(H::TiledHamiltonian, config::SpinConfig)::Float64 =
    _total_energy(H, _zrows(H, config))

"""
    site_coeffs!(c, H::TiledHamiltonian, s::Integer, zrows) -> c

Leave-one-out coefficient vector of site `s`: accumulate into `c` (length `H.nlm`,
**not** zeroed here) the coefficient of `Z_lm(e_s)` from every instance touching `s`,
contracting each template's `folded` against the concrete tesseral columns of the
*other* member sites. Because every cluster's sites are distinct (constructor
invariant), `c` is independent of `e_s`, so the site energy is exactly `c · Z(e_s)`
and a single-spin move has the exact energy change
[`delta_energy`](@ref)`(c, Z(e_s), Z(e_s′))` — any body order, no linearization.
β never enters; `c` is in the model's energy units.
"""
function site_coeffs!(c::Vector{Float64}, H::TiledHamiltonian, s::Integer,
                      zrows::Matrix{Float64})::Vector{Float64}
    @inbounds for j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
        i = H.site_inst[j]
        term = H.terms[H.inst_term[i]]
        _accumulate_instance!(c, Int(H.site_slot[j]), term.coef,
                              Int(H.inst_ptr[i]) - 1, H.inst_sites, term.ls,
                              term.folded, zrows)
    end
    return c
end

# One instance's contribution to the coefficient-of-Z_lm at member position `slot`
# (rank-specialized barrier; `off` is the instance's offset into `inst_sites`).
@inline function _accumulate_instance!(c::Vector{Float64}, slot::Int, coef::Float64,
                                       off::Int, inst_sites::Vector{Int32},
                                       ls::Vector{Int}, folded::Array{Float64,D},
                                       zrows::Matrix{Float64}) where {D}
    @inbounds for idx in CartesianIndices(folded)
        w = coef * folded[idx]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            k == slot && continue
            μk = idx[k] - ls[k] - 1
            p *= zrows[Harmonics.lm_index(ls[k], μk), inst_sites[off + k]]
        end
        p == 0.0 && continue
        μi = idx[slot] - ls[slot] - 1
        c[Harmonics.lm_index(ls[slot], μi)] += w * p
    end
    return c
end

"""
    delta_energy(c, zold, znew) -> Float64

Exact energy change of a single-spin move whose leave-one-out coefficients are `c`
(from [`site_coeffs!`](@ref)) and whose old/new tesseral rows are `zold`/`znew`:
`Σ_k c_k (znew_k − zold_k)`. β-free — the caller applies the Boltzmann factor.
"""
function delta_energy(c::Vector{Float64}, zold::AbstractVector{Float64},
                      znew::AbstractVector{Float64})::Float64
    ΔE = 0.0
    @inbounds for k in eachindex(c)
        ck = c[k]
        ck == 0.0 && continue
        ΔE += ck * (znew[k] - zold[k])
    end
    return ΔE
end

"""
    site_gradient(H::TiledHamiltonian, s::Integer, config::SpinConfig)
        -> SVector{3,Float64}

On-sphere (tangent-projected) gradient of the total energy with respect to the spin
direction of site `s`: `∇E = Σ_k c_k ∇Z_k(e_s)` with the leave-one-out coefficients
of [`site_coeffs!`](@ref) and `SCEFitting.Harmonics.grad_Zlm_unsafe` (so
`e_s · ∇E = 0`). Diagnostics/tests — not on the sweep hot path.
"""
function site_gradient(H::TiledHamiltonian, s::Integer,
                       config::SpinConfig)::SVector{3,Float64}
    zrows = _zrows(H, config)
    c = site_coeffs!(zeros(H.nlm), H, s, zrows)
    e = config[s]
    g = zero(SVector{3,Float64})
    i = 0
    for l = 0:H.lmax, m = -l:l
        i += 1
        ck = c[i]
        ck == 0.0 && continue
        g += ck * Harmonics.grad_Zlm_unsafe(l, m, e)
    end
    return g
end

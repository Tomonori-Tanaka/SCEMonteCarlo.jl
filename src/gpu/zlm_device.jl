# Device-safe tesseral-harmonic row — a deliberate, bitwise-faithful reimplementation
# of the host path `_zlm_row!` → `Harmonics.Zlm_unsafe` → `LegendrePolynomials.dnPl`
# (decision record docs/specs/gpu-prototype.md G4; gate: the zlm section of
# test/unit/test_gpu.jl).
#
# Why it exists: the upstream `dnPl` path cannot compile inside a GPU kernel (its
# `checklength`/`_checkvalues` throws carry runtime strings, and `no_offset_view`
# is a type-unstable wrapper). Everything value-relevant, however, is pure scalar
# arithmetic — so this file replicates the exact operation ORDER of
#   LegendrePolynomials `_unsafednPl!` / `doublefactorial` / `Pl_recursion` /
#   `dPl_recursion` (v0.4), `Harmonics._plm_norm` / `_parity` / `Zlm_unsafe`,
#   and `Base.power_by_squaring` (the `ComplexF64^n` route),
# dropping only the value-neutral wrappers. COUPLED SITES: any upstream change to
# those functions (or to `_zlm_row!`'s loop order) breaks the bitwise gate loudly —
# update this file with it.

@inline _zlm_parity(n::Int)::Int = isodd(n) ? -1 : 1

# Harmonics._plm_norm, verbatim: √((2l+1)/(4π) · (l−m)!/(l+m)!) without factorials.
@inline function _zlm_plm_norm(l::Int, m::Int)::Float64
    acc = (2 * l + 1) / (4π)
    for i = (l - m + 1):(l + m)
        acc /= i
    end
    return sqrt(acc)
end

# Value path of Base.power_by_squaring(x, p) for p ≥ 1 with mul = * — replicated
# because the p < 0 branch of the Base function throws with a runtime string.
# Gate: `_zlm_cpow(z, n) == z^n` for n = 1:6 over a dense grid.
@inline function _zlm_cpow(x::ComplexF64, p::Int)::ComplexF64
    xsq = x * x
    p == 1 && return x
    p == 2 && return xsq
    t = trailing_zeros(p) + 1
    p >>= t
    if (t -= 1) > 0
        x = xsq
    end
    while (t -= 1) > 0
        x = x * x
    end
    y = x
    while p > 0
        t = trailing_zeros(p) + 1
        p >>= t
        while (t -= 1) >= 0
            x = x * x
        end
        y = y * x
    end
    return y
end

"""
    _zlm_dnpl(x, l, n, cache) -> Float64

`dⁿPₗ(x)/dxⁿ`, the operation-order-faithful device replica of
`LegendrePolynomials._unsafednPl!` (assumes `0 ≤ n ≤ l`, which `|m| ≤ l`
guarantees; `cache` needs length ≥ `l − n + 1` — `LMAX + 1` covers every call).
"""
@inline function _zlm_dnpl(x::Float64, l::Int, n::Int,
                           cache::MVector{N,Float64})::Float64 where {N}
    if n == l
        # doublefactorial(Float64, 2l − 1): descending odd products
        p = 1.0
        for i = (2 * l - 1):-2:1
            p *= Float64(i)
        end
        cache[1] = p
    else
        # collectPl!: cache[i] = P_{i−1}(x) for i = 1:(l − n + 1), Bonnet recursion
        Pl = 1.0
        Plm1 = 0.0
        cache[1] = Pl
        for ℓ = 1:(l - n)
            Pl, Plm1 = ((2 * ℓ - 1) * x * Pl - (ℓ - 1) * Plm1) / ℓ, Pl
            cache[ℓ + 1] = Pl
        end
        # derivative lifts: the three dPl_recursion forms, in _unsafednPl!'s order
        for ni = 1:n
            pnn = (2 * ni - 1) * cache[1]
            cache[1] = pnn
            ℓ = ni + 1
            cache[2] = ((2 * ℓ - 1) * (x * pnn + ni * cache[2])) / ℓ
            for li = (ni + 2):min(l, l - n + ni)
                cache[li - ni + 1] = ((2 * li - 1) * (x * cache[li - ni] +
                                                      ni * cache[li - ni + 1]) -
                                      (li - 1) * cache[li - ni - 1]) / li
            end
        end
    end
    return cache[l - n + 1]
end

# Harmonics.Zlm_unsafe, verbatim (with the complex power routed through _zlm_cpow).
@inline function _zlm_device(l::Int, m::Int, u::SVector{3,Float64},
                             cache::MVector{N,Float64})::Float64 where {N}
    n = abs(m)
    plm = _zlm_parity(n) * _zlm_plm_norm(l, n) * _zlm_dnpl(u[3], l, n, cache)
    n == 0 && return plm
    c = _zlm_parity(n) * sqrt(2.0) * plm
    zpow = _zlm_cpow(ComplexF64(u[1], u[2]), n)
    return m > 0 ? c * real(zpow) : c * imag(zpow)
end

"""
    _zlm_row_device!(z, u, ::Val{LMAX}) -> nothing

Fill `z[1:(LMAX+1)²]` with the tesseral row `Z_{lm}(u)` in `lm_index` order —
the device replica of `_zlm_row!` (bitwise-identical by design; gated). `z` may
be any writable `Float64` vector (an `MVector`, or a `@localmem` view inside a
kernel). `LMAX` is a compile-time value so the recursion cache is a stack
`MVector` of static size.
"""
@inline function _zlm_row_device!(z::AbstractVector{Float64}, u::SVector{3,Float64},
                                  ::Val{LMAX})::Nothing where {LMAX}
    cache = MVector{LMAX + 1,Float64}(undef)
    i = 0
    for l = 0:LMAX, m = -l:l
        i += 1
        @inbounds z[i] = _zlm_device(l, m, u, cache)
    end
    return nothing
end

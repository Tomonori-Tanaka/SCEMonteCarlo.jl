"""Sanity checks for CMC oracle formulas (vectorized over many independent chains).

Frame: constraint direction is z (n-frame); the anisotropy axis k(theta) =
(-sin th, 0, cos th) so that the lab easy axis z_lab sits at angle th from M-hat.
Rotation axis u = y.  Claim: <u.T> = -dF/dth, T = sum_i e_i x (-grad_i E).
"""
import numpy as np, time
rng = np.random.default_rng(1)
t0 = time.time()

def rand_unit(shape):
    v = rng.standard_normal(shape + (3,))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

# ---------------- model -----------------
def energy(e, th, J, K, lam):
    k = np.array([-np.sin(th), 0.0, np.cos(th)])
    p = e @ k                                  # (..., N)
    E = -K * np.sum(p**2, axis=-1)
    N = e.shape[-2]
    if J != 0.0:
        for a in range(N):
            for b in range(a + 1, N):
                E = E + J * np.sum(e[..., a, :] * e[..., b, :], axis=-1)
    if lam != 0.0:
        E = E + lam * np.prod(p, axis=-1)      # anisotropic N-body product term
    return E

def torque_u(e, th, J, K, lam):
    # u.T with T = sum e x (-grad E); isotropic J gives zero total torque.
    k = np.array([-np.sin(th), 0.0, np.cos(th)])
    p = e @ k
    g = -2 * K * p[..., None] * k              # grad of -K p^2
    if lam != 0.0:
        N = e.shape[-2]
        for a in range(N):
            others = np.prod(np.delete(p, a, axis=-1), axis=-1)
            g[..., a, :] += lam * others[..., None] * k
    T = np.cross(e, -g)
    return np.sum(T[..., 1], axis=-1)

# ---------------- rotated-iid reference -----------------
def rotated_iid(N, nsamp):
    e = rand_unit((nsamp, N))
    M = e.sum(axis=1)
    m = M / np.linalg.norm(M, axis=1, keepdims=True)
    z = np.array([0, 0, 1.0])
    ax = np.cross(m, z)
    s = np.linalg.norm(ax, axis=1)
    c = m[:, 2]
    ax = ax / s[:, None]
    # Rodrigues: rotate every spin about ax by angle atan2(s, c)
    cr = np.cross(ax[:, None, :], e)
    dt = np.sum(ax[:, None, :] * e, axis=-1, keepdims=True)
    return c[:, None, None] * e + s[:, None, None] * cr + \
        (1 - c[:, None, None]) * dt * ax[:, None, :]

def ref_avg(N, th, beta, J, K, lam, nsamp=2_000_000):
    e = rotated_iid(N, nsamp)
    w = np.exp(-beta * energy(e, th, J, K, lam))
    w /= w.mean()
    tq = torque_u(e, th, J, K, lam)
    Mz = e.sum(axis=1)[:, 2]
    return (w * tq).mean(), (w * Mz).mean(), (w * Mz**2).mean()

# ---------------- N = 2 quadrature -----------------
def quad_F(th, beta, J, K, lam, nc=400, nphi=400):
    xs, ws = np.polynomial.legendre.leggauss(nc)
    c = 0.5 * (xs + 1); wc = 0.5 * ws
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    C, P = np.meshgrid(c, phi, indexing="ij")
    S = np.sqrt(1 - C**2)
    e1 = np.stack([S * np.cos(P), S * np.sin(P), C], -1)
    e2 = 2 * C[..., None] * np.array([0, 0, 1.0]) - e1
    e = np.stack([e1, e2], -2)
    f = 4 * C * np.exp(-beta * energy(e, th, J, K, lam))
    Z = np.sum(wc[:, None] * f) * (2 * np.pi / nphi)
    return -np.log(Z) / beta

# ---------------- vectorized CMC -----------------
def cmc(N, th, beta, J, K, lam, nchain=4000, nstep=3000, nburn=1000, sig=0.5,
        a=2.0, b=1.0, naive_pair=False):
    e = np.tile(np.array([0, 0, 1.0]), (nchain, N, 1))
    acc_tq = 0.0; acc_mz = 0.0; acc_mz2 = 0.0; cnt = 0
    idx = np.arange(nchain)
    E = energy(e, th, J, K, lam)
    for step in range(nstep):
        i = rng.integers(0, N, nchain)
        j = (i + rng.integers(1, N, nchain)) % N
        ei = e[idx, i]; ej = e[idx, j]
        prop = ei + sig * rng.standard_normal((nchain, 3))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        xy = ej[:, :2] + ei[:, :2] - prop[:, :2]
        r2 = np.sum(xy**2, axis=1)
        ok = r2 < 1.0
        zj = np.sign(ej[:, 2]) * np.sqrt(np.clip(1 - r2, 0, None))
        ejn = np.column_stack([xy, zj])
        Mz = e.sum(axis=1)[:, 2]
        Mzn = Mz - ei[:, 2] - ej[:, 2] + prop[:, 2] + zj
        ok &= Mzn > 0
        en = e.copy()
        en[idx, i] = prop; en[idx, j] = ejn
        En = energy(en, th, J, K, lam)
        if naive_pair:
            # BUG model: dE_i and dE_j both evaluated against the OLD partner
            ea = e.copy(); ea[idx, i] = prop
            eb = e.copy(); eb[idx, j] = ejn
            dE = (energy(ea, th, J, K, lam) - E) + (energy(eb, th, J, K, lam) - E)
        else:
            dE = En - E
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.exp(-beta * dE) * (Mzn / Mz)**a * \
                (np.abs(ej[:, 2]) / np.abs(zj))**b
        accp = ok & (rng.random(nchain) < np.where(ok, ratio, 0.0))
        e[accp] = en[accp]; E = np.where(accp, En, E)
        if step >= nburn:
            acc_tq += torque_u(e, th, J, K, lam).mean()
            Mz = e.sum(axis=1)[:, 2]
            acc_mz += Mz.mean(); acc_mz2 += (Mz**2).mean(); cnt += 1
    return acc_tq / cnt, acc_mz / cnt, acc_mz2 / cnt

# ===== 1. E = 0 geometry gate (N=2, N=3) =====
for N in (2, 3):
    ref = rotated_iid(N, 2_000_000).sum(axis=1)[:, 2]
    print(f"N={N} E=0 ref  <Mz>={ref.mean():.4f}  <Mz^2>={np.mean(ref**2):.4f}")
    for (a, b, lab) in [(2, 1, "correct"), (0, 0, "no Jacobian"), (1, 1, "exp 2->1"),
                        (2, -1, "|z| inverted"), (2, 0, "|z| dropped"),
                        (1, 0, "exp1 & z dropped")]:
        tq, mz, mz2 = cmc(N, 0.0, 1.0, 0.0, 0.0, 0.0, nchain=2000, nstep=1500,
                          nburn=300, a=a, b=b)
        print(f"   {lab:18s} <Mz>={mz:.4f} <Mz^2>={mz2:.4f}")
print("t =", time.time() - t0)

# ===== 2. N=2 dimer + uniaxial: quadrature vs reference vs CMC =====
J, K, beta, th = -1.0, 0.5, 2.0, np.pi / 5
h = 1e-4
dF = (quad_F(th + h, beta, J, K, 0) - quad_F(th - h, beta, J, K, 0)) / (2 * h)
print("N=2 quadrature  -dF/dth =", -dF)
print("N=2 rotated-iid <uT>    =", ref_avg(2, th, beta, J, K, 0)[0])
print("N=2 CMC         <uT>    =", cmc(2, th, beta, J, K, 0)[0])
print("N=2 CMC no-Jac  <uT>    =", cmc(2, th, beta, J, K, 0, a=0, b=0)[0])
print("t =", time.time() - t0)

# ===== 3. N=3 with a 3-body anisotropic product: naive pair dE =====
J, K, lam, beta, th = -0.5, 0.3, 1.0, 2.0, np.pi / 5
print("N=3 rotated-iid <uT>, <Mz> =", ref_avg(3, th, beta, J, K, lam)[:2])
print("N=3 CMC              =", cmc(3, th, beta, J, K, lam)[:2])
print("N=3 CMC naive pair dE=", cmc(3, th, beta, J, K, lam, naive_pair=True)[:2])
print("t =", time.time() - t0)

# Mutation sensitivity at larger N: all-to-all ferro E = -(Jt/N)(R^2-N)/2 + single-ion
# uniaxial -K sum p^2 (axis at angle th from constraint z). Ordered regime.
import numpy as np, time
rng = np.random.default_rng(7)
def run(N, beta, Jt, K, th, a, b, nchain=400, nsweep=600, nburn=200):
    k = np.array([-np.sin(th), 0, np.cos(th)])
    e = np.tile([0, 0, 1.0], (nchain, N, 1)); idx = np.arange(nchain)
    def E(e):
        M = e.sum(1); p = e @ k
        return -(Jt / N) * (np.sum(M * M, 1) - N) / 2 - K * np.sum(p * p, 1)
    En = E(e); acc = []; tq = []
    for s in range(nsweep * N):
        i = rng.integers(0, N, nchain); j = (i + rng.integers(1, N, nchain)) % N
        ei = e[idx, i]; ej = e[idx, j]
        pr = ei + 0.4 * rng.standard_normal((nchain, 3)); pr /= np.linalg.norm(pr, axis=1)[:, None]
        xy = ej[:, :2] + ei[:, :2] - pr[:, :2]; r2 = (xy**2).sum(1); ok = r2 < 1
        zj = np.sign(ej[:, 2]) * np.sqrt(np.clip(1 - r2, 0, None))
        Mz = e.sum(1)[:, 2]; Mzn = Mz - ei[:, 2] - ej[:, 2] + pr[:, 2] + zj; ok &= Mzn > 0
        en = e.copy(); en[idx, i] = pr; en[idx, j] = np.column_stack([xy, zj])
        E2 = E(en)
        with np.errstate(all="ignore"):
            ratio = np.exp(-beta * (E2 - En)) * (Mzn / Mz)**a * (np.abs(ej[:, 2]) / np.abs(zj))**b
        ac = ok & (rng.random(nchain) < np.where(ok, ratio, 0))
        e[ac] = en[ac]; En = np.where(ac, E2, En)
        if s >= nburn * N and s % N == 0:
            p = e @ k
            acc.append(e.sum(1)[:, 2].mean() / N)
            # u.T for single-ion: sum e x (2K p k), y-component
            T = np.cross(e, 2 * K * p[..., None] * k).sum(1)[:, 1]
            tq.append(T.mean() / N)
    return np.mean(acc), np.mean(tq), np.std(tq) / np.sqrt(len(tq)) 
t0 = time.time()
for N in (32,):
    for (a, b, lab) in [(2, 1, "correct"), (0, 0, "no Jacobian"), (0, 1, "Mz^2 dropped"), (2, 0, "|z| dropped")]:
        m, t, dt = run(N, 2.0, 3.0, 0.1, np.pi / 4, a, b, nsweep=400, nburn=100)
        print(f"N={N} {lab:14s} m={m:.4f}  tau/N={t:.4f} (+-{dt:.4f} naive)")
print("t", time.time() - t0)

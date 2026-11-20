# Sanity check of the CMC Jacobian (Asselin 2010 App. A) and torque identity (App. C)
# N=3 non-interacting spins, E = -K sum (c.S_k)^2, beta=1. Constraint: M = sum m_k S_k || n.
import numpy as np, math, random, sys
K, beta, th = 2.0, 1.0, 0.7
a = beta*K
n  = np.array([math.sin(th), 0.0, math.cos(th)])
e1 = np.array([math.cos(th), 0.0, -math.sin(th)]); e2 = np.array([0.0, 1.0, 0.0])

def iid_spins(rng, size):
    out = np.empty((0,)); us = []
    need = size
    while need > 0:
        u = rng.uniform(-1, 1, 3*need)
        acc = rng.uniform(0, 1, 3*need) < np.exp(a*(u*u-1))
        us.append(u[acc]); need -= acc.sum()
    u = np.concatenate(us)[:size]
    ph = rng.uniform(0, 2*np.pi, size); s = np.sqrt(1-u*u)
    return np.stack([s*np.cos(ph), s*np.sin(ph), u], -1)   # crystal frame, c = z

def oracle(m, eps=0.1, chunks=12, csize=1_000_000, seed=1):
    rng = np.random.default_rng(seed); obs = []; thetas = []
    for _ in range(chunks):
        S = iid_spins(rng, 3*csize).reshape(csize, 3, 3)
        M = (m[None,:,None]*S).sum(1); R = np.linalg.norm(M, axis=1); Mh = M/R[:,None]
        if len(m) == 3 and np.allclose(m, 1): thetas.append(np.arccos(np.clip(Mh[:,2], -1, 1)))
        sel = Mh @ n > math.cos(eps)
        Ss = S[sel]; cz = Ss[..., 2]
        # torque T = -sum S x dH/dS, dH/dS = -2K (c.S) c  -> T = 2K sum (c.S) S x c ; phi-comp = T . e2
        Sxc_y = Ss[..., 2]*0 - Ss[..., 0]*1.0   # (S x z)_y = S_z*0 - S_x*1 = -S_x
        Tphi = (2*K*cz*Sxc_y).sum(1)
        obs.append(np.stack([R[sel], (cz**2).sum(1), Tphi], -1))
    obs = np.concatenate(obs)
    return obs.mean(0), obs.std(0)/math.sqrt(len(obs)), len(obs), (np.concatenate(thetas) if thetas else None)

def cmc(m, variant, nmoves=150_000, sigma=0.6, seed=2):
    random.seed(seed)
    c = np.array([n@[0,0,1.0]*0 + e1[2], e2[2], n[2]])     # crystal z in constraint frame
    cx, cy, cz_ = float(e1[2]), float(e2[2]), float(n[2])
    S = [[0.0, 0.0, 1.0] for _ in range(3)]
    Mz = sum(m)
    acc = []; N = 3
    def E(s): d = cx*s[0]+cy*s[1]+cz_*s[2]; return -K*d*d
    k = 0; samples = []
    while k < nmoves:
        i = random.randrange(N); j = random.randrange(N-1); j = j if j < i else j+1
        si, sj = S[i], S[j]
        v = [random.gauss(0, sigma) for _ in range(3)]
        t = [si[0]+si[1]*v[2]-si[2]*v[1], si[1]+si[2]*v[0]-si[0]*v[2], si[2]+si[0]*v[1]-si[1]*v[0]]
        nt = math.sqrt(t[0]**2+t[1]**2+t[2]**2); sin_ = [x/nt for x in t]
        r = m[i]/m[j]
        jx = sj[0] - r*(sin_[0]-si[0]); jy = sj[1] - r*(sin_[1]-si[1])
        q = 1 - jx*jx - jy*jy
        ok = q > 0
        if ok:
            jz = math.copysign(math.sqrt(q), sj[2]); sjn = [jx, jy, jz]
            Mzn = Mz + m[i]*(sin_[2]-si[2]) + m[j]*(jz-sj[2])
            ok = Mzn > 0
        if not ok:
            if variant == 'retry': continue      # WRONG: does not count the null move
            k += 1; samples.append(obs(S, m, Mz, c)) if k % 5 == 0 else None; continue
        dE = E(sin_)+E(sjn)-E(si)-E(sj)
        jac = abs(sj[2])/abs(jz)
        if variant != 'noMz2': jac *= (Mzn/Mz)**2
        if random.random() < jac*math.exp(-beta*dE):
            S[i], S[j], Mz = sin_, sjn, Mzn
        k += 1
        if k % 5 == 0: samples.append(obs(S, m, Mz, c))
    A = np.array(samples)
    nb = 50; B = A[:len(A)//nb*nb].reshape(nb, -1, 3).mean(1)
    return A.mean(0), B.std(0)/math.sqrt(nb)

def obs(S, m, Mz, c):
    # in constraint frame; crystal z = c ; T_phi = T . e2, e2 = (0,1,0) in constraint frame
    tot_c2 = 0.0; Tphi = 0.0
    for s in S:
        d = c[0]*s[0]+c[1]*s[1]+c[2]*s[2]
        tot_c2 += d*d
        # (S x c)_y = s_z c_x - s_x c_z
        Tphi += 2*K*d*(s[2]*c[0]-s[0]*c[2])
    return (Mz, tot_c2, Tphi)

for label, m in [('equal m', np.array([1.0, 1.0, 1.0])), ('weighted m', np.array([1.0, 0.4, 1.5]))]:
    mu, se, cnt, thetas = oracle(m)
    print(f'[{label}] oracle (iid + cone eps=0.1, n={cnt}):  <|M|>={mu[0]:.4f}+-{se[0]:.4f}  <sum cz^2>={mu[1]:.4f}+-{se[1]:.4f}  <T_phi>={mu[2]:.4f}+-{se[2]:.4f}')
    for var in ['paper', 'noMz2', 'retry']:
        mu2, se2 = cmc(list(m), var)
        print(f'   CMC {var:6s}: <|M|>={mu2[0]:.4f}+-{se2[0]:.4f}  <sum cz^2>={mu2[1]:.4f}+-{se2[1]:.4f}  <T_phi>={mu2[2]:.4f}+-{se2[2]:.4f}')
    if thetas is not None:
        # F(theta) = -kT ln P(theta) per solid angle;  check  T_phi = -dF/dtheta at th
        h = 0.06
        def lnP(t0):
            cntb = np.count_nonzero(np.abs(thetas - t0) < 0.03)
            return math.log(cntb / (2*math.pi*math.sin(t0)*0.06))
        dF = -(lnP(th+h) - lnP(th-h))/(2*h)/beta
        print(f'   histogram: -dF/dtheta at theta={th} = {-dF:.4f} (compare <T_phi>)')

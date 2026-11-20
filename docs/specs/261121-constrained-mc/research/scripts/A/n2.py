# N=2 exact constrained partition function by 2D quadrature vs iid histogram of M-direction
import numpy as np, math
K, beta = 2.0, 1.0; a = beta*K
def Z2(th, m1, m2, ng=600):
    # constraint frame: n = z'; crystal axis c = (-sin th, 0, cos th)
    cx, cz = -math.sin(th), math.cos(th)
    t = (np.arange(ng)+0.5)*math.pi/ng; p = (np.arange(2*ng)+0.5)*math.pi/ng
    T, P = np.meshgrid(t, p, indexing='ij'); w = np.sin(T)*(math.pi/ng)**2
    s1 = np.stack([np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.cos(T)])
    s2p = -(m1/m2)*s1[:2]; q = 1 - (s2p**2).sum(0); ok = q > 0; qq = np.sqrt(np.where(ok, q, 0))
    tot = 0.0
    for sg in (+1, -1):
        s2z = sg*qq; Mz = m1*s1[2] + m2*s2z
        good = ok & (Mz > 0) & (qq > 1e-12)
        E = -K*((cx*s1[0]+cz*s1[2])**2 + (cx*s2p[0]+cz*s2z)**2)
        f = np.where(good, Mz**2/(m2**2*np.where(good, qq, 1))*np.exp(-beta*E), 0.0)
        tot += (f*w).sum()
    return tot
rng = np.random.default_rng(5)
def iid(size):
    us=[]; need=size
    while need>0:
        u=rng.uniform(-1,1,2*need); acc=rng.uniform(0,1,2*need)<np.exp(a*(u*u-1)); us.append(u[acc]); need-=acc.sum()
    u=np.concatenate(us)[:size]; ph=rng.uniform(0,2*np.pi,size); s=np.sqrt(1-u*u)
    return np.stack([s*np.cos(ph),s*np.sin(ph),u],-1)
th, h = 0.7, 0.06
for m1, m2 in [(1.0,1.0),(1.0,0.4)]:
    dF_quad = -(math.log(Z2(th+h,m1,m2))-math.log(Z2(th-h,m1,m2)))/(2*h)/beta
    cnt_p=cnt_m=0
    for _ in range(10):
        S1=iid(2_000_000); S2=iid(2_000_000); M=m1*S1+m2*S2
        ct=M[:,2]/np.linalg.norm(M,axis=1); tt=np.arccos(np.clip(ct,-1,1))
        cnt_p+=np.count_nonzero(np.abs(tt-(th+h))<0.03); cnt_m+=np.count_nonzero(np.abs(tt-(th-h))<0.03)
    dF_hist = -(math.log(cnt_p/math.sin(th+h))-math.log(cnt_m/math.sin(th-h)))/(2*h)/beta
    print(f'm=({m1},{m2}): dF/dtheta quadrature={dF_quad:.4f}  histogram={dF_hist:.4f} (+-~{2/math.sqrt(min(cnt_p,cnt_m))/(2*h):.3f})')

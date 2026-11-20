import sys,numpy as np
rng=np.random.default_rng(7)
w=np.array([float(x) for x in sys.argv[1].split(',')])
e=rng.normal(size=(2_000_000,len(w),3)); e/=np.linalg.norm(e,axis=-1,keepdims=True)
M=np.einsum('k,nkd->nd',w,e); R=np.linalg.norm(M,axis=-1); mh=M/R[:,None]
c=np.einsum('nkd,nd->nk',e,mh)*np.sign(w)
print(sys.argv[1],'iid <R>=%.4f Pneg=%.4f <P2>=%.4f'%(R.mean(),(c<0).mean(),(0.5*(3*c**2-1)).mean()))

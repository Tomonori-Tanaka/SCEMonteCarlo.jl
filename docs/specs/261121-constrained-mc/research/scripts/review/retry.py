import numpy as np, sys
rng=np.random.default_rng(3)
def unit(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
def rot(e,ax,th):
    c=np.cos(th)[:,None]; s=np.sin(th)[:,None]
    return c*e+s*np.cross(ax,e)+(1-c)*np.sum(ax*e,-1,keepdims=True)*ax
n=np.array([np.sin(.6)*np.cos(.35),np.sin(.6)*np.sin(.35),np.cos(.6)])
w=np.array([float(x) for x in sys.argv[1].split(',')]); N=len(w); retry=sys.argv[2]=='retry'
step=float(sys.argv[3]); nch=2000; nsw=450
e=np.tile(np.sign(w)[None,:,None]*n,(nch,1,1)); ar=np.arange(nch)
R=[];neg=[]
for sw in range(nsw):
    for i in range(N):
        todo=np.ones(nch,bool)
        for it in range(200 if retry else 1):
            idx=np.where(todo)[0]
            if len(idx)==0: break
            m=len(idx)
            k=rng.integers(0,N-1,size=m); j=np.where(k>=i,k+1,k)
            ei=e[idx,i]; ej=e[idx,j]
            ei2=rot(ei,unit(rng.normal(size=(m,3))),step*rng.normal(size=m))
            pi,pi2=ei@n,ei2@n; pj=ej@n
            ejp2=(ej-pj[:,None]*n)-(w[i]/w[j])[:,None]*((ei2-pi2[:,None]*n)-(ei-pi[:,None]*n))
            s2=np.sum(ejp2**2,-1); ok=s2<1
            pj2=np.where(pj>=0,1.,-1.)*np.sqrt(np.clip(1-s2,0,None)); ej2=ejp2+pj2[:,None]*n
            M=np.sum(w*(e[idx]@n),-1); M2=M+w[i]*(pi2-pi)+w[j]*(pj2-pj); ok&=M2>0
            with np.errstate(all='ignore'):
                l=2*np.log(M2/M)+np.log(np.abs(pj))-np.log(np.abs(pj2))
            l=np.where(ok,l,-np.inf); acc=np.log(rng.random(m))<l
            e[idx[acc],i]=ei2[acc]; e[idx[acc],j[acc]]=ej2[acc]
            todo[idx[ok]]=False   # retry only geometric failures
    if sw>=150:
        M=np.einsum('k,nkd->nd',w,e); Rn=np.linalg.norm(M,axis=-1)
        R.append(Rn.mean()); neg.append(((np.sign(w)*(e@n))<0).mean())
R=np.array(R);neg=np.array(neg)
se=lambda x:x.reshape(30,-1).mean(1).std(ddof=1)/np.sqrt(30)
print(sys.argv[1:],'<R>=%.4f±%.4f Pneg=%.4f±%.4f'%(R.mean(),se(R),neg.mean(),se(neg)))

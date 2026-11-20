# Independent vectorized CMC (E=0 or single-ion K) to test closed forms and mutation visibility.
import numpy as np, sys
rng = np.random.default_rng(1)
def unit(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
def rot(e, ax, th):
    c=np.cos(th)[:,None]; s=np.sin(th)[:,None]
    return c*e + s*np.cross(ax,e) + (1-c)*np.sum(ax*e,-1,keepdims=True)*ax
def run(w, n, mode="ok", nch=4000, nsw=1500, step=0.8, K=0.0, beta=1.0, cax=np.array([0,0,1.])):
    w=np.asarray(w,float); N=len(w)
    e=np.tile(np.sign(w)[None,:,None]*n[None,None,:],(nch,1,1)).astype(float)
    def Eon(x): return -K*np.sum(x@cax,-1) if False else -K*(x@cax)**2
    R=[];neg=[];P2=[];Tq=[]
    ar=np.arange(nch)
    for sw in range(nsw):
        for i in range(N):
            # partner uniform from others
            k=rng.integers(0,N-1,size=nch); j=np.where(k>=i,k+1,k)
            ei=e[:,i]; ej=e[ar,j]
            ei2=rot(ei,unit(rng.normal(size=(nch,3))),step*rng.normal(size=nch))
            pi=(ei@n); pi2=(ei2@n)
            eip=ei-pi[:,None]*n; eip2=ei2-pi2[:,None]*n
            pj=ej@n; ejp=ej-pj[:,None]*n
            r=(w[i]/w[j])[:,None]
            if mode=="abs": r=np.abs(r)
            ejp2=ejp - r*(eip2-eip)
            s2=np.sum(ejp2**2,-1)
            ok=s2<1
            sg=np.where(pj>=0,1.0,-1.0)
            if mode=="zplus": sg=np.ones_like(sg)
            pj2=sg*np.sqrt(np.clip(1-s2,0,None))
            ej2=ejp2+pj2[:,None]*n
            M=np.sum(w[None,:]*(e@n),-1)
            M2=M+w[i]*(pi2-pi)+w[j]*(pj2-pj)
            ok&=M2>0
            dE=(-K*pi2**2 if False else 0)  # placeholder
            dE=(-K*((ei2@cax)**2-(ei@cax)**2) -K*((ej2@cax)**2-(ej@cax)**2))
            with np.errstate(divide='ignore',invalid='ignore'):
                if mode=="exp1noz": l=-beta*dE+np.log(M2/M)
                elif mode=="noM2": l=-beta*dE+np.log(np.abs(pj))-np.log(np.abs(pj2))
                elif mode=="noz": l=-beta*dE+2*np.log(M2/M)
                else: l=-beta*dE+2*np.log(M2/M)+np.log(np.abs(pj))-np.log(np.abs(pj2))
            l=np.where(ok,l,-np.inf)
            acc=np.log(rng.random(nch))<l
            if mode=="retry":
                pass
            e[acc,i]=ei2[acc]; e[ar[acc],j[acc]]=ej2[acc]
        if sw>=150:
            M=np.sum(w[None,:,None]*e,1); Rn=np.linalg.norm(M,axis=-1)
            R.append(Rn.mean()); neg.append(((np.sign(w)[None,:]*(e@n))<0).mean())
            mh=M/Rn[:,None]; c=(e@mh.T[:, :1].T) if False else np.einsum('cnk,ck->cn',e,mh)
            P2.append((0.5*(3*c**2-1)).mean())
            G=-2*K*(e@cax)[...,None]*cax  # unprojected ok for cross
            T=np.sum(np.cross(G,e),1); Tq.append(T.mean(0))
    R=np.array(R);neg=np.array(neg);P2=np.array(P2)
    def se(x):
        b=x.reshape(30,-1).mean(1); return b.std(ddof=1)/np.sqrt(len(b))
    return R.mean(),se(R),neg.mean(),se(neg),P2.mean(),se(P2),np.array(Tq).mean(0)
th,ph=0.6,0.35
n=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
cases=[([1,1],"N2 eq"),([1,1,1],"N3 eq"),([1,2],"N2 w(1,2)"),([1,-2],"N2 w(1,-2)"),([1,1,-1],"N3 w(1,1,-1)"),([1,0.4,1.5],"N3 w(1,.4,1.5)")]
for w,name in cases:
    for mode in ["ok","exp1noz","noz","zplus","abs"]:
        if mode=="abs" and min(w)>0: continue
        r=run(w,n,mode,nch=2000,nsw=450)
        print(f"{name:14s} {mode:8s} <R>={r[0]:.4f}±{r[1]:.4f} Pneg={r[2]:.4f}±{r[3]:.4f} <P2>={r[4]:.4f}±{r[5]:.4f}",flush=True)

import sys, numpy as np
exec(open('cmc_mut.py').read().split('th,ph=0.6')[0])
n=np.array([np.sin(.6)*np.cos(.35),np.sin(.6)*np.sin(.35),np.cos(.6)])
w=[float(x) for x in sys.argv[1].split(',')]
for mode in sys.argv[2].split(','):
    r=run(w,n,mode,nch=2000,nsw=450)
    print(f"{sys.argv[1]:12s} {mode:8s} <R>={r[0]:.4f}±{r[1]:.4f} Pneg={r[2]:.4f}±{r[3]:.4f} <P2>={r[4]:.4f}±{r[5]:.4f}",flush=True)

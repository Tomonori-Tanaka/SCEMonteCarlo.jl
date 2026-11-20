import math, importlib.util, sys
spec = importlib.util.spec_from_file_location("n2", "n2.py")
src = open("n2.py").read().split("rng = np.random")[0]
ns = {}; exec(src, ns); Z2 = ns['Z2']
th, h = 0.7, 0.06
for ng in (300, 600, 1200):
    a = -(math.log(Z2(th+h,1.0,0.4,ng))-math.log(Z2(th-h,1.0,0.4,ng)))/(2*h)
    # swapped roles: S_2 free (moment 0.4), S_1 compensator (moment 1.0)
    b = -(math.log(Z2(th+h,0.4,1.0,ng))-math.log(Z2(th-h,0.4,1.0,ng)))/(2*h)
    print(ng, f'S1 free: {a:.4f}   S2 free: {b:.4f}')

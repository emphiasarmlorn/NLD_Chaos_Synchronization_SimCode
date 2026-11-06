import numpy as np, enss, lyap
enss.HANLE=6.0; lyap.enss.HANLE=6.0; lyap.S0=1.0; S0=1.0; F0=0.162933; eps=0.55
d=np.load('lyap_H0.55.npz'); r=d['r']; lam=d['lam']; i=lam.argmax(); rs=r[i]; print('peak lambda',lam[i],'at r',rs)
# robustness: finer step & different d0 at r*, plus neighbours
for dt,d0 in ((0.02,1e-6),(0.01,1e-7)):
    v=lyap.lyap(F0/np.array([rs,rs+0.004,rs-0.004]),eps,T=2400.0,dt=dt,d0=d0,renorm=int(0.5/dt)); print('dt',dt,'d0',d0,'lam',np.round(v,4),flush=True)
# D2 from scalar S_z
from scipy.spatial.distance import pdist
cases={'locked 1/2 (r=0.54)':0.54,'unlocked, lambda~0 (r=0.62)':0.62,f'chaotic (r={rs:.3f})':rs}
fm=np.array([F0/v for v in cases.values()])
x,dt=enss.run(fm,S0=S0,eps=eps,T=2200.0,dt=0.02); x=x[:,int(300/dt):]; tau=12
for (nm,_),s in zip(cases.items(),x):
    s=s-s.mean(); s/=s.std(); res=[]
    for m in (2,3,4,5,6,7,8):
        N=len(s)-(m-1)*tau; Y=np.stack([s[k*tau:k*tau+N] for k in range(m)],1)[::6]; dd=pdist(Y); rr=np.logspace(-1.6,-0.4,14)*np.sqrt(m)
        C=np.array([(dd<q).mean() for q in rr]); ok=C>0; res.append(np.polyfit(np.log(rr[ok]),np.log(C[ok]),1)[0])
    print(nm,'D2(m=2..8)',np.round(res,2),flush=True)

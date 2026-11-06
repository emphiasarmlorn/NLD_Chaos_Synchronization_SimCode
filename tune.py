import numpy as np, sys, json, enss
def f0_of(H,S0):
    enss.HANLE=H
    x,dt=enss.run([1.0],S0=S0,eps=0.0,T=500,dt=0.025); s=x[0][int(120/dt):]; s=s-s.mean()
    if np.ptp(s)<1e-3: return None
    n=len(s)*8; F=np.abs(np.fft.rfft(s*np.hanning(len(s)),n)); f=np.fft.rfftfreq(n,dt)
    m=(f>0.08)&(f<0.24)
    if F[m].max()<0.3*F.max(): return None
    return float(f[m][F[m].argmax()])
def metrics(H,S0,f0,eps,npts=276):
    enss.HANLE=H
    r=np.linspace(0.45,1.0,npts); fm=f0/r
    x,dt=enss.run(fm,S0=S0,eps=eps,T=800.0,dt=0.025); x=x[:,int(200/dt):]
    fx=np.linspace(0,1.2,1200); fdom=np.zeros(npts); osc=np.zeros(npts)
    for i in range(npts):
        s=x[i]-x[i].mean(); n=len(s)*4; sp=np.abs(np.fft.rfft(s*np.hanning(len(s)),n)); fr=np.fft.rfftfreq(n,dt)/fm[i]
        m=(fr>0.03)&(fr<0.96); j=sp[m].argmax(); fdom[i]=fr[m][j]
        drive=sp[(fr>0.97)&(fr<1.03)].max(); osc[i]=sp[m].max()/(drive+1e-12)
    dr=r[1]-r[0]
    def width(p):
        lock=np.abs(fdom-p)<0.006; i0=np.argmin(np.abs(r-p))
        if not lock[i0]: 
            # allow nearby
            k=np.where(lock&(np.abs(r-p)<0.03))[0]
            if len(k)==0: return 0.0
            i0=k[np.argmin(np.abs(r[k]-p))]
        a=i0
        while a>0 and lock[a-1]: a-=1
        b=i0
        while b<npts-1 and lock[b+1]: b+=1
        return (b-a+1)*dr
    alive=osc>0.02
    # edge of 1:1: smallest r after which oscillation is dead for all larger r
    dead_from=npts
    for i in range(npts-1,-1,-1):
        if alive[i]: break
        dead_from=i
    return dict(w12=width(0.5),w23=width(2/3),w34=width(0.75),w35=width(0.6),w45=width(0.8),edge11=float(r[min(dead_from,npts-1)]) if dead_from<npts else 1.0)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='grid':
        out=[]
        for H in map(float,sys.argv[2].split(',')):
            for S0 in np.arange(0.4,3.01,0.2):
                f=f0_of(H,float(S0)); out.append((H,float(S0),f)); print(H,round(S0,2),f,flush=True)
        json.dump(out,open(sys.argv[3],'w'))
    else:
        H,S0,f0=map(float,sys.argv[2:5]); res={}
        for eps in map(float,sys.argv[5].split(',')):
            m=metrics(H,S0,f0,eps); res[eps]=m; print(H,S0,f0,eps,{k:round(v,3) for k,v in m.items()},flush=True)
        json.dump(res,open(sys.argv[6],'w'))

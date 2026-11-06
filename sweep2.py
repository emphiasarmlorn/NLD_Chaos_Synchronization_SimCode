import numpy as np, sys, enss
H,S0,f0,eps,tag=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
lo,hi,n=(float(sys.argv[6]),float(sys.argv[7]),int(sys.argv[8])) if len(sys.argv)>8 else (0.45,1.0,500)
enss.HANLE=H; r=np.linspace(lo,hi,n); fm=f0/r
x,dt=enss.run(fm,S0=S0,eps=eps,T=1400.0,dt=0.02); x=x[:,int(200/dt):]
fx=np.linspace(0,1.05,420); P=np.zeros((n,len(fx)))
for i in range(n):
    s=x[i]-x[i].mean(); nf=len(s)*2; sp=np.abs(np.fft.rfft(s*np.hanning(len(s)),nf)); fr=np.fft.rfftfreq(nf,dt)/fm[i]; P[i]=np.interp(fx,fr,sp)
np.savez(f'res_{tag}.npz',r=r,fx=fx,P=P,f0=f0,eps=eps,H=H,S0=S0)

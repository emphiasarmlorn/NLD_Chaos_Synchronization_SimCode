"""Max Lyapunov exponent of the driven ODE (Benettin two-trajectory method), batched over fm."""
import numpy as np, sys, os, enss
enss.HANLE=float(os.environ.get("HW","12")); S0=float(os.environ.get("S0","1.5"))
def lyap(fm, eps, T=1600.0, Ttr=300.0, dt=0.02, renorm=25, d0=1e-6):
    fm=np.atleast_1d(fm).astype(float); M=len(fm)
    def f(t,BN):
        return -(BN-enss.aS(enss.S_of(BN,enss.s0_t(t,S0,eps,fm))))/enss.TN
    def step(t,BN):
        k1=f(t,BN);k2=f(t+dt/2,BN+dt/2*k1);k3=f(t+dt/2,BN+dt/2*k2);k4=f(t+dt,BN+dt*k3)
        return BN+dt/6*(k1+2*k2+2*k3+k4)
    A=np.tile([0.3,0.2,0.1],(M,1)); t=0.0
    for i in range(int(Ttr/dt)): A=step(t,A); t+=dt
    rng=np.random.default_rng(0); v=rng.normal(size=(M,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    Bv=A+d0*v; acc=np.zeros(M); n=int(T/dt)
    for i in range(1,n+1):
        A=step(t,A); Bv=step(t,Bv); t+=dt
        if i%renorm==0:
            d=Bv-A; nd=np.linalg.norm(d,axis=1); acc+=np.log(nd/d0); Bv=A+d0*d/nd[:,None]
    return acc/(n//renorm*renorm*dt)
if __name__=='__main__':
    eps=float(sys.argv[1]); lo,hi,n=float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]); f0=float(os.environ.get("F0","0.16770833"))
    r=np.linspace(lo,hi,n); lam=lyap(f0/r,eps); np.savez(f'lyap_{sys.argv[5]}.npz',r=r,lam=lam)
    print('done',lam.max(),(lam>0.01).sum())

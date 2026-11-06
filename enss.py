"""ENSS driven-CTC model (Greilich et al., arXiv:2406.06243, Eqs.1-3; tensor a from arXiv:2303.15989).
Batched RK4: many drive frequencies integrated in parallel."""
import numpy as np

AL = np.deg2rad(10.0)
BX, BZ = -1.0, 0.176          # mT, B_ext (tilt ~10 deg)
BEXT = np.array([BX, 0.0, BZ])
AN, BN_, TN = 20.0, 21.0, 0.5  # mT, mT, s
HANLE = 0.1                    # mT, hbar/(mu_B g Ts)
SGN = 1.0                      # sign of g-factor term (g<0 -> -1); tested below
B2 = BEXT @ BEXT

def S_of(BN, s0):
    """Stationary Bloch solution. BN: (M,3), s0: (M,) polarization along z."""
    b = SGN * (BEXT + BN) / HANLE
    S0 = np.zeros_like(BN); S0[:, 2] = s0
    bb = np.sum(b * b, 1, keepdims=True)
    bS0 = np.sum(b * S0, 1, keepdims=True)
    cr = np.cross(b, S0)
    return (S0 + cr + b * bS0) / (1 + bb)

def aS(S):
    return BN_ * (S @ BEXT)[:, None] * BEXT[None, :] / B2 + AN * S

def s0_t(t, S0, eps, fm):
    return 0.5 * S0 * (2 - eps * (1 - np.cos(2 * np.pi * fm * t)))

def run(fm, S0=0.5, eps=0.5, T=1200.0, dt=0.02, Ttrans=0.0, BN0=None, stride=5):
    fm = np.atleast_1d(np.asarray(fm, float)); M = len(fm)
    BN = np.tile([0.3, 0.2, 0.1], (M, 1)) if BN0 is None else BN0.copy()
    def f(t, BN):
        s0 = s0_t(t, S0, eps, fm)
        return -(BN - aS(S_of(BN, s0))) / TN
    n = int(round(T / dt)); out = []
    t = 0.0
    for i in range(n):
        k1 = f(t, BN); k2 = f(t + dt/2, BN + dt/2*k1)
        k3 = f(t + dt/2, BN + dt/2*k2); k4 = f(t + dt, BN + dt*k3)
        BN = BN + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
        if i % stride == 0:
            out.append(S_of(BN, s0_t(t, S0, eps, fm))[:, 2])
    return np.array(out).T, dt * stride   # (M, nt) S_z signal, sampling interval

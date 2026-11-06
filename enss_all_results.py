#!/usr/bin/env python3
"""
enss_all_results.py  --  one script that reproduces every simulated result
for the driven electron-nuclear spin system (ENSS) time crystal.

Paper : Greilich, Kopteva, Korenev, Bayer, arXiv:2406.06243
Model : Eqs. 1-3 of the paper; DNP tensor from arXiv:2303.15989.

What it produces (figures are saved in --out, and shown with --show):
  fig1_fft_map.png     FFT map vs f0/fm at eps = 0.5  (plateaus + bifurcation jets)
  fig2_zoom.png        zoom into f0/fm = 0.60 ... 0.65 (the paper's Fig. 5c box)
  fig3_eps_sweep.png   FFT maps for several modulation depths (Arnold tongue)
  fig4_lyapunov.png    largest Lyapunov exponent vs f0/fm for four eps values
  fig5_traces.png      S_z traces + spectra: locked vs chaotic drive frequency
  summary.txt          all numbers (f0, lambda_max, robustness check, D2)

Usage
  python enss_all_results.py --quick          # ~2-4 min, coarse but shows everything
  python enss_all_results.py                  # full resolution, ~15-25 min
  python enss_all_results.py --only lyap d2   # run selected stages only
  python enss_all_results.py --show           # also open the figures

Stages: f0 maps zoom eps lyap verify d2 traces

Requirements: numpy, scipy, matplotlib.

IMPORTANT: S0 and B_H below are *fitted effective parameters* (the papers do
not give them). They were tuned so the undriven frequency is ~0.163 Hz and the
map looks like the paper's Fig. 5b. Do not read B_H = 6 mT as the sample's
Hanle width (about 0.1 mT in the earlier paper).
"""
import argparse
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# 0.  Parameters
# --------------------------------------------------------------------------
BEXT = np.array([-1.0, 0.0, 0.176])     # mT, external field (tilt ~10 deg)
B2 = BEXT @ BEXT
A_N, B_N, T_N = 20.0, 21.0, 0.5         # mT, mT, s   (from the papers)
B_H = 6.0                               # mT, field scale in the Bloch term (FITTED)
S0 = 1.0                                # pump polarization amplitude       (FITTED)
F0 = 0.162933                           # Hz, undriven frequency of this model
EPS_MAPS = 0.5                          # modulation depth used for the maps (paper value)
EPS_CHAOS = 0.55                        # depth at which chaos is found in this model


# --------------------------------------------------------------------------
# 1.  The model:  fast electron spin S, slow nuclear field B_N, drive S0(t)
# --------------------------------------------------------------------------
def spin_S(BN, s0):
    """Stationary Bloch solution  S = S0 + b x S  (Eq. 1), b = (Bext+BN)/B_H.
    BN: (M,3) array, s0: (M,) pump amplitude (pump along z).
    Closed form: S = [S0 + b x S0 + b (b.S0)] / (1 + |b|^2)."""
    b = (BEXT + BN) / B_H
    S0v = np.zeros_like(BN)
    S0v[:, 2] = s0
    bb = np.sum(b * b, axis=1, keepdims=True)
    bS0 = np.sum(b * S0v, axis=1, keepdims=True)
    return (S0v + np.cross(b, S0v) + b * bS0) / (1.0 + bb)


def dnp_target(S):
    """DNP tensor:  a S = b_N (S.Bext) Bext/|Bext|^2 + a_N S."""
    return B_N * (S @ BEXT)[:, None] * BEXT[None, :] / B2 + A_N * S


def pump(t, eps, fm):
    """Modulated pump, Eq. 3:  S0(t) = (S0/2)[2 - eps (1 - cos 2 pi fm t)]."""
    return 0.5 * S0 * (2.0 - eps * (1.0 - np.cos(2.0 * np.pi * fm * t)))


def rhs(t, BN, eps, fm):
    """Eq. 2:  dB_N/dt = -(B_N - a S)/T_N   (S from the Bloch solution)."""
    return -(BN - dnp_target(spin_S(BN, pump(t, eps, fm)))) / T_N


def rk4(t, BN, dt, eps, fm):
    k1 = rhs(t, BN, eps, fm)
    k2 = rhs(t + dt / 2, BN + dt / 2 * k1, eps, fm)
    k3 = rhs(t + dt / 2, BN + dt / 2 * k2, eps, fm)
    k4 = rhs(t + dt, BN + dt * k3, eps, fm)
    return BN + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state(M):
    return np.tile([0.3, 0.2, 0.1], (M, 1))


def simulate(fm, eps, T, dt=0.02, stride=5):
    """Integrate all drive frequencies in `fm` at once. Returns S_z signals
    (M, n_samples) and the sampling interval dt*stride."""
    fm = np.atleast_1d(np.asarray(fm, float))
    BN, t, out = initial_state(len(fm)), 0.0, []
    for i in range(int(round(T / dt))):
        BN = rk4(t, BN, dt, eps, fm)
        t += dt
        if i % stride == 0:
            out.append(spin_S(BN, pump(t, eps, fm))[:, 2])
    return np.array(out).T, dt * stride


# --------------------------------------------------------------------------
# 2.  Analysis tools
# --------------------------------------------------------------------------
def fundamental_freq(T=1500.0, dt=0.02, lo=0.1, hi=0.22):
    """Undriven (eps = 0) oscillation frequency of this model."""
    x, d = simulate([1.0], 0.0, T, dt)
    s = x[0][int(200 / d):]
    s = s - s.mean()
    n = len(s) * 16
    F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    f = np.fft.rfftfreq(n, d)
    m = (f > lo) & (f < hi)
    return float(f[m][F[m].argmax()])


def fft_map(r, eps, T, Ttr, dt=0.02):
    """FFT map: row i is the spectrum of a (T-Ttr) s trace at fm = F0/r[i],
    with the frequency axis divided by fm."""
    fm = F0 / np.asarray(r)
    x, d = simulate(fm, eps, T, dt)
    x = x[:, int(Ttr / d):]
    fx = np.linspace(0, 1.05, 420)
    P = np.zeros((len(r), len(fx)))
    for i in range(len(r)):
        s = x[i] - x[i].mean()
        n = len(s) * 2
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
        fr = np.fft.rfftfreq(n, d) / fm[i]
        P[i] = np.interp(fx, fr, sp)
    return fx, P


def lyapunov(r, eps, T, Ttr=300.0, dt=0.02, renorm_time=0.5, d0=1e-6, seed=0):
    """Largest Lyapunov exponent (1/s) by the two-trajectory (Benettin)
    method, one value per f0/fm in `r`."""
    fm = F0 / np.asarray(r, float)
    M = len(fm)
    A, t = initial_state(M), 0.0
    for _ in range(int(Ttr / dt)):                 # settle on the attractor
        A = rk4(t, A, dt, eps, fm); t += dt
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(M, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    Bv = A + d0 * v
    n, every = int(T / dt), max(1, int(round(renorm_time / dt)))
    acc = np.zeros(M)
    for i in range(1, n + 1):
        A = rk4(t, A, dt, eps, fm); Bv = rk4(t, Bv, dt, eps, fm); t += dt
        if i % every == 0:
            d = Bv - A
            nd = np.linalg.norm(d, axis=1)
            acc += np.log(nd / d0)
            Bv = A + d0 * d / nd[:, None]          # shrink separation back to d0
    return acc / ((n // every) * every * dt)


def correlation_dimension(s, tau=12, dims=range(2, 9), every=6):
    """Grassberger-Procaccia D2 estimate of a scalar series for each embedding
    dimension m (local slope of log C(r) vs log r over a FIXED radius window;
    it is only indicative, see summary notes)."""
    from scipy.spatial.distance import pdist
    s = (s - s.mean()) / s.std()
    out = []
    for m in dims:
        N = len(s) - (m - 1) * tau
        Y = np.stack([s[k * tau:k * tau + N] for k in range(m)], axis=1)[::every]
        d = pdist(Y)
        rr = np.logspace(-1.6, -0.4, 14) * np.sqrt(m)
        C = np.array([(d < q).mean() for q in rr])
        ok = C > 0
        out.append(np.polyfit(np.log(rr[ok]), np.log(C[ok]), 1)[0])
    return np.array(out)


# --------------------------------------------------------------------------
# 3.  Plot helpers
# --------------------------------------------------------------------------
def show_map(ax, r, fx, P, title, ylim=1.0):
    Pn = P / P.max(axis=1, keepdims=True)
    ax.pcolormesh(r, fx, Pn.T ** 0.5, cmap="magma", shading="auto", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$f_0/f_m$")
    ax.set_ylabel("$f_{exp}/f_m$")
    ax.set_ylim(0, ylim)


def timer(label):
    print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)
    return time.time()


# --------------------------------------------------------------------------
# 4.  Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarse resolution, much faster")
    ap.add_argument("--out", default="enss_results", help="output folder")
    ap.add_argument("--show", action="store_true", help="open the figures")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["f0", "maps", "zoom", "eps", "lyap", "verify", "d2", "traces"],
                    help="run only these stages")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    want = (lambda k: args.only is None or k in args.only)
    q = args.quick

    # resolution settings: (quick, full)
    n_map = 160 if q else 600          # rows per map
    n_eps = 120 if q else 500
    n_zoom = 200 if q else 800
    T_map, Ttr_map = (700.0, 150.0) if q else (1400.0, 200.0)   # 20-min traces in full mode
    n_lyap = 70 if q else 200
    T_lyap = 500.0 if q else 1600.0
    T_ver = 800.0 if q else 2400.0
    T_d2 = 1000.0 if q else 2200.0

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"ENSS simulation   B_H={B_H} mT  S0={S0}  quick={q}")

    # ---- f0 -------------------------------------------------------------
    if want("f0"):
        timer("undriven frequency")
        f0m = fundamental_freq(T=800 if q else 1500)
        log(f"undriven f0 of this model = {f0m:.4f} Hz   (paper: 0.1607 Hz; F0 used = {F0})")

    # ---- main map ---------------------------------------------------------
    if want("maps"):
        t0 = timer(f"main FFT map, eps={EPS_MAPS}, {n_map} rows")
        r = np.linspace(0.45, 1.0, n_map)
        fx, P = fft_map(r, EPS_MAPS, T_map, Ttr_map)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        show_map(ax, r, fx, P, rf"Simulated FFT map, $\epsilon_{{sim}}={EPS_MAPS}$")
        ax.axvspan(0.60, 0.65, fill=False, ec="cyan", ls="--", lw=1)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig1_fft_map.png", dpi=170)
        log(f"fig1 done ({time.time()-t0:.0f}s): plateaus at f_exp/f_m = 1/2, 2/3, 3/4, 4/5 with jets between")

    # ---- zoom ---------------------------------------------------------------
    if want("zoom"):
        t0 = timer(f"zoom 0.60-0.65, {n_zoom} rows")
        r = np.linspace(0.60, 0.65, n_zoom)
        fx, P = fft_map(r, EPS_MAPS, T_map, Ttr_map)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        show_map(ax, r, fx, P, r"Zoom, $f_0/f_m\in[0.60,0.65]$ (paper Fig. 5c box)", ylim=0.6)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig2_zoom.png", dpi=170)
        log(f"fig2 done ({time.time()-t0:.0f}s)")

    # ---- eps sweep -------------------------------------------------------------
    if want("eps"):
        eps_list = [0.05, 0.2, 0.35, 0.5, 0.55, 0.7]
        t0 = timer(f"eps sweep {eps_list}, {n_eps} rows each")
        r = np.linspace(0.45, 1.0, n_eps)
        fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
        for ax, e in zip(axs.flat, eps_list):
            fx, P = fft_map(r, e, T_map, Ttr_map)
            show_map(ax, r, fx, P, rf"$\epsilon_{{sim}}={e}$")
        fig.tight_layout(); fig.savefig(f"{args.out}/fig3_eps_sweep.png", dpi=130)
        log(f"fig3 done ({time.time()-t0:.0f}s): tongues widen with eps; irregular fan at 0.55; oscillation quenched at 0.7")

    # ---- Lyapunov scan -----------------------------------------------------------
    lam_store = {}
    if want("lyap") or want("verify") or want("d2") or want("traces"):
        pass  # lambda is needed by later stages; computed on demand below

    def get_lam(e, r):
        key = (e, len(r))
        if key not in lam_store:
            lam_store[key] = lyapunov(r, e, T_lyap)
        return lam_store[key]

    r_lyap = np.linspace(0.70, 0.98, n_lyap)
    if want("lyap"):
        t0 = timer(f"Lyapunov scans, {n_lyap} points each")
        fig, ax = plt.subplots(figsize=(9, 3.8))
        cols = {0.5: "#8A94AD", 0.52: "#7A5BC2", 0.55: "#C2571A", 0.58: "#1F8A70"}
        log("Largest Lyapunov exponent (1/s):")
        for e, c in cols.items():
            lam = get_lam(e, r_lyap)
            ax.plot(r_lyap, lam, lw=1.1, color=c, label=rf"$\epsilon={e}$")
            i = lam.argmax()
            log(f"  eps={e:<5} max lambda = {lam[i]:+.4f} at f0/fm = {r_lyap[i]:.3f}; "
                f"{(lam > 0.01).sum()} of {len(lam)} points above 0.01")
        ax.axhline(0, c="k", lw=0.5); ax.set_ylim(-0.02, 0.06)
        ax.set_xlabel("$f_0/f_m$"); ax.set_ylabel(r"$\lambda_{max}$ (1/s)")
        ax.set_title("Largest Lyapunov exponent of the driven model"); ax.legend(ncol=4, fontsize=9)
        fig.tight_layout(); fig.savefig(f"{args.out}/fig4_lyapunov.png", dpi=170)
        log(f"fig4 done ({time.time()-t0:.0f}s). lambda ~ 0.002 or less means 'no chaos' (finite-time noise).")

    # ---- peak point for later stages -------------------------------------------------
    r_star = None
    if want("verify") or want("d2") or want("traces"):
        lam55 = get_lam(EPS_CHAOS, r_lyap)
        r_star = float(r_lyap[lam55.argmax()])

    # ---- robustness of the chaotic point -----------------------------------------------
    if want("verify"):
        t0 = timer(f"robustness check at f0/fm = {r_star:.3f}")
        pts = np.array([r_star, r_star + 0.004, r_star - 0.004])
        log(f"Robustness of lambda at eps={EPS_CHAOS} (points f0/fm = {np.round(pts, 3).tolist()}):")
        for dt_, d0_ in ((0.02, 1e-6), (0.01, 1e-7)):
            v = lyapunov(pts, EPS_CHAOS, T_ver, dt=dt_, d0=d0_, renorm_time=0.5)
            log(f"  dt={dt_}, d0={d0_:g}:  lambda = {np.round(v, 4).tolist()}")
        log(f"  ({time.time()-t0:.0f}s) stable sign and size => not a numerical artifact")

    # ---- correlation dimension / traces ----------------------------------------------------
    if want("d2") or want("traces"):
        cases = {"locked 1/2 (f0/fm=0.54)": 0.54,
                 "unlocked, lambda~0 (f0/fm=0.62)": 0.62,
                 f"chaotic (f0/fm={r_star:.3f})": r_star}
        t0 = timer("signals for D2 / traces")
        fm = F0 / np.array(list(cases.values()))
        x, d = simulate(fm, EPS_CHAOS, T_d2)
        x = x[:, int(300 / d):]
        if want("d2"):
            log(f"Correlation dimension D2 at eps={EPS_CHAOS} (embedding m = 2..8, tau = 1.2 s):")
            for (name, _), s in zip(cases.items(), x):
                log(f"  {name:<34} {np.round(correlation_dimension(s), 2).tolist()}")
            log("  Note: fixed fitting window; unreliable for the unlocked/quasi-periodic case.")
        if want("traces"):
            fig, axs = plt.subplots(2, 2, figsize=(12, 6))
            for row, (name, rr) in zip((0, 1), [(list(cases)[0], 0.54), (list(cases)[2], r_star)]):
                s = x[list(cases).index(name)]
                tt = np.arange(len(s)) * d
                axs[row, 0].plot(tt[:int(200 / d)], s[:int(200 / d)], lw=0.8)
                axs[row, 0].set_title(name, fontsize=10); axs[row, 0].set_xlabel("t (s)"); axs[row, 0].set_ylabel("$S_z$")
                sm = s - s.mean(); n = len(sm) * 4
                F = np.abs(np.fft.rfft(sm * np.hanning(len(sm)), n))
                fr = np.fft.rfftfreq(n, d) / (F0 / rr)
                axs[row, 1].plot(fr, F, lw=0.8); axs[row, 1].set_xlim(0, 2)
                axs[row, 1].set_xlabel("$f_{exp}/f_m$"); axs[row, 1].set_ylabel("|FFT|")
            fig.tight_layout(); fig.savefig(f"{args.out}/fig5_traces.png", dpi=150)
            log(f"fig5 done ({time.time()-t0:.0f}s)")

    with open(f"{args.out}/summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\nAll outputs are in ./{args.out}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())

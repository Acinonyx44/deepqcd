"""
DeepQCD for a volatility-regime change in financial returns: our extension of the paper's Sec. 5 recipe to a
finance setting (the paper does a mean shift in AR(1); here it is a variance shift in GARCH(1,1)).

Daily returns with volatility clustering, standardized so the calm regime has unit unconditional variance:

    r_t       = sigma_t z_t,                                  z_t ~ N(0, 1)
    sigma_t^2 = OMEGA_i + ALPHA r_{t-1}^2 + BETA sigma_{t-1}^2,   i = 0 for t < tau (calm), 1 for t >= tau (turbulent)

The change is a jump in the long-run variance OMEGA / (1 - ALPHA - BETA) from 1 to VOL1^2, e.g. VOL1 = 2 means
"annualized vol goes from 10% to 20%". ALPHA and BETA do not change, so day-to-day clustering looks the same in both
regimes - which is exactly what makes a naive detector fire on an ordinary volatility cluster.

Detectors (the benchmarks know all parameters; DeepQCD sees only r_t, or [r_t, r_t^2] with --sq):
  - DeepQCD          same network / hyperparameters as Sec. 5.1, trained on simulated streams with tau ~ geo(RHO)
  - IID CUSUM/Shiryaev  treat returns as IID N(0, 1) vs N(0, VOL1^2): the practitioner's misspecified model
  - GARCH CUSUM/Shiryaev  conditional densities N(0, sigma_{0,t}^2) vs N(0, sigma_{1,t}^2) with both variance
                     recursions run in parallel from t = 1 (the usual approximation of the true, non-recursive
                     change-point LR of a GARCH process; it is the strongest model-based reference we have)
  - rolling variance  alarm when the mean of r^2 over the last K days exceeds h: window-limited and model-free,
                     the kind of rule a desk actually uses

Evaluation as in Sec. 5.1: Bayesian (tau ~ geo(RHO): ADD vs PFA) and minimax (tau = inf: FAP, tau = 1: ADD), plus the
conditional ADD for a late change (tau = 200), which is the number that matters here: at tau = 1 the variance starts
from its calm level and ramps up, and a recurrent detector gets the start-up bonus discussed in deepqcd_iid.py.

Run:  .venv/bin/python deepqcd_vol.py [--quick] [--sq]     -> prints results, saves figures/vol[_sq].png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from qcd import (DeepQCD, NetDetector, Recursive, bayes_metrics, cadd, check_causality, cusum, decision_statistics,
                 interp_at, shiryaev, stopping_times, train)

QUICK = '--quick' in sys.argv  # smoke-test mode: tiny dataset, few epochs, few streams
SQ = '--sq' in sys.argv        # feature transformation: feed [r_t, r_t^2] instead of r_t alone
Q = 10 if QUICK else 1

# ---- problem ----
ALPHA, BETA = 0.05, 0.90        # GARCH(1,1): persistence 0.95, typical for daily equity index returns
VOL1 = 2.0                      # turbulent-regime unconditional vol (calm regime = 1)
OMEGA = np.array([1.0, VOL1 ** 2]) * (1 - ALPHA - BETA)  # long-run variance = OMEGA / (1 - ALPHA - BETA)
RHO = 0.001                     # geometric prior of the change-point (mean 1000 days = 4 years)
K_ROLL = 20                     # window of the rolling-variance rule (one trading month)

# ---- training (as in the paper's Sec. 5) ----
N_TRAIN, N_VAL, T_TRAIN = 3200 // Q, 500 // Q, 2000
EPOCHS = 2 if QUICK else 20

# ---- evaluation ----
N_BAYES = 5000 // Q
N_FAP = 500 // Q
N_ADD = 5000 // Q
MAX_T = 20_000 if QUICK else 200_000  # give up on a stream after this many steps (FAPs beyond this are dropped)
TAU_LATE = 200                  # late change: detectors and the GARCH variance have reached steady state

np.random.seed(0)
torch.manual_seed(0)


class GARCHSampler:
    """Generates the GARCH streams of n change-points tau chunk by chunk, carrying (r_{t-1}, sigma_{t-1}^2) across
    chunks. Call with consecutive time ranges: sample(t0, L) -> observations x_{t0+1..t0+L}, shape (n, L, P)."""

    def __init__(self, n, tau):
        self.tau = np.asarray(tau, dtype=float)
        self.r_prev = np.zeros(n)
        self.sig2 = np.ones(n)  # start at the calm regime's unconditional variance

    def __call__(self, t0, L):
        r = np.empty((len(self.r_prev), L))
        for i in range(L):
            omega = np.where(t0 + i + 1 >= self.tau, OMEGA[1], OMEGA[0])
            self.sig2 = omega + ALPHA * self.r_prev ** 2 + BETA * self.sig2
            self.r_prev = np.sqrt(self.sig2) * np.random.randn(len(self.r_prev))
            r[:, i] = self.r_prev
        return features(r)


def features(r):
    """Feature transformation (paper Sec. 4.1): raw returns, or returns and squared returns."""
    f = np.stack([r, r ** 2], axis=-1) if SQ else r[..., None]
    return f.astype(np.float32)


def sample(n, T, tau):
    """Full streams (n, T, P) with labels 1{t >= tau} (n, T), for training."""
    x = GARCHSampler(n, tau)(0, T)
    y = (np.arange(1, T + 1)[None, :] >= np.asarray(tau)[:, None]).astype(np.float32)
    return x, y


def llr_iid(x, x_prev):
    """log N(r; 0, VOL1^2) / N(r; 0, 1): the IID model with the right unconditional vols (wrong dynamics)."""
    return 0.5 * x[..., 0] ** 2 * (1 - 1 / VOL1 ** 2) - np.log(VOL1)


class GARCHLLR:
    """log N(r_t; 0, sigma_{1,t}^2) / N(r_t; 0, sigma_{0,t}^2), where sigma_{i,t}^2 follows regime i's recursion
    from t = 1, driven by the observed returns. Stateful: keeps both variances across chunks."""

    def reset(self, n):
        self.sig2 = np.ones((2, n))

    def __call__(self, x, x_prev):
        r, out = x[..., 0], np.empty(x.shape[:2])
        r_prev = x_prev[:, 0]
        for t in range(r.shape[1]):
            self.sig2 = OMEGA[:, None] + ALPHA * r_prev ** 2 + BETA * self.sig2
            s0, s1 = self.sig2
            out[:, t] = 0.5 * r[:, t] ** 2 * (1 / s0 - 1 / s1) + 0.5 * np.log(s0 / s1)
            r_prev = r[:, t]
        return out


class RollingVariance:
    """d_t = mean of r^2 over the last K returns (fewer at the start). Window-limited, model-free."""

    def __init__(self, K, h):
        self.K, self.h = K, h

    def reset(self, n):
        self.tail = np.zeros((n, self.K - 1))

    def __call__(self, x):
        r2 = x[..., 0].astype(np.float64) ** 2
        ext = np.concatenate([self.tail, r2], axis=1)
        S = np.concatenate([np.zeros((len(r2), 1)), np.cumsum(ext, axis=1)], axis=1)
        end = np.arange(self.K, self.K + r2.shape[1])
        self.tail = ext[:, ext.shape[1] - (self.K - 1):]
        return (S[:, end] - S[:, end - self.K]) / self.K


def main():
    tag = 'vol_sq' if SQ else 'vol'
    net = DeepQCD(2 if SQ else 1)
    print(f'Training DeepQCD on {"[r, r^2]" if SQ else "raw r"}')
    x, y = sample(N_TRAIN, T_TRAIN, np.random.geometric(RHO, N_TRAIN))
    xv, yv = sample(N_VAL, T_TRAIN, np.full(N_VAL, T_TRAIN // 2))
    train(net, x, y, xv, yv, epochs=EPOCHS)

    print('Causality check')
    check_causality(net, sample(4, 300, np.full(4, 150))[0])

    print(f'Bayesian setting: tau ~ geo({RHO}), {N_BAYES} streams')
    h_prob = np.append(np.linspace(0.01, 0.99, 50), 0.995)
    tau = np.random.geometric(RHO, N_BAYES).astype(float)
    names_b = ['DeepQCD', 'GARCH Shiryaev', 'IID Shiryaev']
    stops = stopping_times([NetDetector(net, h_prob), Recursive(shiryaev(RHO), GARCHLLR(), h_prob),
                            Recursive(shiryaev(RHO), llr_iid, h_prob)],
                           N_BAYES, GARCHSampler(N_BAYES, tau), max_t=MAX_T)
    pfa, add_b = {}, {}
    for name, s in zip(names_b, stops):
        pfa[name], add_b[name] = bayes_metrics(s, tau)

    print(f'Minimax setting: FAP with tau = inf ({N_FAP} streams), ADD with tau = 1 ({N_ADD} streams)')
    detectors = lambda: [NetDetector(net, np.linspace(0.01, 0.95, 50)),
                         Recursive(cusum, GARCHLLR(), np.linspace(0.2, 8, 50)),   # FAP ~ 4e4 at h = 8
                         Recursive(cusum, llr_iid, np.linspace(0.2, 14, 50)),
                         RollingVariance(K_ROLL, np.linspace(1.0, 3.5, 50))]
    names_m = ['DeepQCD', 'GARCH CUSUM', 'IID CUSUM', f'rolling var ({K_ROLL}d)']
    fap = dict(zip(names_m, (s.mean(0) for s in stopping_times(
        detectors(), N_FAP, GARCHSampler(N_FAP, np.full(N_FAP, np.inf)), max_t=MAX_T))))
    add_m = dict(zip(names_m, (s.mean(0) - 1 for s in stopping_times(
        detectors(), N_ADD, GARCHSampler(N_ADD, np.ones(N_ADD)), chunk=200, max_t=MAX_T))))
    print(f'Minimax setting, late change: conditional ADD with tau = {TAU_LATE} ({N_ADD} streams)')
    add_l = dict(zip(names_m, (cadd(s, TAU_LATE)[0] for s in stopping_times(
        detectors(), N_ADD, GARCHSampler(N_ADD, np.full(N_ADD, TAU_LATE)), max_t=MAX_T))))

    print('\nResults (ADD in days, interpolated at matched false-alarm levels)')
    for level in (0.1, 0.01):
        print(f'  PFA = {level:<5}  ' + '  '.join(f'{k} {interp_at(pfa[k], add_b[k], level):.1f}' for k in names_b))
    for level in (250, 1000, 5000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} {interp_at(fap[k], add_m[k], level):.1f}' for k in names_m))
    for level in (250, 1000, 5000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} {interp_at(fap[k], add_l[k], level):.1f}' for k in names_m)
              + f'   (CADD, tau = {TAU_LATE})')

    # one stream around the change
    tau1 = 1000
    t = np.arange(1, tau1 + 201)
    x1 = sample(1, len(t), np.array([tau1]))[0]
    d_net = decision_statistics(NetDetector(net, None), x1)[0]
    d_sh = decision_statistics(Recursive(shiryaev(RHO), GARCHLLR(), None), x1)[0]
    d_iid = decision_statistics(Recursive(shiryaev(RHO), llr_iid, None), x1)[0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in names_b:
        ax[0].plot(pfa[k], add_b[k], label=k)
    ax[0].set(xscale='log', xlim=(1, 0.005), xlabel='PFA', ylabel='ADD (days)', title='Bayesian: vol 1 -> 2, GARCH(1,1)')
    for i, k in enumerate(names_m):
        ax[1].plot(fap[k], add_m[k], color=f'C{i}', label=f'{k}, tau = 1')
        ax[1].plot(fap[k], add_l[k], color=f'C{i}', ls='--', label=f'{k}, tau = {TAU_LATE} (cond.)')
    ax[1].set(xscale='log', xlim=(10, 50000), xlabel='FAP (days)', ylabel='ADD (days)', title='Minimax')
    ax[2].plot(t, x1[0, :, 0], color='lightgray', lw=0.8, label='$r_t$')
    ax[2].plot(t, d_net, label='DeepQCD $d_t$')
    ax[2].plot(t, d_sh, '--', label='GARCH Shiryaev posterior')
    ax[2].plot(t, d_iid, ':', label='IID Shiryaev posterior')
    ax[2].axvline(tau1, color='gray', lw=0.8)
    ax[2].set(xlim=(tau1 - 200, tau1 + 200), ylim=(-6, 6), xlabel='t (days)', title=f'One stream, change at t = {tau1}')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', f'{tag}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

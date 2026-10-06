"""
DeepQCD for a transient change (paper Sec. 5.3, Fig. 9): the change appears at tau1 and disappears at tau2,

    x_t ~ N(0, 1)  for t < tau1,     x_t ~ N(MU1, 1)  for tau1 <= t < tau2,     x_t ~ N(0, 1)  for t >= tau2,

and the goal is to catch it while it is there. With Gamma the first alarm,
    PFA = P(Gamma < tau1),   PD = P(tau1 <= Gamma < tau2),   PMD = P(Gamma >= tau2),   PFA + PD + PMD = 1,
and the curve of interest is PD vs PFA. Training labels are 1 only inside [tau1, tau2), so d_t must learn to come
back down after the change ends. Benchmark: the window-limited CUSUM of Guepie et al. [28], which knows f0, f1
and the duration K, and CUSUM / SR lose their optimality here because they assume a persistent change.

Training:   tau1 ~ geo(0.001), tau2 - tau1 ~ geo(0.002), streams of length 3000 (mean transient length 500).
Validation: rates swapped, as in the notebook.
Test:       tau1 = 1000, K = 25 (much shorter than the training transients), 10,000 streams.

Differences from the notebook (original/Sec. 5.3 -- TransientQCD/Justification_TransientQCD.ipynb):
  - vectorized evaluation, both detectors on the same streams; window CUSUM starts from zero LLRs instead of K
    pre-change samples (identical for h > 0 after the first K steps, 999 pre-change steps before tau1 here)
  - the notebook's second experiment (detecting the return f1 -> f0, not in the paper) is not reproduced

Run:  .venv/bin/python deepqcd_transient.py [--quick]     -> prints results, saves figures/transient.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from qcd import (DeepQCD, NetDetector, WindowCUSUM, check_causality, decision_statistics, figure_path, interp_at,
                 stopping_times, train)

QUICK = '--quick' in sys.argv  # smoke-test mode: tiny dataset, few epochs, few streams
Q = 10 if QUICK else 1

# ---- problem ----
MU1 = 1.0                       # post-change mean (pre-change mean 0, unit variance)
RHO1, RHO2 = 0.001, 0.002       # geometric onset and duration in the training data

# ---- training (as in the notebook) ----
N_TRAIN, N_VAL, T_TRAIN = 3200 // Q, 500 // Q, 3000
EPOCHS = 2 if QUICK else 20

# ---- evaluation ----
TAU1, K = 1000, 25              # test transient: [TAU1, TAU1 + K)
TAU2 = TAU1 + K
N_TEST = 10000 // Q

np.random.seed(0)
torch.manual_seed(0)


def sample(n, t, tau1, tau2):
    """Observations at (1-indexed) times t for n streams, and the labels 1{tau1 <= t < tau2}."""
    on = (t[None, :] >= tau1[:, None]) & (t[None, :] < tau2[:, None])
    x = np.random.randn(n, len(t), 1) + on[..., None] * MU1
    return x.astype(np.float32), on.astype(np.float32)


def training_set(n, rho1, rho2):
    tau1 = np.random.geometric(rho1, n)
    tau2 = tau1 + np.random.geometric(rho2, n)
    return sample(n, np.arange(1, T_TRAIN + 1), tau1, tau2)


def llr(x, x_prev):
    """log f1(x) / f0(x) for N(MU1, 1) vs N(0, 1)."""
    return x[..., 0] * MU1 - MU1 ** 2 / 2


def main():
    net = DeepQCD(1)
    print('Training DeepQCD')
    x, y = training_set(N_TRAIN, RHO1, RHO2)
    xv, yv = training_set(N_VAL, RHO2, RHO1)  # rates swapped, as in the notebook
    train(net, x, y, xv, yv, epochs=EPOCHS)

    print('Causality check')
    check_causality(net, sample(4, np.arange(1, 301), np.full(4, 150), np.full(4, 200))[0])

    print(f'Transient change at tau1 = {TAU1} lasting K = {K} steps, {N_TEST} streams')
    tau1, tau2 = np.full(N_TEST, TAU1), np.full(N_TEST, TAU2)
    sampler = lambda t0, L: sample(N_TEST, np.arange(t0 + 1, t0 + L + 1), tau1, tau2)[0]
    detectors = [NetDetector(net, np.linspace(0.01, 0.99, 50)),
                 WindowCUSUM(llr, K, np.linspace(0.2, 20, 50))]
    names = ['DeepQCD', 'window-limited CUSUM']
    # an alarm at t >= tau2 is a miss whatever its time, so the streams only need to run up to tau2 - 1
    stops = stopping_times(detectors, N_TEST, sampler, chunk=512, max_t=TAU2 - 1, warn=False)
    pfa = {k: (s < TAU1).mean(0) for k, s in zip(names, stops)}
    pd = {k: ((s >= TAU1) & (s < TAU2)).mean(0) for k, s in zip(names, stops)}

    print('\nResults (PD interpolated at matched PFA levels)')
    for level in (0.1, 0.03, 0.01):
        print(f'  PFA = {level:<5}  ' + '  '.join(f'{k} PD {interp_at(pfa[k], pd[k], level):.3f}' for k in names))

    # one stream: both statistics around the transient
    t = np.arange(1, TAU2 + 76)
    x1 = sample(1, t, np.array([TAU1]), np.array([TAU2]))[0]
    d_net = decision_statistics(NetDetector(net, None), x1)[0]
    d_wlc = decision_statistics(WindowCUSUM(llr, K, None), x1)[0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in names:
        ax[0].plot(pfa[k], pd[k], label=k)
    ax[0].set(xscale='log', xlim=(1, 0.002), xlabel='PFA', ylabel='PD', title=f'Transient change, K = {K} (Fig. 9)')
    for a, d, name in [(ax[1], d_net, 'DeepQCD $d_t$'), (ax[2], d_wlc, 'window-limited CUSUM $d_t$')]:
        a.axvspan(TAU1, TAU2, color='orange', alpha=0.15, label='change present')
        a.plot(t, x1[0, :, 0], color='lightgray', lw=0.8, label='$x_t$')
        a.plot(t, d, label=name)
        a.set(xlim=(TAU1 - 100, TAU2 + 75), xlabel='t', title='One stream')
    ax[1].set(ylim=(-3, 4))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    out = figure_path('transient.png')
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

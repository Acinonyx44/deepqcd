"""
DeepQCD in the IID Gaussian setting (paper Sec. 5.1, Figs. 5 & 6): a fast, vectorized PyTorch version of the
two Sec. 5.1 notebooks. Same network and hyperparameters as the notebooks.

    f0 = N(0, I_P),  f1 = N(MU1, I_P),  training change-point tau ~ geo(RHO)

Differences from the notebooks (original/Sec. 5.1 -- IID/):
  - All test streams are evaluated at once, in time chunks, carrying the recurrent state across chunks
    (the notebooks call model2.predict once per time step, which takes days on current Keras).
  - All detectors see exactly the same streams, so the benchmarks use the correct N(0, I_P) noise
    (the Minimax notebook's CUSUM/SR cells use `mu + np.random.randn()`, i.e. one noise value for all 7 dims).

Extra, not in the paper: the minimax ADD is also measured for a late change (tau = 200, conditional on no false
alarm before it). The paper's tau = 1 measurement rewards a detector that is extra sensitive right after start-up,
which a recurrent network can learn and the time-invariant CUSUM / SR recursions cannot.

Run:  .venv/bin/python deepqcd_iid.py [--quick]     -> prints results, saves figures/iid.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from qcd import (DeepQCD, NetDetector, Recursive, interp_at, bayes_metrics, cadd, check_causality, cusum,
                 decision_statistics, shiryaev, shiryaev_roberts, stopping_times, train)

QUICK = '--quick' in sys.argv  # smoke-test mode: tiny dataset, few epochs, few streams
Q = 10 if QUICK else 1

# ---- problem ----
P = 7                   # data dimension
MU1 = np.ones(P)        # post-change mean (pre-change mean is 0)
RHO = 0.001             # geometric prior of the change-point

# ---- training (as in the notebooks) ----
N_TRAIN, N_VAL, T_TRAIN = 3200 // Q, 500 // Q, 2000
EPOCHS = 2 if QUICK else 20

# ---- evaluation ----
N_BAYES = 5000 // Q     # streams for the Bayesian ADD-PFA curve (Fig. 5)
N_FAP = 500 // Q        # streams with no change, for FAP (Fig. 6)
N_ADD = 5000 // Q       # streams with tau = 1, for minimax ADD (Fig. 6)
MAX_T = 20_000 if QUICK else 10 ** 6  # give up on a stream after this many steps
TAU_LATE = 200          # extra (not in the paper): change after the detectors have reached steady state

np.random.seed(0)
torch.manual_seed(0)


def sample(n, t, tau):
    """Observations at (1-indexed) times t for n streams: x_t ~ f1 if t >= tau else f0. Also returns the labels."""
    post = t[None, :] >= tau[:, None]
    x = np.random.randn(n, len(t), P) + post[..., None] * MU1
    return x.astype(np.float32), post.astype(np.float32)


def sampler(n, tau):
    """Chunked stream generator for stopping_times: x_{t0+1..t0+L} of n streams with change-points tau."""
    return lambda t0, L: sample(n, np.arange(t0 + 1, t0 + L + 1), tau)[0]


def llr(x, x_prev):
    """log f1(x) / f0(x) for N(MU1, I) vs N(0, I); IID, so x_prev is unused."""
    return x @ MU1 - MU1 @ MU1 / 2


def main():
    net = DeepQCD(P)
    print('Training DeepQCD')
    x, y = sample(N_TRAIN, np.arange(1, T_TRAIN + 1), np.random.geometric(RHO, N_TRAIN))
    xv, yv = sample(N_VAL, np.arange(1, T_TRAIN + 1), np.full(N_VAL, T_TRAIN // 2))  # fixed tau, as in notebook
    train(net, x, y, xv, yv, epochs=EPOCHS)

    print('Causality check')
    check_causality(net, sample(4, np.arange(1, 301), np.full(4, 150))[0])

    print(f'Bayesian setting: tau ~ geo({RHO}), {N_BAYES} streams')
    h_prob = np.append(np.linspace(0.01, 0.99, 50), 0.995)
    tau = np.random.geometric(RHO, N_BAYES).astype(float)
    dq, sh = stopping_times([NetDetector(net, h_prob), Recursive(shiryaev(RHO), llr, h_prob)],
                            N_BAYES, sampler(N_BAYES, tau), max_t=MAX_T)
    pfa, add_b = {}, {}
    for name, s in [('DeepQCD', dq), ('Shiryaev', sh)]:
        pfa[name], add_b[name] = bayes_metrics(s, tau)

    print(f'Minimax setting: FAP with tau = inf ({N_FAP} streams), ADD with tau = 1 ({N_ADD} streams)')
    detectors = lambda: [NetDetector(net, np.linspace(0.01, 0.91, 50)),
                         Recursive(cusum, llr, np.linspace(0.1, 9, 50)),
                         Recursive(shiryaev_roberts, llr, np.linspace(1, 20000, 50))]
    names = ['DeepQCD', 'CUSUM', 'Shiryaev-Roberts']
    fap = dict(zip(names, (s.mean(0) for s in stopping_times(detectors(), N_FAP, sampler(N_FAP, np.full(N_FAP, np.inf)), max_t=MAX_T))))
    add_m = dict(zip(names, (s.mean(0) - 1 for s in stopping_times(detectors(), N_ADD, sampler(N_ADD, np.ones(N_ADD)), chunk=200, max_t=MAX_T))))
    print(f'Minimax setting, late change: conditional ADD with tau = {TAU_LATE} ({N_ADD} streams)')
    add_l = dict(zip(names, (cadd(s, TAU_LATE)[0] for s in stopping_times(
        detectors(), N_ADD, sampler(N_ADD, np.full(N_ADD, float(TAU_LATE))), max_t=MAX_T))))

    print('\nResults (ADD interpolated at matched false-alarm levels)')
    for level in (0.1, 0.01):
        print(f'  PFA = {level:<5}  ' + '  '.join(f'{k} ADD {interp_at(pfa[k], add_b[k], level):.2f}' for k in pfa))
    for level in (100, 1000, 10000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} ADD {interp_at(fap[k], add_m[k], level):.2f}' for k in fap))
    for level in (100, 1000, 10000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} CADD {interp_at(fap[k], add_l[k], level):.2f}' for k in fap)
              + f'   (tau = {TAU_LATE})')

    # one stream, to look at what the network learned: d_t vs the Shiryaev posterior P(tau <= t | x_1..x_t)
    tau1 = 1000
    t = np.arange(1, tau1 + 51)
    x1 = sample(1, t, np.array([tau1]))[0]
    d_net = decision_statistics(NetDetector(net, None), x1)[0]
    d_sh = decision_statistics(Recursive(shiryaev(RHO), llr, None), x1)[0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in pfa:
        ax[0].plot(pfa[k], add_b[k], label=k)
    ax[0].set(xscale='log', xlim=(1, 0.005), xlabel='PFA', ylabel='ADD', title='IID Bayesian (Fig. 5)')
    for i, k in enumerate(fap):
        ax[1].plot(fap[k], add_m[k], color=f'C{i}', label=f'{k}, tau = 1')
        ax[1].plot(fap[k], add_l[k], color=f'C{i}', ls='--', label=f'{k}, tau = {TAU_LATE} (cond.)')
    ax[1].set(xscale='log', xlim=(10, 50000), xlabel='FAP', ylabel='ADD', title='IID minimax (Fig. 6 + late change)')
    ax[2].plot(t, d_net, label='DeepQCD $d_t$')
    ax[2].plot(t, d_sh, '--', label='Shiryaev $P(\\tau \\leq t | x_{1:t})$')
    ax[2].axvline(tau1, color='gray', lw=0.8)
    ax[2].set(xlim=(tau1 - 100, tau1 + 50), xlabel='t', title=f'One stream, change at t = {tau1}')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'iid.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

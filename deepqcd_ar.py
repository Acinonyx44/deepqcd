"""
DeepQCD for a first-order autoregressive observation process (paper Sec. 5.2, Figs. 7 & 8):

    x_t = MU0 + LAM0 x_{t-1} + eps_t   (t < tau)          eps_t ~ N(0, 1),  x_0 = 0
    x_t = MU1 + LAM1 x_{t-1} + eps_t   (t >= tau)

Both the drift (MU0 -> MU1) and the correlation (LAM0 -> LAM1) change, and the observations are correlated over
time, so the model-based benchmarks must use the conditional likelihood ratio of Eq. (10) (the "modified"
CUSUM / SR of Polunchenko & Raghavan [22]; the same LR turned into a "modified" Shiryaev). DeepQCD gets the raw
x_t only and has to learn the temporal dependence itself. Same network and hyperparameters as Sec. 5.1.

Differences from the notebook (original/Sec. 5.2 -- AR/Justification_AR.ipynb):
  - one consistent time convention everywhere: x_0 = 0, observations x_1, x_2, ..., label 1{t >= tau}
    (the notebook leaves x_1 = 0 in the training data but draws it at random in the online tests)
  - vectorized evaluation, all detectors on the same streams

Extra, not in the paper: the minimax ADD is also measured for a late change (tau = 200, conditional on no false
alarm before it). The paper's tau = 1 measurement rewards a detector that is extra sensitive right after start-up,
which a recurrent network can learn and the time-invariant CUSUM / SR recursions cannot.

Run:  .venv/bin/python deepqcd_ar.py [--quick]     -> prints results, saves figures/ar.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import sources
from qcd import (DeepQCD, NetDetector, Recursive, interp_at, bayes_metrics, cadd, check_causality, cusum,
                 decision_statistics, figure_path, shiryaev, shiryaev_roberts, stopping_times, train)

QUICK = '--quick' in sys.argv  # smoke-test mode: tiny dataset, few epochs, few streams
Q = 10 if QUICK else 1

# ---- problem (defined in sources.py, shared with detect.py) ----
SRC = sources.AR(mu0=0.0, lam0=-0.3, mu1=1.0, lam1=0.2)
MU0, LAM0, MU1, LAM1, RHO = SRC.mu0, SRC.lam0, SRC.mu1, SRC.lam1, SRC.rho
ARSampler, sample, llr = SRC.sampler, SRC.training_set, SRC.llr

# ---- training (as in the notebook) ----
N_TRAIN, N_VAL, T_TRAIN = 3200 // Q, 500 // Q, 2000
EPOCHS = 2 if QUICK else 20

# ---- evaluation (trial counts as in the notebook) ----
N_BAYES = 5000 // Q     # streams for the Bayesian ADD-PFA curve (Fig. 7)
N_FAP = 1000 // Q       # streams with no change, for FAP (Fig. 8)
N_ADD = 10000 // Q      # streams with tau = 1, for minimax ADD (Fig. 8)
MAX_T = 20_000 if QUICK else 10 ** 6  # give up on a stream after this many steps
TAU_LATE = 200          # extra (not in the paper): change after the detectors have reached steady state

np.random.seed(0)
torch.manual_seed(0)


def main():
    net = DeepQCD(1)
    print('Training DeepQCD')
    x, y = sample(N_TRAIN, T_TRAIN, np.random.geometric(RHO, N_TRAIN))
    xv, yv = sample(N_VAL, T_TRAIN, np.full(N_VAL, T_TRAIN // 2))  # fixed tau, as in notebook
    train(net, x, y, xv, yv, epochs=EPOCHS)

    print('Causality check')
    check_causality(net, sample(4, 300, np.full(4, 150))[0])

    print(f'Bayesian setting: tau ~ geo({RHO}), {N_BAYES} streams')
    h_prob = np.append(np.linspace(0.01, 0.99, 50), 0.995)
    tau = np.random.geometric(RHO, N_BAYES).astype(float)
    dq, sh = stopping_times([NetDetector(net, h_prob), Recursive(shiryaev(RHO), llr, h_prob)],
                            N_BAYES, ARSampler(N_BAYES, tau), max_t=MAX_T)
    pfa, add_b = {}, {}
    for name, s in [('DeepQCD', dq), ('mod. Shiryaev', sh)]:
        pfa[name], add_b[name] = bayes_metrics(s, tau)

    print(f'Minimax setting: FAP with tau = inf ({N_FAP} streams), ADD with tau = 1 ({N_ADD} streams)')
    detectors = lambda: [NetDetector(net, np.linspace(0.01, 0.9, 50)),
                         Recursive(cusum, llr, np.linspace(0.1, 8.6, 50)),
                         Recursive(shiryaev_roberts, llr, np.linspace(0.2, 9600, 50))]
    names = ['DeepQCD', 'mod. CUSUM', 'mod. Shiryaev-Roberts']
    fap = dict(zip(names, (s.mean(0) for s in stopping_times(
        detectors(), N_FAP, ARSampler(N_FAP, np.full(N_FAP, np.inf)), max_t=MAX_T))))
    add_m = dict(zip(names, (s.mean(0) - 1 for s in stopping_times(
        detectors(), N_ADD, ARSampler(N_ADD, np.ones(N_ADD)), chunk=200, max_t=MAX_T))))
    print(f'Minimax setting, late change: conditional ADD with tau = {TAU_LATE} ({N_ADD} streams)')
    add_l = dict(zip(names, (cadd(s, TAU_LATE)[0] for s in stopping_times(
        detectors(), N_ADD, ARSampler(N_ADD, np.full(N_ADD, TAU_LATE)), max_t=MAX_T))))

    print('\nResults (ADD interpolated at matched false-alarm levels)')
    for level in (0.1, 0.01):
        print(f'  PFA = {level:<5}  ' + '  '.join(f'{k} ADD {interp_at(pfa[k], add_b[k], level):.2f}' for k in pfa))
    for level in (100, 1000, 10000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} ADD {interp_at(fap[k], add_m[k], level):.2f}' for k in fap))
    for level in (100, 1000, 10000):
        print(f'  FAP = {level:<5}  ' + '  '.join(f'{k} CADD {interp_at(fap[k], add_l[k], level):.2f}' for k in fap)
              + f'   (tau = {TAU_LATE})')

    # one stream, to look at what the network learned: d_t vs the modified-Shiryaev posterior
    tau1 = 1000
    t = np.arange(1, tau1 + 51)
    x1 = sample(1, len(t), np.array([tau1]))[0]
    d_net = decision_statistics(NetDetector(net, None), x1)[0]
    d_sh = decision_statistics(Recursive(shiryaev(RHO), llr, None), x1)[0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in pfa:
        ax[0].plot(pfa[k], add_b[k], label=k)
    ax[0].set(xscale='log', xlim=(1, 0.005), xlabel='PFA', ylabel='ADD', title='AR(1) Bayesian (Fig. 7)')
    for i, k in enumerate(fap):
        ax[1].plot(fap[k], add_m[k], color=f'C{i}', label=f'{k}, tau = 1')
        ax[1].plot(fap[k], add_l[k], color=f'C{i}', ls='--', label=f'{k}, tau = {TAU_LATE} (cond.)')
    ax[1].set(xscale='log', xlim=(10, 25000), xlabel='FAP', ylabel='ADD', title='AR(1) minimax (Fig. 8 + late change)')
    ax[2].plot(t, x1[0, :, 0], color='lightgray', lw=0.8, label='$x_t$')
    ax[2].plot(t, d_net, label='DeepQCD $d_t$')
    ax[2].plot(t, d_sh, '--', label='mod. Shiryaev $P(\\tau \\leq t | x_{1:t})$')
    ax[2].axvline(tau1, color='gray', lw=0.8)
    ax[2].set(xlim=(tau1 - 100, tau1 + 50), ylim=(-3, 4), xlabel='t', title=f'One stream, change at t = {tau1}')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    out = figure_path('ar.png')
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

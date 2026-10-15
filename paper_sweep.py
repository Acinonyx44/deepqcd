"""
The paper's Sec. 5 figures (Figs. 5-9), at the paper's settings and at a few others, so one can see how each result
moves when a parameter does. Every network is trained exactly as in the notebooks (original_qcd.Model / fit:
LSTM(16) -> Dense(10, relu) -> Dense(1, sigmoid), Adam 1e-3, BCE, batch 32, 20 epochs, 3200 training streams).

    iid <p> <mu> <seed>          Figs. 5-6: f0 = N(0, I_p), f1 = N(mu 1_p, I_p), tau ~ geo(0.001); paper p = 7, mu = 1
    ar <mu1> <lam1> <seed>       Figs. 7-8: AR(1), (mu, lam) from (0, -0.3) to (mu1, lam1); paper mu1 = 1, lam1 = 0.2
    transient <mu> <seed>        Fig. 9: N(0, 1) -> N(mu, 1) -> N(0, 1), tau1 ~ geo(0.001), tau2 - tau1 ~ geo(0.002);
                                 tested at tau1 = 1000, K = 25 (paper) and tau1 = 200, K = 10 / 50; paper mu = 1

Metrics as in the paper: Bayesian PFA = P(Gamma < tau), ADD = E[(Gamma - tau)^+] with tau ~ geo(0.001); minimax
FAP = E_inf[Gamma] (tau = inf; censored at MAX_T and estimated as total time / alarms) and ADD at tau = 1; transient
PFA, PD, PMD as in the notebook. Two extras that the paper does not have: the conditional ADD at tau = 200, and for
the IID minimax figure the benchmarks run on the notebook's own baseline generator, which draws one noise value and
copies it to all p coordinates (notes/notebook-review.md, minimax notebook cells 21, 23, 28, 30).

Run:  .venv/bin/python paper_sweep.py iid 7 1.0 0      -> runs/paper_sweep/iid_p7_mu1.0.s0.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

import sources
from original_qcd import Model, fit, scores, transient_streams, window_dec_stat_streams
from qcd import NetDetector, Recursive, bayes_metrics, cadd, cusum, shiryaev, shiryaev_roberts, stopping_times

QUICK = bool(os.environ.get('QUICK'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'quick' if QUICK else '', 'paper_sweep')
N_TRAIN, N_VAL = (64, 32) if QUICK else (3200, 500)
EPOCHS = 1 if QUICK else 20
N_BAYES, N_FAP, N_ADD, N_LATE = (500, 100, 500, 300) if QUICK else (10000, 1000, 10000, 3000)
N_TRANS = 1000 if QUICK else 10000
MAX_T, TAU_LATE = 30000, 200


def prob_grid(n=80):
    """The notebooks' grid np.linspace(0.01, 0.99, 50), plus thresholds packed towards 1 for the low-false-alarm end."""
    return np.unique(np.r_[np.linspace(0.01, 0.99, 50), 1 - np.logspace(-2, -7, n)])


def r(a, k=5):
    return [None if not np.isfinite(v) else round(float(v), k) for v in np.asarray(a, dtype=float)]


def fap_of(stop):
    """E_inf[Gamma] from streams censored at MAX_T: total observed time / number of alarms (exponential MLE)."""
    t = np.minimum(stop, MAX_T)
    alarms = np.isfinite(stop).sum(0)
    return np.where(alarms > 0, t.sum(0) / np.maximum(alarms, 1), np.nan), 1 - alarms / len(stop)


def train_net(src_train, P, seed, T, log):
    torch.manual_seed(seed)
    model = Model(P)
    x, y, xv, yv = src_train()
    hist = fit(model, x, y, xv, yv, epochs=EPOCHS, seed=seed, log=log)
    model.eval()
    return model, hist


def persistent(kind, a, b, seed):
    """Figs. 5-8 for one IID or AR configuration."""
    np.random.seed(seed)
    if kind == 'iid':
        p, mu = int(a), float(b)
        src = sources.IID(p)
        src.mu1 = mu * np.ones(p)
        tag, P = f'iid_p{p}_mu{mu}', p
    else:
        src = sources.AR(mu1=float(a), lam1=float(b))
        tag, P = f'ar_mu{float(a)}_lam{float(b)}', 1
    T, rho = 2000, 0.001

    def data():  # notebook: tau ~ geo(0.001) for training, fixed tau = timesteps // 2 for validation
        x, y = src.training_set(N_TRAIN, T, np.random.geometric(rho, N_TRAIN))
        xv, yv = src.training_set(N_VAL, T, np.full(N_VAL, T // 2))
        return x, y, xv, yv
    model, hist = train_net(data, P, seed, T, print)
    bench = {k: (make, h) for k, make, h in src.benchmarks()}
    names = list(bench)  # Shiryaev, CUSUM, Shiryaev-Roberts (modified ones for AR)
    hp = prob_grid()
    res = {'kind': kind, 'tag': tag, 'seed': seed, 'history': hist, 'P': P}
    if kind == 'iid':
        res.update(p=P, mu=float(b), kl=P * float(b) ** 2 / 2)
    else:
        res.update(mu1=float(a), lam1=float(b))

    # Bayesian (Fig. 5 / 7): DeepQCD vs (modified) Shiryaev, same streams
    tau = np.random.geometric(rho, N_BAYES).astype(float)
    sh_make, _ = bench[names[0]]
    st = stopping_times([NetDetector(model, hp), sh_make(hp)], N_BAYES, src.sampler(N_BAYES, tau), max_t=MAX_T, warn=False)
    res['bayes'] = {k: dict(zip(('pfa', 'add'), map(r, bayes_metrics(s, tau)))) for k, s in zip(['DeepQCD', names[0]], st)}
    print(f'  Bayesian done', flush=True)

    # minimax (Fig. 6 / 8): FAP at tau = inf, ADD at tau = 1, and (extra) conditional ADD at tau = 200
    dets = lambda: [NetDetector(model, hp)] + [bench[k][0](bench[k][1]) for k in names[1:]]
    labels = ['DeepQCD'] + names[1:]
    fap = [fap_of(s) for s in stopping_times(dets(), N_FAP, src.sampler(N_FAP, np.full(N_FAP, np.inf)), max_t=MAX_T, warn=False)]
    add1 = [s.mean(0) - 1 for s in stopping_times(dets(), N_ADD, src.sampler(N_ADD, np.ones(N_ADD)), chunk=200, max_t=MAX_T, warn=False)]
    late = [cadd(s, TAU_LATE)[0] for s in stopping_times(dets(), N_LATE, src.sampler(N_LATE, np.full(N_LATE, float(TAU_LATE))), chunk=500, max_t=MAX_T, warn=False)]
    res['minimax'] = {k: {'fap': r(f[0], 2), 'censored': r(f[1], 4), 'add1': r(a1, 4), 'add200': r(l, 4)}
                      for k, f, a1, l in zip(labels, fap, add1, late)}
    print(f'  minimax done', flush=True)

    if kind == 'iid' and P > 1:
        # the notebook's baseline generator: mu + np.random.randn() puts the SAME noise draw in all p coordinates
        def shared(n, tau):
            tau = np.asarray(tau, dtype=float)
            return lambda t0, L: (np.random.randn(n, L, 1) + (np.arange(t0 + 1, t0 + L + 1)[None, :, None] >= tau[:, None, None]) * src.mu1).astype(np.float32)
        bd = lambda: [bench[k][0](bench[k][1]) for k in names[1:]]
        fap_b = [fap_of(s) for s in stopping_times(bd(), N_FAP, shared(N_FAP, np.full(N_FAP, np.inf)), max_t=MAX_T, warn=False)]
        add_b = [s.mean(0) - 1 for s in stopping_times(bd(), N_ADD, shared(N_ADD, np.ones(N_ADD)), chunk=200, max_t=MAX_T, warn=False)]
        res['minimax_notebook_baselines'] = {k: {'fap': r(f[0], 2), 'add1': r(a1, 4)} for k, f, a1 in zip(names[1:], fap_b, add_b)}
        print(f'  notebook-generator baselines done', flush=True)
    return tag, res


def transient(mu, seed):
    """Fig. 9 for post-change mean mu, at three (tau1, K) settings."""
    rng = np.random.default_rng(seed)
    f0 = lambda L: rng.standard_normal((L, 1))
    f1 = lambda L: mu + rng.standard_normal((L, 1))

    def data():
        x, y, _ = transient_streams(N_TRAIN, 3000, 0.001, 0.002, f0, f1, rng)
        xv, yv, _ = transient_streams(N_VAL, 3000, 0.002, 0.001, f0, f1, rng)
        return x, y, xv, yv
    model, hist = train_net(data, 1, seed, 3000, print)
    res = {'kind': 'transient', 'tag': f'transient_mu{mu}', 'mu': mu, 'seed': seed, 'history': hist, 'evals': {}}
    for tau1, K in [(1000, 25), (1000, 10), (1000, 50), (200, 25), (200, 10), (200, 50)]:
        stats = {'DeepQCD': ([], []), 'WL-CUSUM': ([], [])}
        for c in range(0, N_TRANS, 2000):
            n = min(2000, N_TRANS - c)
            x = rng.standard_normal((n, tau1 - 1 + K))
            x[:, tau1 - 1:] += mu
            init = rng.standard_normal((n, K))  # N cell 22: the window starts with K pre-change samples
            for k, s in (('DeepQCD', scores(model, x[..., None].astype(np.float32))),
                         ('WL-CUSUM', window_dec_stat_streams(x, init, mu))):
                stats[k][0].append(s[:, :tau1 - 1].max(1))
                stats[k][1].append(s.max(1))
        out = {}
        for k, (a, b) in stats.items():
            m_pre, m_all = np.concatenate(a), np.concatenate(b)
            h = np.unique(np.quantile(np.r_[m_pre, m_all], np.linspace(0, 1, 400)))
            pfa = (m_pre[None, :] >= h[:, None]).mean(1)
            pmd = (m_all[None, :] < h[:, None]).mean(1)
            pd = 1 - pfa - pmd
            o = np.argsort(pfa)
            out[k] = {'pfa': r(pfa[o]), 'pd': r(pd[o]),
                      'pd_at': {str(al): float(pd[pfa <= al].max()) if (pfa <= al).any() else 0.0 for al in (0.01, 0.05, 0.1)}}
        res['evals'][f'{tau1},{K}'] = out
        print(f'  tau1 {tau1} K {K}: PD at PFA 0.05  DeepQCD {out["DeepQCD"]["pd_at"]["0.05"]:.3f}  WL-CUSUM {out["WL-CUSUM"]["pd_at"]["0.05"]:.3f}', flush=True)
    return res['tag'], res


if __name__ == '__main__':
    torch.set_num_threads(1)
    start = time.time()
    kind = sys.argv[1]
    if kind == 'transient':
        tag, res = transient(float(sys.argv[2]), int(sys.argv[3]))
    else:
        tag, res = persistent(kind, sys.argv[2], sys.argv[3], int(sys.argv[4]))
    res['seconds'] = round(time.time() - start)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{tag}.s{res["seed"]}.json'), 'w') as f:
        json.dump(res, f)
    print(f'{tag}: done in {res["seconds"]} s')

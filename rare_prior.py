"""
Rare changes: train DeepQCD where changes are common, deploy where they are rare, and correct the prior.

In the Bayesian setting the network's output approximates the Shiryaev posterior P(tau <= t | x_1..x_t), and that
posterior depends on the change rate rho. For a small rho and before the first few thousand steps, the posterior odds
are close to rho times the Shiryaev-Roberts statistic, so moving from the training rate rho_tr to the deployment rate
rho_dep shifts the log-odds by about log(rho_dep / rho_tr). A network trained on enriched data (changes common) can
then be deployed where changes are rare by subtracting a constant from its logit:

    logit p_dep = logit p_train + log(rho_dep / rho_tr)

A constant shift is monotone, so it does not change which stopping rules exist (the PFA-ADD curve); it changes which
threshold gives which false-alarm rate. The Bayesian rule of thumb "stop when the posterior reaches 1 - alpha" gives
PFA <= alpha only if the posterior is calibrated, so the shift decides whether that rule can be used at all.

The shift is exact for the exact posterior (Shiryaev(1e-3) shifted by log 0.1 behaves as Shiryaev(1e-4)), but a trained
network is not a perfectly calibrated posterior. So the recipe tested here has two steps, neither of which needs
labelled data at the rare deployment rate: calibrate the network at the training rate, where labelled streams are
plentiful (logit -> a * logit + b, fitted by logistic regression), then apply the shift.

Experiment (paper Sec. 5.1 with p = 1, f0 = N(0, 1), f1 = N(1, 1); networks trained exactly as in the notebooks):
    A  enriched     trained with tau ~ geo(1e-3) (the paper's rate), 3,200 streams of 2,000 steps
    B  true rate    trained with tau ~ geo(1e-4), the same budget: only about 18% of its streams contain a change
Deployment: tau ~ geo(1e-4) (truncated at 60,000 steps, 99.75% of the mass), 3,000 streams, against the optimal
Shiryaev(1e-4) and the misspecified Shiryaev(1e-3). Reported: PFA-ADD curves; realized PFA and ADD when stopping at
posterior 1 - alpha, with and without the correction; and a reliability check of each posterior.

Run:  .venv/bin/python rare_prior.py [--quick]      -> runs/rare/results.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

import sources
from original_qcd import Model, fit
from qcd import NetDetector, Recursive, bayes_metrics, shiryaev, stopping_times

QUICK = '--quick' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'rare')
RHO_TR, RHO_DEP = 1e-3, 1e-4
SHIFT = float(np.log(RHO_DEP / RHO_TR))
N_TRAIN, N_VAL, T = (64, 32, 2000) if QUICK else (3200, 500, 2000)
EPOCHS = 1 if QUICK else 20
N_EVAL, MAX_T, TAU_MAX = (300, 30000, 25000) if QUICK else (3000, 80000, 60000)
N_REL, T_REL = (50, 8000) if QUICK else (400, 30000)
ALPHAS = (0.1, 0.05, 0.01, 0.001)

logit = lambda p: np.log(p) - np.log1p(-p)
sig = lambda z: 1 / (1 + np.exp(-z))


def stat_chunks(det, x, chunk=2000):
    """d_t of one detector on streams x (n, T, P), fed in time chunks (keeps memory flat for long streams)."""
    det.reset(len(x))
    return np.concatenate([det(x[:, i:i + chunk]) for i in range(0, x.shape[1], chunk)], axis=1)


def platt(net, src, rho, seed, n=500, T_cal=6000):
    """Fit logit P(change by t) = a * logit(d_t) + b on labelled streams at the training rate (plentiful), by IRLS."""
    rng = np.random.default_rng(seed)
    tau = rng.geometric(rho, n).astype(float)
    x = src.sampler(n, tau)(0, T_cal)
    d = stat_chunks(NetDetector(net, None), x)
    z = logit(np.clip(d, 1e-12, 1 - 1e-12)).ravel()
    y = (np.arange(1, T_cal + 1)[None, :] >= tau[:, None]).ravel().astype(float)
    keep = rng.choice(len(z), min(len(z), 400000), replace=False)
    Z, Y = np.c_[z[keep], np.ones(len(keep))], y[keep]
    w = np.array([1.0, 0.0])
    for _ in range(50):
        p = sig(Z @ w)
        g = Z.T @ (p - Y)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def geo_trunc(rng, rho, n, cap):
    tau = rng.geometric(rho, n)
    while (tau > cap).any():
        bad = tau > cap
        tau[bad] = rng.geometric(rho, bad.sum())
    return tau.astype(float)


def train_net(src, rho, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = src.training_set(N_TRAIN, T, np.random.geometric(rho, N_TRAIN))
    xv, yv = src.training_set(N_VAL, T, np.full(N_VAL, T // 2))
    frac = float((y.max(1) > 0).mean())
    model = Model(1)
    hist = fit(model, x, y, xv, yv, epochs=EPOCHS, seed=seed)
    model.eval()
    return model, hist, frac


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    src = sources.IID(1)
    src.mu1 = np.ones(1)
    nets, info = {}, {}
    for name, rho in (('A', RHO_TR), ('B', RHO_DEP)):
        print(f'training {name} with tau ~ geo({rho})', flush=True)
        nets[name], hist, frac = train_net(src, rho, 0)
        info[name] = {'rho': rho, 'history': hist, 'streams_with_change': frac}
        print(f'  {frac:.0%} of the training streams contain a change', flush=True)

    a_cal, b_cal = platt(nets['A'], src, RHO_TR, 5)
    info['A']['platt'] = [a_cal, b_cal]
    print(f'Platt calibration of A at the training rate: logit -> {a_cal:.3f} * logit + {b_cal:.3f}', flush=True)
    rng = np.random.default_rng(1)
    np.random.seed(1)
    tau = geo_trunc(rng, RHO_DEP, N_EVAL, TAU_MAX)
    base = 1 - np.logspace(np.log10(0.5), -9, 120)
    h_corr = sig(logit(1 - np.array(ALPHAS)) - SHIFT)         # raw thresholds at which the corrected posterior = 1 - alpha
    h_platt = sig((logit(1 - np.array(ALPHAS)) - b_cal) / a_cal)          # calibrated at the training rate, no shift
    h_both = sig((logit(1 - np.array(ALPHAS)) - SHIFT - b_cal) / a_cal)  # calibrated, then shifted to the deployment rate
    grid_A = np.unique(np.r_[base, 1 - np.array(ALPHAS), h_corr, h_platt, h_both])
    grid = np.unique(np.r_[base, 1 - np.array(ALPHAS)])
    dets = [NetDetector(nets['A'], grid_A), NetDetector(nets['B'], grid),
            Recursive(shiryaev(RHO_DEP), src.llr, grid), Recursive(shiryaev(RHO_TR), src.llr, grid)]
    names = ['DeepQCD A (enriched)', 'DeepQCD B (true rate)', 'Shiryaev(1e-4), optimal', 'Shiryaev(1e-3), wrong prior']
    stops = stopping_times(dets, N_EVAL, src.sampler(N_EVAL, tau), chunk=2000, max_t=MAX_T, warn=True)
    print(f'evaluated ({time.time() - t0:.0f} s)', flush=True)

    res = {'rho_tr': RHO_TR, 'rho_dep': RHO_DEP, 'shift': SHIFT, 'n_eval': N_EVAL, 'nets': info, 'curves': {}, 'rule': {}}
    for nm, det, st in zip(names, dets, stops):
        st = np.where(np.isinf(st), MAX_T, st)
        pfa, add = bayes_metrics(st, tau)
        o = np.argsort(pfa)
        res['curves'][nm] = {'pfa': np.round(pfa[o], 5).tolist(), 'add': np.round(add[o], 3).tolist()}
        rows = {}
        for a in ALPHAS:
            j = int(np.argmin(np.abs(det.h - (1 - a))))
            rows[str(a)] = {'pfa': float(pfa[j]), 'add': float(add[j])}
        res['rule'][nm] = rows
        if nm.startswith('DeepQCD A'):
            corr = {}
            for a, hc in zip(ALPHAS, h_corr):
                j = int(np.argmin(np.abs(det.h - hc)))
                corr[str(a)] = {'pfa': float(pfa[j]), 'add': float(add[j])}
            res['rule']['DeepQCD A, prior-corrected'] = corr
            for label, hs in (('DeepQCD A, calibrated', h_platt), ('DeepQCD A, calibrated + prior-corrected', h_both)):
                rr = {}
                for a, hc in zip(ALPHAS, hs):
                    j = int(np.argmin(np.abs(det.h - hc)))
                    rr[str(a)] = {'pfa': float(pfa[j]), 'add': float(add[j])}
                res['rule'][label] = rr
    for nm, rows in res['rule'].items():
        print(f'{nm:30s} ' + '  '.join(f'alpha {a}: PFA {r["pfa"]:.4f} ADD {r["add"]:.1f}' for a, r in rows.items()), flush=True)

    # reliability: predicted posterior against the empirical frequency of {t >= tau}
    tau_r = geo_trunc(rng, RHO_DEP, N_REL, T_REL - 2000)
    x = src.sampler(N_REL, tau_r)(0, T_REL)
    on = (np.arange(1, T_REL + 1)[None, :] >= tau_r[:, None])
    post = {'DeepQCD A (enriched)': stat_chunks(NetDetector(nets['A'], None), x),
            'DeepQCD B (true rate)': stat_chunks(NetDetector(nets['B'], None), x),
            'Shiryaev(1e-4), optimal': stat_chunks(Recursive(shiryaev(RHO_DEP), src.llr, None), x)}
    zA = logit(np.clip(post['DeepQCD A (enriched)'], 1e-12, 1 - 1e-12))
    post['DeepQCD A, prior-corrected'] = sig(zA + SHIFT)
    post['DeepQCD A, calibrated + prior-corrected'] = sig(a_cal * zA + b_cal + SHIFT)
    edges = np.linspace(-12, 12, 25)
    rel = {}
    for nm, p in post.items():
        z = logit(np.clip(p, 1e-12, 1 - 1e-12)).ravel()
        b = np.digitize(z, edges)
        pts = []
        for k in range(1, len(edges)):
            sel = b == k
            if sel.sum() >= 200:
                pts.append([float(sig(z[sel]).mean()), float(on.ravel()[sel].mean()), int(sel.sum())])
        rel[nm] = pts
    res['reliability'] = rel
    for k, m in nets.items():
        torch.save(m.state_dict(), os.path.join(OUT, f'net_{k}.pt'))
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'))
    print(f'done in {res["seconds"]} s')


if __name__ == '__main__':
    main()

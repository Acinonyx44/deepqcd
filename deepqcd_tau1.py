"""
The real datasets under the paper's own real-data protocol (Sec. 6.2, the N-BaIoT experiment), for a side-by-side
with deepqcd_real.py:

    "we simulate an infinite data stream by uniformly sampling from the normal dataset for the pre-change case
     and the [attack] dataset for the post-change case. For the ADD and the FAP calculations, we set tau = 1
     and tau = inf, respectively."

Per dataset:

  pools     normal pool = every pre-change row, attack pool = every post-change row (the first H after the
            labelled onset), cut from the same standardized windows as deepqcd_real.py: the training windows
            give the training pools, the validation windows the validation pools, the test windows the test pools.
  training  streams of L_TRAIN rows drawn independently and uniformly from the training pools, change-point
            uniform in [L_TRAIN/16, 15 L_TRAIN/16] (the paper: length 2048, tau ~ U[128, 1920]); a quarter of
            the streams have no change, because the FAP streams below are longer than the training streams.
            DeepQCD and DeepQCD-hybrid exactly as in deepqcd_real.py, 3 seeds each.
  rivals    the same fitted-Gaussian CUSUM, MEWMA, Shewhart and domain rules as deepqcd_real.py.
  testing   tau = inf: N_FAP streams of T_FAP rows from the normal test pool; FAP = observed steps / false alarms
                       (the mean time to a false alarm, censored streams included);
            tau = 1:   N_ADD streams of T_ADD rows from the attack test pool, so the very first row is already
                       post-change; ADD = mean of (alarm time - 1), a miss counting T_ADD.
            Curves are drawn only where at least MIN_ALARMS false alarms were seen, so the FAP is measured,
            not extrapolated.

What this protocol does to the data: rows are shuffled, so every bit of temporal structure (trends, oscillations,
the order of a fault's development) is gone and the problem becomes exactly IID p -> q. That is why the curves
are smooth (unlimited resampled streams) and also why they say little about the real recordings; read them next
to figures/real.png, never instead of it. Two detectors are handicapped by tau = 1 by construction: the hybrid's
self-referenced input and the self-calibrating chart (keystrokes) both take their baseline from the stream's first
steps, which here are already post-change.

    .venv/bin/python deepqcd_tau1.py                 # all datasets (~1.5 h on 4 CPUs)
    .venv/bin/python deepqcd_tau1.py skab tep        # a subset
    .venv/bin/python deepqcd_tau1.py --quick         # smoke test
    .venv/bin/python deepqcd_tau1.py --report        # table + figures/real_tau1.png from runs/tau1/
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import realdata
from deepqcd_real import DOMAIN, MEWMA, GaussLLR, hybrid_features, prepare
from qcd import DeepQCD, NetDetector, Recursive, cusum, decision_statistics, figure_path, train

QUICK = '--quick' in sys.argv
SEEDS = 1 if QUICK else 3
EPOCHS = 2 if QUICK else 20
N_TRAIN, L_TRAIN = (200, 256) if QUICK else (1500, 1024)
N_VAL = N_TRAIN // 5
N_FAP, T_FAP = (100, 500) if QUICK else (500, 2000)
N_ADD, T_ADD = (100, 200) if QUICK else (500, 500)
MIN_ALARMS = 10
FAP_LEVELS = (100, 1000)
OUT = os.path.join('runs', 'quick', 'tau1') if QUICK else os.path.join('runs', 'tau1')


# ---------------------------------------------------------------- pools and streams

def pools_from_windows(x, y, m):
    """Normal and attack rows of labelled, masked windows."""
    return x[(m > 0) & (y == 0)], x[(m > 0) & (y == 1)]


def pools_from_test(xs, tau, length):
    pre = np.concatenate([xs[i, :tau[i]] for i in range(len(xs))])
    post = np.concatenate([xs[i, tau[i]:length[i]] for i in range(len(xs))])
    return pre, post


def iid_streams(pre, post, n, L, rng, tau=None):
    """n streams of L rows drawn independently from the pools; row t (0-based) is post-change iff t >= tau - 1.
    tau: (n,) 1-based change-points, np.inf for no change. Returns x (n, L, P) and labels (n, L)."""
    tau = np.full(n, np.inf) if tau is None else np.asarray(tau, float)
    lab = np.arange(1, L + 1)[None] >= tau[:, None]
    x = np.where(lab[..., None], post[rng.integers(len(post), size=(n, L))], pre[rng.integers(len(pre), size=(n, L))])
    return x.astype(np.float32), lab.astype(np.float32)


def train_streams(pre, post, n, rng):
    tau = rng.integers(L_TRAIN // 16, 15 * L_TRAIN // 16 + 1, n).astype(float)
    tau[rng.random(n) < 0.25] = np.inf
    return iid_streams(pre, post, n, L_TRAIN, rng, tau)


def stats(det, x, chunk=100):
    """Decision statistics in row chunks (each chunk starts fresh streams, so this is exact)."""
    return np.concatenate([decision_statistics(det, x[i:i + chunk]) for i in range(0, len(x), chunk)])


# ---------------------------------------------------------------- minimax curves

def minimax_curve(d_fap, d_add):
    """Sweep h; returns FAP (mean steps to a false alarm on tau = inf streams), ADD (tau = 1, misses = T_ADD),
    DR (fraction caught within T_ADD) and the number of false alarms behind each FAP."""
    clean = lambda d: np.nan_to_num(d, nan=-np.inf, posinf=1e300)
    cf, ca = np.maximum.accumulate(clean(d_fap), 1), np.maximum.accumulate(clean(d_add), 1)
    fin = lambda v: v[np.isfinite(v)]
    h = np.unique(np.r_[np.quantile(fin(cf[:, -1]), np.linspace(0, 1, 400)),
                        np.quantile(fin(ca.ravel()), np.linspace(0, 1, 400))])
    gf = np.stack([np.searchsorted(r, h) for r in cf])  # 0-based index of the first alarm, T if none
    ga = np.stack([np.searchsorted(r, h) for r in ca])
    alarms = (gf < cf.shape[1]).sum(0)
    fap = np.minimum(gf + 1, cf.shape[1]).sum(0) / np.maximum(alarms, 1)
    fap = np.where(alarms > 0, fap, np.inf)
    add = np.minimum(ga, ca.shape[1]).mean(0)  # tau = 1: an alarm at the first row has delay 0
    dr = (ga < ca.shape[1]).mean(0)
    return fap, add, dr, alarms


def at_fap(curve, level):
    """The best ADD among thresholds whose measured FAP is at least `level` (and backed by MIN_ALARMS alarms)."""
    fap, add, dr, n = curve
    ok = (fap >= level) & (n >= MIN_ALARMS)
    if not ok.any():
        return np.nan, np.nan
    i = np.flatnonzero(ok)[np.argmin(add[ok])]
    return float(add[i]), float(dr[i])


# ---------------------------------------------------------------- one dataset

def run(name):
    t0 = time.time()
    p = prepare(name)
    d = p['d']
    rng = np.random.default_rng(0)
    tr_pre, tr_post = pools_from_windows(p['xt'], p['yt'], p['mt'])
    va_pre, va_post = pools_from_windows(p['xv'], p['yv'], p['mv'])
    te_pre, te_post = pools_from_test(p['xs'], p['tau'], p['length'])
    print(f'\n=== {d.label} ({name}): pools train {len(tr_pre)}/{len(tr_post)}, test {len(te_pre)}/{len(te_post)} rows')

    xt, yt = train_streams(tr_pre, tr_post, N_TRAIN, rng)
    xv, yv = train_streams(va_pre, va_post, N_VAL, rng)
    xf, _ = iid_streams(te_pre, te_post, N_FAP, T_FAP, rng)                         # tau = inf
    xa, _ = iid_streams(te_pre, te_post, N_ADD, T_ADD, rng, np.ones(N_ADD))          # tau = 1

    # the same rivals as deepqcd_real.rivals, fitted on the same training rows
    pre64 = p['pre_n'].astype(np.float64)
    llr = GaussLLR(pre64, p['post_n'].astype(np.float64))
    rival_dets = {'CUSUM (fitted Gaussians)': lambda: Recursive(cusum, llr, None),
                  'MEWMA chart': lambda: MEWMA(pre64),
                  'Shewhart chart': lambda: MEWMA(pre64, lam=1.0),
                  **{DOMAIN[k][0]: (lambda cls=DOMAIN[k][1]: cls(p['pre_n'])) for k in d.extra}}
    curves = {k: minimax_curve(stats(make(), xf), stats(make(), xa)) for k, make in rival_dets.items()}

    mew = MEWMA(pre64)
    feats = {'DeepQCD': lambda x: x,
             'DeepQCD-hybrid': lambda x: hybrid_features(x, np.ones(x.shape[:2], np.float32), llr, mew)}
    for tag, f in feats.items():
        ft, fv, ff, fa = f(xt), f(xv), f(xf), f(xa)
        hidden = 16 if ft.shape[2] <= 10 else 32
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            net = DeepQCD(ft.shape[2], hidden)
            path = os.path.join(OUT, 'models', f'{name}.{tag}.s{seed}.pt')
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, weights_only=True))
                net.eval()
            else:
                print(f'  {tag} seed {seed}: training on {len(ft)} IID-resampled streams of {L_TRAIN}')
                train(net, ft, yt, fv, yv, epochs=EPOCHS)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(net.state_dict(), path)
            det = NetDetector(net, None)
            curves[f'{tag} #{seed}'] = minimax_curve(stats(det, ff), stats(det, fa))

    res = {'name': name, 'label': d.label, 'group': d.group, 'unit': d.unit, 'P': p['P'],
           'pools': {'train': [len(tr_pre), len(tr_post)], 'test': [len(te_pre), len(te_post)]},
           'detectors': {k: {str(lv): dict(zip(('ADD', 'DR'), at_fap(c, lv))) for lv in FAP_LEVELS}
                         for k, c in curves.items()}}
    print_table(res)
    print(f'  ({time.time() - t0:.0f}s)')
    return res, curves


# ---------------------------------------------------------------- reporting

def summarize(res):
    """Seeds collapsed to their median at each FAP level."""
    out, fams = {}, {}
    for k, v in res['detectors'].items():
        if ' #' in k:
            fams.setdefault(k.split(' #')[0], []).append(v)
        else:
            out[k] = v
    for fam, seeds in fams.items():
        out[fam] = {lv: {m: float(np.nanmedian([s[lv][m] for s in seeds]))
                         if np.isfinite([s[lv][m] for s in seeds]).any() else np.nan for m in ('ADD', 'DR')}
                    for lv in map(str, FAP_LEVELS)}
    return out


def print_table(res):
    print(f'  {"detector":28s}' + ''.join(f'   FAP {lv:>5}: ADD    DR' for lv in FAP_LEVELS))
    for k, v in summarize(res).items():
        print(f'  {k:28s}' + ''.join(f'          {v[str(lv)]["ADD"]:6.1f} {v[str(lv)]["DR"]:5.2f}' for lv in FAP_LEVELS))


def plot(figs, path):
    cols = min(4, len(figs))
    rows = int(np.ceil(len(figs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    colors = {'CUSUM (fitted Gaussians)': 'C1', 'MEWMA chart': 'C2', 'Shewhart chart': 'C7',
              **{v[0]: 'C3' for v in DOMAIN.values()}}
    for ax, (res, curves) in zip(axes.flat, figs):
        for k, (fap, add, dr, n) in curves.items():
            ok = np.isfinite(fap) & (n >= MIN_ALARMS)
            o = np.argsort(fap[ok])
            deep = k.startswith('DeepQCD')
            fam = k.split(' #')[0]
            ax.plot(fap[ok][o], add[ok][o], color=('C4' if 'hybrid' in k else 'C0') if deep else colors[k],
                    lw=1 if deep else 1.8, alpha=0.7 if deep else 1,
                    label=(f'{fam} ({SEEDS} seeds)' if k.endswith('#0') else None) if deep else k)
        ax.set_xscale('log')
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.set_title(res['label'], fontsize=9)
        ax.set_xlabel('FAP (mean steps to a false alarm, tau = inf)')
        ax.set_ylabel(f'ADD at tau = 1 ({res["unit"]})')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(figs):]:
        ax.axis('off')
    fig.suptitle('Real datasets under the paper\'s Sec. 6.2 protocol: rows resampled IID from the normal / '
                 'changed pools, ADD at tau = 1, FAP at tau = inf', y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    print(f'\nSaved {path}')


def save(res, curves):
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, res['name'])
    with open(stem + '.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    np.savez(stem + '.npz', **{f'{k}|{m}': v for k, c in curves.items() for m, v in zip(('fap', 'add', 'dr', 'n'), c)})


def load(name):
    stem = os.path.join(OUT, name)
    if not os.path.exists(stem + '.json'):
        return None
    res, z = json.load(open(stem + '.json')), np.load(stem + '.npz')
    return res, {k: tuple(z[f'{k}|{m}'] for m in ('fap', 'add', 'dr', 'n')) for k in res['detectors']}


def report():
    figs = []
    for n in realdata.LOADERS:
        r = load(n)
        if r is None:
            print(f'  (no saved run for {n})')
            continue
        print(f'\n=== {r[0]["label"]} ({n})')
        print_table(r[0])
        figs.append(r)
    plot(figs, figure_path('real_tau1.png'))


def main():
    if '--report' in sys.argv:
        return report()
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(realdata.LOADERS)
    for n in names:
        save(*run(n))
    report()


if __name__ == '__main__':
    main()

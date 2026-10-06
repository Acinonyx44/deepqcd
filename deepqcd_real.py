"""
DeepQCD on real data: every dataset in realdata.py through one protocol, against two data-driven
classical detectors.

    .venv/bin/python deepqcd_real.py                 # all datasets, ~1 h on 4 CPUs
    .venv/bin/python deepqcd_real.py skab tep        # a subset
    .venv/bin/python deepqcd_real.py --quick         # smoke test
    .venv/bin/python deepqcd_real.py --report        # table + figures/real.png from the saved runs/real/
    .venv/bin/python deepqcd_real.py --hybrid        # the hybrid variant (adds to the saved runs; see below)

Protocol, per dataset:

  windows   Every training and test stream is a window cut from an episode: a pre-change stretch of random
            length in [pmin, pmax] followed by at most H post-change steps, padded to pmax + H and masked.
            Randomizing the pre-change length makes tau random even where the recordings all change at the
            same index. A quarter of the training windows come from change-free data where the dataset has
            some, so the network also learns to stay quiet.
  training  DeepQCD exactly as in Sec. 5 (LSTM -> dense -> sigmoid, BCE against 0/1 labels, Adam, early
            stopping on held-out episodes), 3 seeds. Inputs standardized with pre-change training statistics.
  rivals    Both fitted on the same training windows, as a practitioner would without a model:
              CUSUM    log-likelihood ratio of two Gaussians: f0 fitted to pre-change samples, f1 to
                       post-change samples (full covariances, 10 % shrinkage). Optimal if the data were IID
                       Gaussian with one kind of change, which they are not.
              MEWMA    multivariate EWMA chart (lambda = 0.1) of the observations whitened with f0: needs
                       only normal data, reacts to any mean shift.
  testing   Run on the same test windows; sweep the threshold; for each h:
              PFA      fraction of windows that alarm before the change
              ADD      mean delay over the windows that did not false-alarm, a window that never alarms
                       counting its full post-change length (so misses are penalized, not dropped)
              DR       fraction of those windows that alarm within the horizon
            and report ADD and DR at the operating point for PFA <= 0.05, 0.1, 0.25 (the lowest threshold
            meeting the budget), i.e. at a matched false-alarm level, never at a matched threshold.
            The full curves are saved to runs/real/<name>.npz.
  hybrid    --hybrid trains a variant whose LSTM also gets the classical statistics (fitted log-LR, MEWMA T^2)
            and the input re-referenced to the stream's own first 20 steps; saved as <name>.hybrid.*, and
            --report shows it next to plain DeepQCD.
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import realdata
from qcd import DeepQCD, NetDetector, Recursive, cusum, decision_statistics, figure_path, train

QUICK = '--quick' in sys.argv
HYBRID = '--hybrid' in sys.argv  # train the hybrid variant instead of plain DeepQCD (results saved separately)
SEEDS = 1 if QUICK else 3
N_TRAIN = 200 if QUICK else 2000
N_TEST = 100 if QUICK else 600
EPOCHS = 2 if QUICK else 20
LEVELS = (0.05, 0.1, 0.25)
PFA_FLOOR = 1e-3  # where PFA = 0 is drawn on the log axis (1/N_TEST < 2e-3)


# ---------------------------------------------------------------- windows

def windows(eps, n, rng, d, calm=(), p_calm=0.0, cycle=False):
    """n windows of length pmax + H. Returns x (n, L, P), y (n, L), mask (n, L), tau (n,), length (n,)."""
    L = d.pmax + d.H
    P = eps[0][0].shape[1]
    x, y, m = np.zeros((n, L, P), np.float32), np.zeros((n, L), np.float32), np.zeros((n, L), np.float32)
    tau, length = np.zeros(n, int), np.zeros(n, int)
    calm = [c for c in calm if len(c) >= d.pmin + 1]
    for i in range(n):
        if calm and rng.random() < p_calm:
            c = calm[rng.integers(len(calm))]
            k = min(L, len(c))
            s = rng.integers(len(c) - k + 1)
            seg, t = c[s:s + k], k  # no change inside the window
        else:
            xe, te = eps[i % len(eps)] if cycle else eps[rng.integers(len(eps))]
            pre = rng.integers(min(d.pmin, te), min(d.pmax, te) + 1)
            post = min(d.H, len(xe) - te)
            seg, t = xe[te - pre:te + post], pre
        x[i, :len(seg)], m[i, :len(seg)], y[i, t:len(seg)] = seg, 1, 1
        tau[i], length[i] = t, len(seg)
    return x, y, m, tau, length


def synthetic_windows(src, n, rng, d):
    """Training windows from a simulated source (S&P 500: the GARCH regime change)."""
    L = d.pmax + d.H
    np.random.seed(int(rng.integers(2 ** 31)))
    tau = np.where(rng.random(n) < 0.25, L + 1, rng.integers(d.pmin, d.pmax + 1, n) + 1)  # 1-based for sources
    x, y = src.training_set(n, L, tau)
    return x, y, np.ones((n, L), np.float32)


# ---------------------------------------------------------------- data-driven classical detectors

def _gauss(z, shrink=0.1):
    mu = z.mean(0)
    S = np.atleast_2d(np.cov(z.T))
    S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S)) + 1e-6 * np.eye(len(S))
    return mu, S


class GaussLLR:
    """log f1(x)/f0(x) for two fitted Gaussians."""

    def __init__(self, pre, post):
        self.mu0, S0 = _gauss(pre)
        self.mu1, S1 = _gauss(post)
        self.P0, self.P1 = np.linalg.inv(S0), np.linalg.inv(S1)
        self.c = 0.5 * (np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])

    def __call__(self, x, x_prev):
        a, b = x - self.mu0, x - self.mu1
        return self.c + 0.5 * (np.einsum('nlp,pq,nlq->nl', a, self.P0, a) - np.einsum('nlp,pq,nlq->nl', b, self.P1, b))


class MEWMA:
    """Multivariate EWMA chart: z_t = lam x~_t + (1 - lam) z_{t-1} on whitened x~; d_t = z_t' z_t (2 - lam) / lam."""

    def __init__(self, pre, lam=0.1):
        self.mu, S = _gauss(pre)
        self.W = np.linalg.inv(np.linalg.cholesky(S)).T
        self.lam, self.h = lam, None

    def reset(self, n):
        self.z = np.zeros((n, len(self.mu)))

    def __call__(self, x):
        xw = (x.astype(np.float64) - self.mu) @ self.W
        out = np.empty(x.shape[:2])
        for t in range(x.shape[1]):
            self.z = self.lam * xw[:, t] + (1 - self.lam) * self.z
            out[:, t] = (self.z ** 2).sum(1) * (2 - self.lam) / self.lam
        return out


class STALTA:
    """Seismology's standard trigger: short-term over long-term average of signal energy (0.5 s / 10 s EWMAs at
    100 Hz). The long-term average starts at the known noise energy, as on a running station."""

    def __init__(self, pre, sta=50, lta=1000):
        self.e0 = float(np.median((pre.astype(np.float64) ** 2).sum(1)))  # robust: a few traces are very loud
        self.a, self.b, self.h = 1 / sta, 1 / lta, None

    def reset(self, n):
        self.s, self.l = np.full(n, self.e0), np.full(n, self.e0)

    def __call__(self, x):
        e = (x.astype(np.float64) ** 2).sum(2)
        out = np.empty(e.shape)
        for t in range(e.shape[1]):
            self.s += self.a * (e[:, t] - self.s)
            self.l += self.b * (e[:, t] - self.l)
            out[:, t] = self.s / self.l
        return out


class FreezeIndex:
    """Moore et al. (2008) / Baechlin et al. (2010) freeze index for gait: power in the 3-8 Hz 'freeze' band over
    power in the 0.5-3 Hz locomotion band, over the trailing 4 s of the vertical axis (32 Hz)."""

    def __init__(self, pre, fs=32, win=128, axis=1):
        f = np.fft.rfftfreq(win, 1 / fs)
        self.fb, self.lb = (f >= 3) & (f <= 8), (f >= 0.5) & (f < 3)
        self.win, self.axis, self.h = win, axis, None

    def reset(self, n):
        self.tail = np.zeros((n, self.win - 1))

    def __call__(self, x):
        v = np.concatenate([self.tail, x[:, :, self.axis].astype(np.float64)], 1)
        self.tail = v[:, -(self.win - 1):]
        w = np.lib.stride_tricks.sliding_window_view(v, self.win, axis=1)  # (n, L, win)
        w = w - w.mean(2, keepdims=True)
        p = np.abs(np.fft.rfft(w * np.hanning(self.win), axis=2)) ** 2
        return np.log((p[..., self.fb].sum(2) + 1e-9) / (p[..., self.lb].sum(2) + 1e-9))


class SelfRefMEWMA:
    """A chart that calibrates itself on each stream: mean and variance (per feature) from the first `burn`
    observations of the stream, then a diagonal MEWMA against that baseline. Needs no training data at all,
    and is the natural rival when every stream has its own normal (here: each typist)."""

    def __init__(self, pre, burn=15, lam=0.2):
        self.burn, self.lam, self.h = burn, lam, None

    def reset(self, n):
        self.buf, self.t, self.z = [], 0, None

    def __call__(self, x):
        x = x.astype(np.float64)
        out = np.zeros(x.shape[:2])
        for t in range(x.shape[1]):
            if self.t < self.burn:
                self.buf.append(x[:, t])
                if self.t == self.burn - 1:
                    b = np.stack(self.buf, 1)
                    self.mu, self.sd = b.mean(1), b.std(1) + 0.1
                    self.z = np.zeros_like(self.mu)
            else:
                self.z = self.lam * (x[:, t] - self.mu) / self.sd + (1 - self.lam) * self.z
                out[:, t] = (self.z ** 2).mean(1) * (2 - self.lam) / self.lam
            self.t += 1
        return out


DOMAIN = {'stalta': ('STA/LTA trigger', STALTA), 'freeze_index': ('Freeze index', FreezeIndex),
          'selfref': ('Self-calibrating chart', SelfRefMEWMA)}


# ---------------------------------------------------------------- evaluation

def tradeoff(dstat, tau, length):
    """Sweep h over the statistic's own values. Returns PFA, ADD, DR arrays (one entry per threshold)."""
    n, L = dstat.shape
    valid = np.arange(L)[None] < length[:, None]
    dm = np.where(valid, np.nan_to_num(dstat, nan=-np.inf, posinf=1e300), -np.inf)
    cm = np.maximum.accumulate(dm, axis=1)
    pre_max = np.array([cm[i, tau[i] - 1] if tau[i] > 0 else -np.inf for i in range(n)])
    cand = np.r_[np.quantile(pre_max[np.isfinite(pre_max)], np.linspace(0, 1, 300)),
                 np.quantile(dm[valid & np.isfinite(dm)], np.linspace(0, 1, 300))]
    h = np.unique(cand)
    stop = np.stack([np.searchsorted(cm[i], h) for i in range(n)])  # first index with running max >= h
    early = stop < tau[:, None]
    det = (stop < length[:, None]) & ~early
    delay = np.where(det, stop - tau[:, None], (length - tau)[:, None])
    ok = ~early
    cnt = np.maximum(ok.sum(0), 1)
    return early.mean(0), np.where(ok, delay, 0).sum(0) / cnt, det.sum(0) / cnt


def at_level(pfa, val, level, add):
    """The operating point for a false-alarm budget: the lowest threshold with PFA <= level (the one with the
    smallest delay). Read off the step curve rather than interpolated, because some detectors jump from
    PFA 1 straight to 0 (e.g. a CUSUM whose pre-change LLR is strongly negative never false-alarms)."""
    ok = pfa <= level
    if not ok.any():
        return np.nan
    i = np.flatnonzero(ok)[np.argmin(add[ok])]
    return float(val[i])


# ---------------------------------------------------------------- hybrid DeepQCD

def hybrid_features(x, m, llr, mewma, burn=20):
    """Inputs of the hybrid variant: the observations, the observations re-referenced to the stream's own
    first `burn` steps (causal running mean until then), the fitted log-LR and log(1 + MEWMA T^2). The LSTM
    starts from what the classical charts already know and only has to learn what they miss."""
    xm = x * m[..., None]
    k = np.minimum(np.arange(x.shape[1]), burn - 1)
    c = np.cumsum(xm, 1)[:, k] / (k + 1)[None, :, None]  # mean of the first min(t, burn) steps
    l = np.clip(llr(x.astype(np.float64), None), -50, 50) / 10
    t2 = np.log1p(decision_statistics(mewma, x)) / 3
    f = np.concatenate([x, x - c, l[..., None], t2[..., None]], 2)
    return (f * m[..., None]).astype(np.float32)


# ---------------------------------------------------------------- one dataset

def run(name):
    t0 = time.time()
    d = realdata.LOADERS[name]()
    print(f'\n=== {d.label}  ({name}; {len(d.train)} train / {len(d.test)} test episodes; {d.notes})')
    rng = np.random.default_rng(0)
    P = d.test[0][0].shape[1]

    # training / validation windows (validation episodes held out)
    if d.synthetic is not None:
        xt, yt, mt = synthetic_windows(d.synthetic, N_TRAIN, rng, d)
        xv, yv, mv = synthetic_windows(d.synthetic, N_TRAIN // 5, rng, d)
    else:
        k = max(1, len(d.train) // 5)
        perm = rng.permutation(len(d.train))
        tr_eps = [d.train[i] for i in perm[k:]] or d.train
        va_eps = [d.train[i] for i in perm[:k]]
        p_calm = 0.25 if d.calm else 0.0
        xt, yt, mt = windows(tr_eps, N_TRAIN, rng, d, d.calm, p_calm)[:3]
        xv, yv, mv = windows(va_eps, max(N_TRAIN // 5, 50), rng, d, d.calm, p_calm)[:3]
    pre = xt[(mt > 0) & (yt == 0)].astype(np.float64)
    mu, sd = pre.mean(0), pre.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    xt, xv = norm(xt) * mt[..., None], norm(xv) * mv[..., None]
    pre_n, post_n = xt[(mt > 0) & (yt == 0)], xt[(mt > 0) & (yt == 1)]

    # test windows: every test episode in turn, each with fresh pre-change lengths
    xs, _, ms, tau, length = windows(d.test, N_TEST, np.random.default_rng(1), d, cycle=True)
    xs = norm(xs) * ms[..., None]

    curves = {}
    stats = {}
    llr = GaussLLR(pre_n.astype(np.float64), post_n.astype(np.float64))
    stats['CUSUM (fitted Gaussians)'] = decision_statistics(Recursive(cusum, llr, None), xs)
    stats['MEWMA chart'] = decision_statistics(MEWMA(pre_n.astype(np.float64)), xs)
    for key in d.extra:
        label, cls = DOMAIN[key]
        stats[label] = decision_statistics(cls(pre_n), xs)
    for name_, s in stats.items():
        curves[name_] = tradeoff(s, tau, length)
    tag = 'DeepQCD-hybrid' if HYBRID else 'DeepQCD'
    if HYBRID:
        mew = MEWMA(pre_n.astype(np.float64))
        xt, xv = hybrid_features(xt, mt, llr, mew), hybrid_features(xv, mv, llr, mew)
        xs = hybrid_features(xs, ms, llr, mew)
    hidden = 16 if xt.shape[2] <= 10 else 32
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        net = DeepQCD(xt.shape[2], hidden)
        print(f'  {tag} seed {seed}: training on {len(xt)} windows of {xt.shape[1]} steps')
        train(net, xt, yt, xv, yv, epochs=EPOCHS, w=mt, wv=mv)
        curves[f'{tag} #{seed}'] = tradeoff(decision_statistics(NetDetector(net, None), xs), tau, length)

    res = {'name': name, 'label': d.label, 'group': d.group, 'unit': d.unit, 'P': P, 'n_train_eps': len(d.train),
           'n_test_eps': len(d.test), 'H': d.H, 'notes': d.notes, 'detectors': {}}
    for k, (pfa, add, dr) in curves.items():
        res['detectors'][k] = {f'{lv}': {'ADD': at_level(pfa, add, lv, add), 'DR': at_level(pfa, dr, lv, add)}
                               for lv in LEVELS}
    print_table(res)
    print(f'  ({time.time() - t0:.0f}s)')
    return res, curves


def summarize(res):
    """Seeds of each network (DeepQCD, DeepQCD-hybrid) collapsed to the median (and range) at each level."""
    rows = {}
    fams = {}
    for k, v in res['detectors'].items():
        if ' #' in k:
            fams.setdefault(k.split(' #')[0], []).append(v)
        else:
            rows[k] = v
    out = {}
    for fam, seeds in fams.items():
        for lv in map(str, LEVELS):
            a = np.array([s_[lv]['ADD'] for s_ in seeds], float)
            r = np.array([s_[lv]['DR'] for s_ in seeds], float)
            ok = np.isfinite(a).any()
            out.setdefault(fam, {})[lv] = {'ADD': np.nanmedian(a) if ok else np.nan,
                                           'lo': np.nanmin(a) if ok else np.nan, 'hi': np.nanmax(a) if ok else np.nan,
                                           'DR': np.nanmedian(r) if np.isfinite(r).any() else np.nan}
    out.update(rows)
    return out


def print_table(res):
    print(f'  {"detector":28s}' + ''.join(f'   PFA {lv:<4}: ADD    DR ' for lv in LEVELS))
    for k, v in summarize(res).items():
        line = f'  {k:28s}'
        for lv in map(str, LEVELS):
            line += f'          {v[lv]["ADD"]:7.1f} {v[lv]["DR"]:5.2f}'
        if 'lo' in v[str(LEVELS[0])]:
            line += '   (median of seeds; ADD range ' + ', '.join(
                f'{v[str(lv)]["lo"]:.1f}-{v[str(lv)]["hi"]:.1f}' for lv in LEVELS) + ')'
        print(line)


def plot(figs, path):
    cols = min(4, len(figs))
    rows = int(np.ceil(len(figs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    colors = {'CUSUM (fitted Gaussians)': 'C1', 'MEWMA chart': 'C2', **{v[0]: 'C3' for v in DOMAIN.values()}}
    for ax, (res, curves) in zip(axes.flat, figs):
        for k, (pfa, add, dr) in curves.items():
            # threshold order (PFA falls, ADD rises), drawn as the achievable frontier: for any budget between
            # two operating points the next stricter one applies. PFA = 0 sits at the floor so it stays visible.
            deep = k.startswith('DeepQCD')
            fam = k.split(' #')[0]
            ax.plot(np.maximum(pfa, PFA_FLOOR), add, drawstyle='steps-pre',
                    color=('C4' if 'hybrid' in k else 'C0') if deep else colors[k], lw=1 if deep else 1.8,
                    alpha=0.7 if deep else 1, label=(f'{fam} (3 seeds)' if k.endswith('#0') else None) if deep else k)
        ax.set_xscale('log')
        ax.set_xlim(1, PFA_FLOOR)
        ax.set_xticks([1, 0.1, 0.01, PFA_FLOOR], ['1', '0.1', '0.01', '0'])
        ax.set_title(res['label'], fontsize=9)
        ax.set_xlabel('PFA (alarm before the change)')
        ax.set_ylabel(f'ADD, misses = horizon ({res["unit"]})')
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(figs):]:
        ax.axis('off')
    for ax in list(axes.flat)[:len(figs)]:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f'\nSaved {path}')


def report():
    """Rebuild the table and figures/real.png from the saved runs/real/<name>.{json,npz}."""
    figs = []
    for n in realdata.LOADERS:
        p = os.path.join('runs', 'real', n)
        if not os.path.exists(p + '.json'):
            print(f'  (no saved run for {n})')
            continue
        res = json.load(open(p + '.json'))
        z = np.load(p + '.npz')
        curves = {k: tuple(z[f'{k}|{m}'] for m in ('pfa', 'add', 'dr')) for k in res['detectors']}
        if os.path.exists(p + '.hybrid.json'):  # add the hybrid seeds from the --hybrid run
            hres, hz = json.load(open(p + '.hybrid.json')), np.load(p + '.hybrid.npz')
            for k in hres['detectors']:
                if k.startswith('DeepQCD-hybrid'):
                    res['detectors'][k] = hres['detectors'][k]
                    curves[k] = tuple(hz[f'{k}|{m}'] for m in ('pfa', 'add', 'dr'))
        print(f'\n=== {res["label"]}  ({n})')
        print_table(res)
        figs.append((res, curves))
    plot(figs, figure_path('real.png'))


def main():
    if '--report' in sys.argv:
        return report()
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(realdata.LOADERS)
    os.makedirs(os.path.join('runs', 'real'), exist_ok=True)
    figs = []
    for n in names:
        res, curves = run(n)
        figs.append((res, curves))
        if not QUICK:
            stem = os.path.join('runs', 'real', n + ('.hybrid' if HYBRID else ''))
            with open(stem + '.json', 'w') as f:
                json.dump(res, f, indent=1, default=float)
            np.savez(stem + '.npz',
                     **{f'{k}|{m}': v for k, c in curves.items() for m, v in zip(('pfa', 'add', 'dr'), c)})
    # a full run draws the tracked summary figure; a subset only a scratch one (rebuild with --report)
    full = len(names) == len(realdata.LOADERS) and not QUICK and not HYBRID
    plot(figs, figure_path('real.png') if full else os.path.join('runs', 'real', f'real_{"_".join(names)}.png'))


if __name__ == '__main__':
    main()

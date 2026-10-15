"""
Transient QCD on real-world data, built only from the authors' own functions (original_qcd.py).

Prof. Wang's directive: "Work on transient as the paper said, using real world data. Use functions and
algorithms strictly from the original QCD code." The paper's transient section (Sec. 5.3) and its notebook fix
everything below except where the data comes from:

    model          f0 for t < tau1, f1 for tau1 <= t < tau2, f0 again for t >= tau2           (paper Sec. 5.3)
    training       3200 streams of length 3000, tau1 ~ geo(0.001), tau2 - tau1 ~ geo(0.002),
                   label 1 exactly on [tau1, tau2); validation 500 streams, rates swapped        (N cells 7, 9)
    network        LSTM(16) -> Dense(10, relu) -> Dense(1, sigmoid), Adam, BCE, batch 32, 20 epochs (N 11-13)
    benchmark      window-limited CUSUM, window_dec_stat over the last K observations, window
                   started with K pre-change samples                                           (N cells 6, 22)
    test           tau1 = 1000, K = 25, tau2 = tau1 + K; 10,000 experiments; PFA = P(alarm before tau1),
                   PMD = P(no alarm before tau2), PD = 1 - PFA - PMD                            (N cells 19, 22, 25)

Real data enters the way the authors' own real-data code (Sec. 6.2, IoT) builds streams: contiguous blocks of
normal data at random offsets and contiguous blocks of the changed regime (IoT `random_sample`,
`random_sample_eva`), standardized by the normal training data (IoT `do_normalization`). Here f1 is a real
labelled event taken from its onset, so tau1 is a true onset and the change lasts as long as the event did.

A dataset qualifies when (C1) its labels mark events that start and end, after which the data returns to normal;
(C2) most events last at least K = 25 steps (the test uses the first K post-change samples); (C3) it has
contiguous normal stretches of at least tau1 - 1 + K = 1024 steps; (C4) it has enough events for disjoint
training and test sets. notes/transient-real.md has the table for all our datasets.

One piece is not in the original code and cannot be: window_dec_stat(obs_window) computes the CUSUM of
x - mu_post / 2, the log-likelihood ratio of N(mu_post, 1) against N(0, 1) divided by mu_post, so it needs a
scalar observation with that model. Real data is multivariate, so each observation is mapped to the scalar
    z = w'(x - m0) / sqrt(w' C0 w),   w = C0^{-1} (m1 - m0),
whose Gaussian LLR is exactly the Gaussian LLR of x (equal covariances), with m0, C0 fitted on normal training
data and m1 on training events; then mu_post = mean of z over training events, and window_dec_stat is called
unchanged. This is the standard way to give a model-based detector a model, and only the benchmark uses it.

Two ways to form the test experiments:
    blocks   the authors' construction (IoT random_sample_eva): a normal block of tau1 - 1 + K steps (the first K
             start the CUSUM window), then the first K steps of a real test event; 10,000 experiments
    native   no splicing: each test event with its own real preceding normal stretch; one experiment per event

Run:  .venv/bin/python transient_real.py <dataset> <all|z> <seed>      -> runs/transient_real/<dataset>.<input>.s<seed>.json
      .venv/bin/python transient_real.py --survey                      -> runs/transient_real/survey.json
      .venv/bin/python transient_real.py --splice                      -> runs/transient_real/splice.json
      .venv/bin/python transient_real.py --table                       -> the results table of notes/transient-real.md
"""
import json
import os
import sys
import time

import numpy as np
import torch

import realdata
from original_qcd import Model, block, experiment1, fit, normalization, scores, transient_streams, window_dec_stat_streams

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'quick' if os.environ.get('QUICK') else '', 'transient_real')
DATASETS = ['hai', 'fog', 'smd', 'occupancy', 'occupancy-nolight', 'skab']

N_TRAIN, N_VAL, TIMESTEPS = (64, 32, 3000) if os.environ.get("QUICK") else (3200, 500, 3000)      # N cells 7, 9
RHO1, RHO2 = 0.001, 0.002
EPOCHS, BATCH = (1 if os.environ.get("QUICK") else 20), 32                           # N cell 13
N_TEST = 2000 if os.environ.get("QUICK") else 10000                                   # N cell 19
TAU1S, KS = (1000, 200), (25, 10, 50)             # paper: tau1 = 1000, K = 25; the others are the toggles
ALPHAS = (0.01, 0.05, 0.1)


def pools(d):
    tr_norm = [x[:t] for x, t in d.train] + list(d.calm)
    tr_ev = [x[t:] for x, t in d.train]
    te_norm = [x[:t] for x, t in d.test]
    return tr_norm, tr_ev, te_norm, d.test


def projection(normal, events):
    """The Gaussian-LLR scalar for window_dec_stat (see the module docstring). Returns z(x), mu_post."""
    a, b = np.concatenate(normal, 0).astype(np.float64), np.concatenate(events, 0).astype(np.float64)
    m0, m1 = a.mean(0), b.mean(0)
    C0 = np.cov(a, rowvar=False).reshape(a.shape[1], a.shape[1]) + 1e-3 * np.eye(a.shape[1])
    w = np.linalg.solve(C0, m1 - m0)
    s = np.sqrt(w @ C0 @ w)
    z = lambda x: (((x - m0) @ w) / s).astype(np.float32)
    return z, float(z(b).mean())


def curve(m_pre, m_all, h):
    """PFA, PD and PMD at thresholds h from the per-experiment maxima (see experiment1)."""
    pfa = (m_pre[None, :] >= h[:, None]).mean(1)
    pmd = (m_all[None, :] < h[:, None]).mean(1)
    return pfa, 1 - pfa - pmd, pmd


def summarize(m_pre, m_all):
    """The whole PD-PFA curve on a threshold grid that resolves every experiment, plus max PD s.t. PFA <= alpha
    (the paper's Eq. 11) at a few alpha."""
    h = np.unique(np.r_[m_pre, m_all, np.inf])
    h = h[np.isfinite(h) | (h == np.inf)]
    if len(h) > 400:
        h = np.unique(np.r_[np.quantile(h[np.isfinite(h)], np.linspace(0, 1, 399)), np.inf])
    pfa, pd, pmd = curve(m_pre, m_all, h)
    best = {str(a): float(pd[pfa <= a].max()) if (pfa <= a).any() else 0.0 for a in ALPHAS}
    o = np.argsort(pfa)
    return {'pfa': np.round(pfa[o], 5).tolist(), 'pd': np.round(pd[o], 5).tolist(), 'pd_at': best}


def maxima(d, tau1):
    """Per-experiment max of the statistic before tau1 and before tau2 (d covers t = 1 .. tau2 - 1)."""
    return d[:, :tau1 - 1].max(1), d.max(1)


def run(name, inp, seed):
    t_start = time.time()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    d = realdata.LOADERS[name]()
    tr_norm, tr_ev, te_norm, te_eps = pools(d)
    norm = normalization(tr_norm)                                   # IoT do_normalization
    tr_norm, tr_ev, te_norm = [[norm(x) for x in p] for p in (tr_norm, tr_ev, te_norm)]
    te_eps = [(norm(x), t) for x, t in te_eps]
    z, mu_post = projection(tr_norm, tr_ev)
    feat = (lambda x: z(x)[..., None]) if inp == 'z' else (lambda x: x)
    tr_norm_in, tr_ev_in = [feat(x) for x in tr_norm], [feat(x) for x in tr_ev]
    P = tr_norm_in[0].shape[1]
    print(f'{name} [{inp}] seed {seed}: P = {P}, {len(tr_ev)} training events, {len(te_eps)} test events, '
          f'mu_post (z units) = {mu_post:.2f}', flush=True)

    # ---- training data: N cell 7 with real blocks (IoT random_sample)
    f0 = lambda L: block(tr_norm_in, L, rng)
    f1 = lambda L: tr_ev_in[rng.integers(len(tr_ev_in))][:L]
    x, y, taus = transient_streams(N_TRAIN, TIMESTEPS, RHO1, RHO2, f0, f1, rng)
    xv, yv, _ = transient_streams(N_VAL, TIMESTEPS, RHO2, RHO1, f0, f1, rng)    # N cell 9: rates swapped
    has = taus[:, 0] < TIMESTEPS
    print(f'  training streams: {has.mean():.0%} contain a change; mean change length '
          f'{(taus[has, 1] - taus[has, 0]).mean():.0f}', flush=True)

    model = Model(P)
    hist = fit(model, x, y, xv, yv, batch_size=BATCH, epochs=EPOCHS, seed=seed)
    os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, 'models', f'{name}.{inp}.s{seed}.pt'))
    model.eval()
    del x, y, xv, yv

    res = {'dataset': name, 'label': d.label, 'unit': d.unit, 'input': inp, 'seed': seed, 'P_raw': int(te_eps[0][0].shape[1]),
           'mu_post': mu_post, 'history': hist, 'n_train_events': len(tr_ev), 'n_test_events': len(te_eps), 'evals': {}}

    # ---- Experiment 1 (N cells 19, 22) for each (tau1, K) and each construction
    for tau1 in TAU1S:
        for K in KS:
            ev = [x[t:t + K] for x, t in te_eps if len(x) - t >= K]
            out = {}
            # blocks: the authors' construction, N_TEST experiments in chunks
            if ev and max(len(p) for p in te_norm) >= 1:
                mp_d, ma_d, mp_g, ma_g = [], [], [], []
                for c in range(0, N_TEST, 1000):
                    xs, inits = [], []
                    for _ in range(1000):
                        pre = block(te_norm, tau1 - 1 + K, rng)
                        xs.append(np.concatenate([pre[K:], ev[rng.integers(len(ev))]], 0))
                        inits.append(pre[:K])
                    xs, inits = np.stack(xs), np.stack(inits)
                    dd = scores(model, feat(xs))
                    gg = window_dec_stat_streams(z(xs), z(inits), mu_post)
                    if c == 0:  # the maxima shortcut must count exactly what N cell 19 counts
                        hh = np.quantile(dd, np.linspace(0.05, 0.999, 30))
                        fa, miss = experiment1(dd, np.full(len(dd), tau1), np.full(len(dd), tau1 + K), hh)
                        pfa_, pd_, pmd_ = curve(*maxima(dd, tau1), hh)
                        assert np.allclose(fa / len(dd), pfa_) and np.allclose(miss / len(dd), pmd_)
                    for arr, (a, b) in ((dd, (mp_d, ma_d)), (gg, (mp_g, ma_g))):
                        u, v = maxima(arr, tau1)
                        a.append(u)
                        b.append(v)
                cat = np.concatenate
                out['blocks'] = {'n': N_TEST, 'n_events': len(ev),
                                 'DeepQCD': summarize(cat(mp_d), cat(ma_d)), 'WL-CUSUM': summarize(cat(mp_g), cat(ma_g))}
            # native: each test event with its own preceding normal stretch, no splicing
            nat = [(x, t) for x, t in te_eps if len(x) - t >= K and t >= tau1 - 1 + K]
            if nat:
                xs = np.stack([x[t - (tau1 - 1):t + K] for x, t in nat])
                inits = np.stack([x[t - (tau1 - 1) - K:t - (tau1 - 1)] for x, t in nat])
                dd = scores(model, feat(xs))
                gg = window_dec_stat_streams(z(xs), z(inits), mu_post)
                out['native'] = {'n': len(nat), 'n_events': len(nat),
                                 'DeepQCD': summarize(*maxima(dd, tau1)), 'WL-CUSUM': summarize(*maxima(gg, tau1))}
            res['evals'][f'{tau1},{K}'] = out
            b = out.get('blocks', {})
            nv = out.get('native', {})
            print(f'  tau1 {tau1:4d} K {K:2d}  blocks PD@PFA0.05: DeepQCD {b.get("DeepQCD", {}).get("pd_at", {}).get("0.05", float("nan")):.3f}'
                  f'  WL-CUSUM {b.get("WL-CUSUM", {}).get("pd_at", {}).get("0.05", float("nan")):.3f}'
                  f'   native (n={nv.get("n", 0)}): {nv.get("DeepQCD", {}).get("pd_at", {}).get("0.05", float("nan")):.3f}'
                  f' / {nv.get("WL-CUSUM", {}).get("pd_at", {}).get("0.05", float("nan")):.3f}', flush=True)

    # ---- example streams for the page: f0 block, a whole real test event, f0 block (the shape of N cell 7)
    ex = []
    long_ev = [x[t:] for x, t in te_eps if len(x) - t >= 25]
    for k in range(4):
        e = long_ev[(k * 7) % len(long_ev)][:400]
        pre, post = block(te_norm, 300 + 25, rng), block(te_norm, 200, rng)
        xs = np.concatenate([pre[25:], e, post], 0)
        dd = scores(model, feat(xs[None]))[0]
        gg = window_dec_stat_streams(z(xs)[None], z(pre[:25])[None], mu_post)[0]
        ex.append({'tau1': 301, 'tau2': 301 + len(e), 'z': np.round(z(xs), 3).tolist(),
                   'd': np.round(dd, 4).tolist(), 'g': np.round(gg, 3).tolist()})
    res['examples'] = ex
    res['seconds'] = round(time.time() - t_start)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{name}.{inp}.s{seed}.json'), 'w') as f:
        json.dump(res, f)
    print(f'  done in {res["seconds"]} s', flush=True)


# ---------------------------------------------------------------- which datasets qualify

KIND = {  # what the labels of each dataset mark (from realdata.py)
    'skab': 'interval', 'occupancy': 'interval', 'occupancy-nolight': 'interval', 'smd': 'interval', 'hai': 'interval', 'fog': 'interval',
    'fishkiller': 'interval', 'yahoo': 'interval',
    'tep': 'persistent fault', 'cmapss': 'persistent degradation', 'seismic': 'onset only (no end labelled)',
    'keystroke': 'persistent (typist change)', 'iot-mirai': 'persistent (attack continues)',
    'nab': 'points / change points', 'tcpd': 'change points', 'beedance': 'change points', 'hasc': 'change points',
    'pumpdump': 'single pump instant', 'sp500': 'dated onsets (our choice)', 'pmubage': 'onset only (simulated grid)',
}


def survey():
    rows = []
    for name, kind in KIND.items():
        d = realdata.LOADERS[name]()
        r = {'dataset': name, 'label': d.label, 'unit': d.unit, 'kind': kind, 'P': int((d.test or d.train)[0][0].shape[1])}
        if kind == 'interval':
            tr_norm, tr_ev, te_norm, te = pools(d)
            dur_tr = np.array([len(e) for e in tr_ev])
            dur_te = np.array([len(x) - t for x, t in te])
            longest = max([len(p) for p in te_norm])
            r.update(n_train=len(tr_ev), n_test=len(te), med_dur=float(np.median(np.r_[dur_tr, dur_te])),
                     test_ge_K=int((dur_te >= 25).sum()), train_ge_K=int((dur_tr >= 25).sum()),
                     longest_test_normal=int(longest),
                     native_1000=int(sum(len(x) - t >= 25 and t >= 1024 for x, t in te)),
                     native_200=int(sum(len(x) - t >= 25 and t >= 224 for x, t in te)))
            r['C1'] = True
            r['C2'] = bool(np.median(dur_te) >= 25)
            r['C3'] = bool(longest >= 1024)
            r['C4'] = bool(r['train_ge_K'] >= 5 and r['test_ge_K'] >= 5)
        else:
            r.update(C1=False, C2=None, C3=None, C4=None)
        print(r, flush=True)
        rows.append(r)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'survey.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def splice_check(n_rep=20, seed=0):
    """How much the blocks construction changes the data at the change point. For each test event: the distance
    between the mean of the 10 steps before the onset and the first 5 steps after it (standardized units), with the
    real preceding stretch (native) and with a random normal test block in front (blocks). Baseline: the same
    distance between two random normal blocks, i.e. what a splice alone does when nothing changes."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in DATASETS:
        d = realdata.LOADERS[name]()
        tr_norm, tr_ev, te_norm, te = pools(d)
        norm = normalization(tr_norm)
        te_norm = [norm(x) for x in te_norm]
        ev = [(norm(x), t) for x, t in te if len(x) - t >= 25]
        jump = lambda pre, post: float(np.linalg.norm(pre[-10:].mean(0) - post[:5].mean(0)))
        nat = [jump(x[:t], x[t:]) for x, t in ev if t >= 30]
        spl = [jump(block(te_norm, 200, rng), x[t:]) for _ in range(n_rep) for x, t in ev]
        base = [jump(block(te_norm, 200, rng), block(te_norm, 200, rng)) for _ in range(20 * n_rep)]
        out[name] = {'native': float(np.median(nat)), 'spliced': float(np.median(spl)), 'normal_splice': float(np.median(base)),
                     'n_native': len(nat)}
        print(name, out[name], flush=True)
    with open(os.path.join(OUT, 'splice.json'), 'w') as f:
        json.dump(out, f, indent=1)


def table():
    """Markdown summary for notes/transient-real.md: max PD s.t. PFA <= alpha, DeepQCD (mean over seeds) / WL-CUSUM."""
    import glob
    lines = []
    for con, tau1 in (('blocks', 1000), ('native', 1000), ('blocks', 200), ('native', 200)):
        lines += [f'**{con}, tau1 = {tau1}, K = 25** (max PD with PFA <= alpha; DeepQCD all sensors, mean of seeds / '
                  f'DeepQCD on z / WL-CUSUM)', '', '| dataset | n | alpha = 0.01 | alpha = 0.05 | alpha = 0.10 |', '|---|---|---|---|---|']
        for name in DATASETS:
            runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, f'{name}.all.s*.json')))]
            zp = os.path.join(OUT, f'{name}.z.s0.json')
            zr = json.load(open(zp)) if os.path.exists(zp) else None
            if not runs:
                continue
            ev = runs[0]['evals'][f'{tau1},25'].get(con)
            if not ev:
                lines.append(f'| {name} | 0 | - | - | - |')
                continue
            cells = []
            for a in ('0.01', '0.05', '0.1'):
                d = np.mean([r['evals'][f'{tau1},25'][con]['DeepQCD']['pd_at'][a] for r in runs])
                z = zr['evals'][f'{tau1},25'][con]['DeepQCD']['pd_at'][a] if zr else float('nan')
                cells.append(f"{d:.2f} / {z:.2f} / {ev['WL-CUSUM']['pd_at'][a]:.2f}")
            lines.append(f"| {name} | {ev['n']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    if sys.argv[1] == '--table':
        print(table())
    elif sys.argv[1] == '--splice':
        splice_check()
    elif sys.argv[1] == '--survey':
        survey()
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]))

"""
Detecting market makers' stress unwinds from the public tape of a synthetic S&P 500 futures market (mmsim.py).

The question: can a firm watching only the anonymous book and trades tell, quickly and with few false alarms, that some
market maker has started offloading a position (instantly, gradually, passively, by short-gamma hedging) or has pulled its
quotes? And is the detection early enough to be worth anything?

Data (walk-forward): the market starts at the last S&P 500 close (FRED, via mmsim.calibrate) and is simulated day by day,
each day opening at the previous close. Days 1-30 train (the last 3 validate), days 31-40 are held out.

Detectors, all run once per second on the same 9 public features:
    DeepQCD        the paper's LSTM(16) -> Dense(10) -> Dense(1), trained exactly as in the notebooks (original_qcd.fit) on
                   3,200 windows of 3,000 s cut from the training days, labels 1 while any market maker is under stress
    WL-CUSUM       window_dec_stat (original code) over the last 60 s, one per stress policy: the Gaussian log-likelihood
                   direction of that policy against normal data, fitted on the training days; the statistic is the largest
                   of the six statistics, each first divided by its own 95% quantile of the pre-change maximum. It also
                   gets the absolute values of the signed features, because stress can push either way
    flow z-score   |signed volume over the last 30 s| divided by its training standard deviation: the trader's rule of thumb

Test (the paper's transient protocol): for each of the six policies, 300 fresh streams starting at a random time of day,
with one stress event at tau1 = 1200 s and news (a confounder, label 0) arriving at its usual rate throughout; a further
300 streams have a news jump at tau1 and no stress at all. Detectors run from t = 1 with alarms ignored for the first 120 s
(state warm-up, same for all). PFA = alarm before tau1, PD = alarm in [tau1, tau1 + K) with K = 120 s. One threshold per
detector, set so that the pooled PFA over all stress streams equals alpha.

Edge: for detected directional unwinds, how far the price has been pushed from value in the unwind's direction at the alarm
(known only to the simulator), and the result of trading against the unwind for 1 and 5 minutes after the alarm, net of a
1-tick cost; the same after alarms on news, in the news direction. In this simulator a flow-driven price deviation decays
toward value by construction (mmsim.ETA), so the edge measures how early the detector fires, not whether such an edge
exists in real markets.

Run:  .venv/bin/python mm_qcd.py [--quick] [--kappa 0.25]      -> runs/mm/results.json (runs/mm_kappa0.25/ with --kappa)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

import mmsim
from original_qcd import Model, fit, scores, window_dec_stat_streams
from transient_real import projection

QUICK = '--quick' in sys.argv
KAPPA = float(sys.argv[sys.argv.index('--kappa') + 1]) if '--kappa' in sys.argv else 1.0
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'mm' if KAPPA == 1.0 else f'mm_kappa{KAPPA}')
N_TRAIN_DAYS, N_TEST_DAYS = (4, 2) if QUICK else (30, 10)
N_WIN, N_VAL, T_WIN = (64, 32, 3000) if QUICK else (3200, 500, 3000)
EPOCHS = 1 if QUICK else 20
N_STREAMS = 20 if QUICK else 300
TAU1, K, AFTER, WARM = 1200, 120, 900, 120
WL_K, FLOW_W = 60, 30
ALPHAS = (0.01, 0.05, 0.1)
SIGNED = [0, 4, 5, 7]                 # features whose sign is the direction of the pressure
DIRECTIONAL = ('instant', 'twap', 'skew', 'lean')


def walk_forward(cal):
    """Days simulated in sequence, each opening at the previous close (the open of day d needs day d-1's close, so the
    days are generated one batch at a time with the batch's opening level carried forward)."""
    days, S = [], cal['close']
    for d in range(N_TRAIN_DAYS + N_TEST_DAYS):
        sim = mmsim.simulate(mmsim.SESSION, 1000 + d, S, cal['sigma_daily'], kappa_scale=KAPPA)
        days.append(sim)
        S = float(sim['mid'][-1])
        print(f'  day {d + 1:2d}: open {sim["mid"][0]:.2f} close {S:.2f}, {len(sim["events"])} stress events, '
              f'{len(sim["news"])} news, label 1 for {sim["y"].mean():.0%} of the day', flush=True)
    return days


def _stream(args):
    kind, k, j, S, sig = args
    rng = np.random.default_rng(50000 + 1000 * k + j)
    t0 = int(rng.integers(2000, 19000))
    if kind == 'news':
        return mmsim.simulate(TAU1 + AFTER, 90000 + j, S, sig, t0=t0, shock_rate=0.0, force_news=TAU1, kappa_scale=KAPPA)
    return mmsim.simulate(TAU1 + AFTER, 50000 + 1000 * k + j, S, sig, t0=t0, force=(TAU1, k), kappa_scale=KAPPA)


def policy_labels(d):
    """Per second: index of the policy of the active stress event, -1 if none."""
    lab = -np.ones(len(d['y']), int)
    for e in d['events']:
        lab[e['start']:e['end']] = mmsim.POLICIES.index(e['policy'])
    return lab


def flow_stat(X, sd):
    """|signed volume summed over the last FLOW_W seconds| / training sd of that sum."""
    c = np.cumsum(np.c_[np.zeros((len(X), 1)), X[:, :, 5]], axis=1)
    w = c[:, FLOW_W:] - c[:, :-FLOW_W]
    out = np.zeros(X.shape[:2])
    out[:, FLOW_W - 1:] = np.abs(w) / sd
    return out


def first_alarm(stat, h, start):
    """First index >= start where stat >= h, or -1."""
    hit = np.flatnonzero(stat[start:] >= h)
    return int(hit[0]) + start if len(hit) else -1


def main():
    t_start = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    cal = mmsim.calibrate(os.path.join(ROOT, 'data', 'mm', 'SP500.csv'))
    print(f'calibration: S&P 500 close {cal["close"]} on {cal["date"]}, daily volatility {cal["sigma_daily"]:.4%}', flush=True)

    # ---- walk-forward days
    days = walk_forward(cal)
    train, test = days[:N_TRAIN_DAYS], days[N_TRAIN_DAYS:]
    Xtr = np.concatenate([d['X'] for d in train])
    ytr = np.concatenate([d['y'] for d in train])
    mu, sd = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].std(0) + 1e-6       # do_normalization: normal training data
    norm = lambda X: np.clip((X - mu) / sd, -10, 10).astype(np.float32)

    # ---- test streams
    S_test = float(test[-1]['mid'][-1])
    jobs = [('mm', k, j, S_test, cal['sigma_daily']) for k in range(len(mmsim.MM)) for j in range(N_STREAMS)]
    jobs += [('news', 9, j, S_test, cal['sigma_daily']) for j in range(N_STREAMS)]
    with Pool(4) as pool:
        sims = pool.map(_stream, jobs, chunksize=8)
    print(f'simulated {len(sims)} test streams ({time.time() - t_start:.0f} s)', flush=True)
    # ---- DeepQCD: native windows of the training days
    rng = np.random.default_rng(0)
    fit_days, val_days = train[:-3] if len(train) > 3 else train, train[-3:]

    def windows(ds, n):
        xs, ys = np.empty((n, T_WIN, len(mmsim.FEATURES)), np.float32), np.empty((n, T_WIN), np.float32)
        for i in range(n):
            d = ds[rng.integers(len(ds))]
            a = rng.integers(0, len(d['y']) - T_WIN)
            xs[i], ys[i] = norm(d['X'][a:a + T_WIN]), d['y'][a:a + T_WIN]
        return xs, ys
    x, y = windows(fit_days, N_WIN)
    xv, yv = windows(val_days, N_VAL)
    print(f'training DeepQCD on {N_WIN} windows of {T_WIN} s ({y.mean():.0%} labelled 1)', flush=True)
    torch.manual_seed(0)
    model = Model(len(mmsim.FEATURES))
    hist = fit(model, x, y, xv, yv, epochs=EPOCHS, seed=0)
    torch.save(model.state_dict(), os.path.join(OUT, 'deepqcd.pt'))
    model.eval()
    del x, y, xv, yv

    # ---- benchmark models fitted on the training days
    aug = lambda Z: np.concatenate([Z, np.abs(Z[..., SIGNED])], axis=-1)
    Ztr = aug(norm(Xtr))
    z, mu_post = projection([Ztr[ytr == 0]], [Ztr[ytr == 1]])
    pol = np.concatenate([policy_labels(d) for d in train])
    per = [projection([Ztr[ytr == 0]], [Ztr[pol == p]]) for p in range(len(mmsim.MM)) if (pol == p).sum() > 100]
    c = np.cumsum(np.r_[0, Xtr[:, 5]])
    flow_sd = float((c[FLOW_W:] - c[:-FLOW_W])[ytr[FLOW_W - 1:] == 0].std())
    print(f'WL-CUSUM: mu_post = {mu_post:.2f}; flow z-score: sd of {FLOW_W}-s signed volume = {flow_sd:.1f}', flush=True)

    X = np.stack([s['X'] for s in sims])
    Xn = norm(X)
    groups = [m[5] for m in mmsim.MM for _ in range(N_STREAMS)] + ['news'] * N_STREAMS
    groups = np.array(groups)
    is_mm = groups != 'news'
    A = aug(Xn)
    # one window-limited CUSUM per policy; each is put on the same false-alarm scale (its 95% quantile of the
    # pre-change maximum) before taking the largest, so that no single policy's statistic sets the threshold for all
    subs = [window_dec_stat_streams(zp(A), zp(A[:, :WL_K]), m) for zp, m in per]
    scale = [max(float(np.quantile(g[is_mm, WARM:TAU1].max(1), 0.95)), 1e-9) for g in subs]
    stats = {'DeepQCD': scores(model, Xn),
             'WL-CUSUM': np.max([g / c for g, c in zip(subs, scale)], axis=0),
             'flow z-score': flow_stat(X, flow_sd)}

    res = {'calibration': cal, 'features': mmsim.FEATURES, 'makers': [list(m) for m in mmsim.MM], 'history': hist,
           'settings': {'kappa_scale': KAPPA, 'tau1': TAU1, 'K': K, 'warm': WARM, 'n_streams': N_STREAMS, 'train_days': N_TRAIN_DAYS,
                        'test_days': N_TEST_DAYS, 'wl_k': WL_K, 'flow_w': FLOW_W, 'mu_post': mu_post},
           'days': [{'open': float(d['mid'][0]), 'close': float(d['mid'][-1]), 'events': len(d['events']),
                     'news': len(d['news']), 'frac': float(d['y'].mean())} for d in days],
           'detectors': {}}
    for name, st in stats.items():
        m_pre = st[:, WARM:TAU1].max(1)
        m_in = st[:, TAU1:TAU1 + K].max(1)
        h_grid = np.unique(np.quantile(m_pre[is_mm], np.linspace(0, 1, 301)))
        curves = {}
        for g in [m[5] for m in mmsim.MM] + ['news']:
            sel = groups == g
            pfa = (m_pre[sel][None, :] >= h_grid[:, None]).mean(1)
            pd = ((m_pre[sel][None, :] < h_grid[:, None]) & (m_in[sel][None, :] >= h_grid[:, None])).mean(1)
            o = np.argsort(pfa)
            curves[g] = {'pfa': np.round(pfa[o], 4).tolist(), 'pd': np.round(pd[o], 4).tolist()}
        at = {}
        for a in ALPHAS:
            h = float(np.quantile(m_pre[is_mm], 1 - a))
            row = {'h': h}
            for g in [m[5] for m in mmsim.MM] + ['news']:
                sel = np.flatnonzero(groups == g)
                det = [i for i in sel if m_pre[i] < h <= m_in[i]]
                delays = [first_alarm(st[i], h, TAU1) - TAU1 for i in det]
                row[g] = {'pd': len(det) / len(sel), 'pfa': float((m_pre[sel] >= h).mean()),
                          'delay_median': float(np.median(delays)) if delays else None}
            at[str(a)] = row
        # edge at alpha = 0.05
        h = at['0.05']['h']
        edge = {}
        for g in list(DIRECTIONAL) + ['news']:
            rows = []
            for i in np.flatnonzero(groups == g):
                if not (m_pre[i] < h <= m_in[i]):
                    continue
                s = sims[i]
                G = first_alarm(st[i], h, TAU1)
                if g == 'news':
                    side = int(np.sign(s['news'][[n['t'] for n in s['news']].index(TAU1)]['jump'])) if any(n['t'] == TAU1 for n in s['news']) else 0
                    end = TAU1 + K
                else:
                    ev = [e for e in s['events'] if e['start'] == TAU1][0]
                    side, end = ev['side'], ev['end']
                if side == 0:
                    continue
                mid, gap = s['mid'], s['gap']
                n = len(mid) - 1
                fade = lambda a, b: -side * (mid[min(b, n)] - mid[min(a, n)]) / mmsim.TICK - 1.0   # trade against it, 1 tick cost
                rows.append([side * gap[G], fade(G, G + 60), fade(G, G + 300), side * gap[min(TAU1 + 60, n)], side * gap[min(end, n)], G - TAU1])
            r = np.array(rows) if rows else np.zeros((0, 6))
            mean = lambda c: float(r[:, c].mean()) if len(r) else None
            se = lambda c: float(r[:, c].std() / np.sqrt(len(r))) if len(r) > 1 else None
            edge[g] = {'n': len(r), 'gap_at_alarm': mean(0), 'gap_at_alarm_se': se(0), 'fade_60': mean(1), 'fade_60_se': se(1),
                       'fade_300': mean(2), 'fade_300_se': se(2), 'gap_onset_60': mean(3), 'gap_at_end': mean(4), 'delay_mean': mean(5)}
        res['detectors'][name] = {'curves': curves, 'at': at, 'edge': edge}
        print(f'{name}: PD at pooled PFA 0.05 -> ' + ', '.join(f'{g} {at["0.05"][g]["pd"]:.2f}' for g in [m[5] for m in mmsim.MM])
              + f' | alarms after news {at["0.05"]["news"]["pd"]:.2f}', flush=True)

    # ---- examples for the page: one stream per policy and one news stream, 5 min before to 10 min after tau1
    ex = {}
    for g in [m[5] for m in mmsim.MM] + ['news']:
        i = int(np.flatnonzero(groups == g)[0])
        a, b = TAU1 - 300, TAU1 + 600
        s = sims[i]
        ev = [e for e in s['events'] if e['start'] == TAU1]
        ex[g] = {'mid': np.round((s['mid'][a:b] - s['mid'][TAU1]) / mmsim.TICK, 2).tolist(),
                 'sv': X[i, a:b, 5].astype(int).tolist(), 'imb': np.round(X[i, a:b, 4], 3).tolist(),
                 'd': np.round(stats['DeepQCD'][i, a:b], 4).tolist(), 'y': s['y'][a:b].astype(int).tolist(),
                 'event': ev[0] if ev else None}
    res['examples'] = ex
    # ---- one held-out day for the page, at 10-second resolution
    d0 = test[0]
    step = 10
    res['day'] = {'mid': np.round(d0['mid'][::step], 2).tolist(), 'y': d0['y'][::step].astype(int).tolist(),
                  'inventory': d0['inventory'][::30].T.tolist(), 'events': d0['events'], 'news': d0['news'],
                  'd': np.round(scores(model, norm(d0['X'])[None])[0][::step], 4).tolist()}
    res['seconds'] = round(time.time() - t_start)
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(res, f)
    print(f'done in {res["seconds"]} s')


if __name__ == '__main__':
    main()

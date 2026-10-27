"""
Final protocol for the bank studies, at scale: laundering on IBM AMLworld (HI-Small, LI-Small, HI-Medium) and card
fraud on IBM TabFormer and the Sparkov simulator (data: bank_data.py). The network, its training and the transient
metrics are those of the use cases (usecases_common.py, original_qcd.py). Five things are new:

1. Three periods: training before p0, calibration [p0, p1), test [p1, p2). Every threshold is chosen on the
   calibration period (the paper's Eq. 11 there) and applied unchanged to the test period, which reports the realized
   PFA next to PD. The paper's own choice, Eq. 11 on the test streams, is kept as 'oracle'.
2. DeepQCD with five seeds, and their ensemble (the mean of the five outputs): the detector a bank would deploy.
3. Bootstrap intervals (1,000 resamples of the event streams and of the normal windows) for PD and PFA, and for the
   paired difference in PD between the DeepQCD ensemble and gradient boosting.
4. Streaming: every account or card is scored at every transaction of the calibration and test periods, from its last
   PRE + 1 transactions, as a bank would run the alarm. Thresholds are set in the calibration period for a target
   number of false alert-days per 1,000 active entity-days (an entity-day is active if the entity transacts that day;
   an alert-day is false if no flagged span of the entity covers the alerting transaction). The test period reports
   the realized rate, the share of laundering accounts / fraud bursts alerted while active, the share of their money
   that came at or after the first alert, and the flagged transactions before it. Normal entities are subsampled at
   `rate` and weighted back.
5. --curve: a learning curve: DeepQCD (two seeds) and gradient boosting trained on 10%, 30% and 100% of the training
   events (and of the normal windows).

Episodes. Flagged rows of one entity at most `gap` rows apart form an episode (AMLworld: one per account, from its
first to its last laundering transfer). Training labels are 1 from the first flagged row to one after the last, at
most maxch + slack rows (as in the use cases). The streaming span is the whole episode plus slack.

Run:  .venv/bin/python bank_final.py <dataset> [--quick] [--curve]      -> runs/bank/<dataset>[_curve].json
"""
import json
import os
import sys
import time

import numpy as np
import torch

import bank_data as bd
import usecases_common as uc
from original_qcd import Model, fit, normalization, scores
from transient_real import projection

DATA = sys.argv[1]
QUICK = '--quick' in sys.argv
CURVE = '--curve' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'bank')
SEEDS = (0,) if QUICK else (0, 1, 2, 3, 4)
EPOCHS = 2 if QUICK else 20
ALPHAS = (0.01, 0.05, 0.1)
TARGETS = (1, 5, 20)                       # false alert-days per 1,000 active entity-days
B = 200 if QUICK else 1000
AML_P = dict(pre=15, maxch=10, slack=1, warm=3, wl_k=5, gap=None, wins=(5, 15), n_norm=20000, n_no=8000, unit='account')
CARD_P = dict(pre=30, maxch=30, slack=1, warm=5, wl_k=5, gap=20, wins=(5, 15), n_norm=25000, n_no=8000, unit='card')
CFG = {'aml-hi-small': dict(AML_P, periods=('2022-09-06', '2022-09-08', '2022-09-11'), rate=1.0),
       'aml-li-small': dict(AML_P, periods=('2022-09-06', '2022-09-08', '2022-09-11'), rate=1.0),
       'aml-hi-medium': dict(AML_P, periods=('2022-09-10', '2022-09-13', '2022-09-17'), rate=0.2, n_norm=40000),
       'tabformer': dict(CARD_P, periods=('2014-01-01', '2016-01-01', '2021-01-01'), rate=0.5),
       'sparkov': dict(CARD_P, periods=('2012-09-01', '2012-11-01', '2013-03-11'), rate=1.0, n_norm=20000)}


def rules_for(name, F):
    roll = lambda a, k: np.cumsum(a, 1) - np.c_[np.zeros((len(a), k)), np.cumsum(a, 1)[:, :-k]]
    if name.startswith('aml'):
        def above_usual(A):
            usual = np.cumsum(A[:, :, F['log_usd']], 1) / np.arange(1, A.shape[1] + 1)[None, :]
            return A[:, :, F['log_usd']] - np.c_[usual[:, :1], usual[:, :-1]]
        return {'Rule: ACH with a new counterparty': lambda A: A[:, :, F['ACH']] * A[:, :, F['new_cp']],
                'Rule: amount above usual': above_usual,
                'Rule: new counterparties, last 5': lambda A: roll(A[:, :, F['new_cp']], 5),
                'Rule: hub contact': lambda A: A[:, :, F['cp_degree']],
                'Rule: cross-currency': lambda A: A[:, :, F['cross_cur']]}
    if name == 'tabformer':
        return {'Rule: online, above usual amount': lambda A: A[:, :, F['online']] * A[:, :, F['amount_z']],
                'Rule: new state or country': lambda A: np.maximum(A[:, :, F['new_state']], A[:, :, F['foreign']]),
                'Rule: new merchant, above usual amount': lambda A: A[:, :, F['new_merchant']] * A[:, :, F['amount_z']],
                'Rule: transaction errors': lambda A: A[:, :, F['error']],
                'Rule: velocity (24 h)': lambda A: A[:, :, F['tx_24h']],
                'Rule: above usual amount': lambda A: A[:, :, F['amount_z']]}
    return {'Rule: above usual amount': lambda A: A[:, :, F['amount_z']],
            'Rule: night, above usual amount': lambda A: A[:, :, F['night']] * A[:, :, F['amount_z']],
            'Rule: online, above usual amount': lambda A: A[:, :, F['online']] * A[:, :, F['amount_z']],
            'Rule: risky category, above usual amount': lambda A: A[:, :, F['risky_category']] * A[:, :, F['amount_z']],
            'Rule: velocity (24 h)': lambda A: A[:, :, F['tx_24h']],
            'Rule: far from home': lambda A: A[:, :, F['log_km_home']]}


def episodes(d, gap, slack, maxch):
    """Episodes of flagged rows: entity, first row a, training end b (capped), span end (whole episode + slack)."""
    flag, start = d['flag'], d['start']
    N = len(flag)
    end_of = np.r_[start[1:], N]
    fl = np.flatnonzero(flag)
    ent = np.searchsorted(start, fl, side='right') - 1
    brk = np.r_[True, ent[1:] != ent[:-1]]
    if gap is not None:
        brk |= np.r_[True, np.diff(fl) > gap]
    first, last = fl[brk], fl[np.r_[brk[1:], True]]
    e_ent = ent[brk]
    e_end = end_of[e_ent]
    b = np.minimum(np.minimum(last + 1 + slack, first + maxch + slack), e_end)
    span = np.minimum(last + 1 + slack, e_end)
    return e_ent, first, b, span, last


def main():
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get('QCD_THREADS', 4)))
    cfg = CFG[DATA]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    d = bd.load(DATA)
    for k in ('tx', 'fmt', 'fmts'):
        d.pop(k, None)
    X_all, start, ts, flag, usd = d['X'], d['start'], d['ts'], d['flag'], d['usd']
    N, P = X_all.shape
    F = {f: i for i, f in enumerate(d['features'])}
    pre, maxch, slack, warm, K = cfg['pre'], cfg['maxch'], cfg['slack'], cfg['warm'], cfg['wl_k']
    L = pre + maxch + slack
    p0, p1, p2 = (np.datetime64(p, 's').astype(np.int64) for p in cfg['periods'])
    ent_of = np.repeat(np.arange(len(start)), np.diff(np.r_[start, N]))
    within = np.arange(N) - start[ent_of]
    e_ent, ea, eb, espan, elast = episodes(d, cfg['gap'], slack, maxch)
    lab = np.zeros(N, np.float32)
    dirty = np.zeros(N + 1, np.int64)                      # rows that may not appear in normal windows
    for a, b in zip(ea, eb):
        lab[a:b] = 1
    if cfg['gap'] is None:                                  # AML: every row of an involved account
        inv = np.zeros(len(start), bool)
        inv[e_ent] = True
        dmask = inv[ent_of]
    else:                                                   # cards: within gap rows of a flagged row
        dmask = np.zeros(N, bool)
        dd = np.zeros(N + 1, np.int64)
        lo_ = np.maximum(start[e_ent], ea - cfg['gap'])
        hi_ = np.minimum(np.r_[start[1:], N][e_ent], elast + cfg['gap'] + 1)
        np.add.at(dd, lo_, 1)
        np.add.at(dd, hi_, -1)
        dmask = np.cumsum(dd[:-1]) > 0
    dirty[1:] = np.cumsum(dmask)
    clean = lambda st, n: (dirty[st + n] - dirty[st]) == 0
    per = np.where(ts[ea] < p0, 0, np.where(ts[ea] < p1, 1, np.where(ts[ea] < p2, 2, 3)))
    if QUICK:
        keep = rng.random(len(ea)) < 0.3
        e_ent, ea, eb, espan, elast, per = e_ent[keep], ea[keep], eb[keep], espan[keep], elast[keep], per[keep]
    print(f'{DATA}: {N:,} steps, {len(start):,} {cfg["unit"]}s, {len(ea):,} episodes '
          f'(train {np.sum(per == 0)}, calibration {np.sum(per == 1)}, test {np.sum(per == 2)}, after {np.sum(per == 3)})', flush=True)

    # event kind for breakdowns: the pattern (AML) or online (cards) of the episode
    kinds = d['kinds']
    if DATA.startswith('aml'):
        ek = []
        for a, s_ in zip(ea, espan):
            k = d['kind'][a:s_][flag[a:s_]]
            k = k[k > 1]
            ek.append(kinds[np.bincount(k).argmax()] if len(k) else 'unlisted')
        ek = np.array(ek)
        efmt = np.array(['ACH' if X_all[a, F['ACH']] else 'not ACH' for a in ea])
    else:
        ek = np.array([kinds[d['kind'][a]] for a in ea])
        efmt = None

    # ---------------------------------------------------------------- windows
    def train_windows(frac=1.0, r=None):
        r = r or np.random.default_rng(1)
        xs, ys = [], []
        idx = np.flatnonzero(per == 0)
        if frac < 1:
            idx = r.choice(idx, max(10, int(frac * len(idx))), replace=False)
        for i in idx:
            s_, e_ = start[e_ent[i]], np.r_[start[1:], N][e_ent[i]]
            for _ in range(3):
                st = ea[i] - int(r.integers(5, pre + 1))
                if st >= s_ and st + L <= e_:
                    xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
        n_norm = int(frac * (4000 if QUICK else cfg['n_norm']))
        cand = r.choice(np.flatnonzero(ts < p0), 6 * n_norm)
        cand = cand[cand + L <= N]
        ok = (ent_of[cand] == ent_of[cand + L - 1]) & (ts[cand + L - 1] < p0) & clean(cand, L)
        for st in cand[ok][:n_norm]:
            xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
        return np.stack(xs), np.stack(ys)

    def stream_set(period, r):
        X, tau1, tau2, ev = [], [], [], []
        for i in np.flatnonzero(per == period):
            a = ea[i]
            s_ = start[e_ent[i]]
            if a - s_ < pre or flag[a - pre:a].any() or lab[a - pre:a].any():
                continue
            seg = np.zeros((L, P), np.float32)
            seg[:eb[i] - a + pre] = X_all[a - pre:eb[i]]
            X.append(seg); tau1.append(pre); tau2.append(eb[i] - a + pre); ev.append(i)
        n_ev = len(X)
        lo, hi = (p0, p1) if period == 1 else (p1, p2)
        n_no = 2000 if QUICK else cfg['n_no']
        cand = r.choice(np.flatnonzero((ts >= lo) & (ts < hi)), 8 * n_no)
        cand = cand[cand + pre <= N]
        ok = (ent_of[cand] == ent_of[cand + pre - 1]) & (ts[cand + pre - 1] < hi) & clean(cand, pre)
        for st in cand[ok][:n_no]:
            seg = np.zeros((L, P), np.float32)
            seg[:pre] = X_all[st:st + pre]
            X.append(seg); tau1.append(pre); tau2.append(pre)
        X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
        valid = np.arange(L)[None, :] < np.where(np.arange(len(X)) < n_ev, tau2, pre)[:, None]
        return dict(X=X, tau1=tau1, tau2=tau2, n_ev=n_ev, valid=valid, ev=np.array(ev, int))

    xs, ys = train_windows()
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)
    sets = {'cal': stream_set(1, np.random.default_rng(2)), 'test': stream_set(2, np.random.default_rng(3))}
    for s in sets.values():
        s['Xn'] = norm(s['X'])
    print(f'training windows {len(xs):,} ({(ys.max(1) > 0).mean():.0%} with an event); calibration {sets["cal"]["n_ev"]} events, '
          f'test {sets["test"]["n_ev"]} events ({time.time() - t0:.0f} s)', flush=True)

    if CURVE:
        return learning_curve(xs, ys, norm, sets, cfg, train_windows, t0)

    # ---------------------------------------------------------------- detectors
    nv = len(xs) // 7
    perm = np.random.default_rng(0).permutation(len(xs))
    va, tr = perm[:nv], perm[nv:]
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = Model(P)
        fit(m, xs_n[tr], ys[tr], xs_n[va], ys[va], epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
        m.eval()
        models.append(m)
        print(f'  DeepQCD seed {seed} trained ({time.time() - t0:.0f} s)', flush=True)
    bst = uc.gbm_fit(xs_n, ys, lags=5, wins=cfg['wins'], rounds=4000)
    w_lr = uc.logistic_fit(xs_n, ys)
    z, mu = projection([x[y == 0] for x, y in zip(xs_n, ys) if (y == 0).any()],
                       [x[y == 1] for x, y in zip(xs_n, ys) if (y == 1).any()])
    rules = rules_for(DATA, F)
    nrm = xs[[y.max() == 0 for y in ys]]
    scale = {k: max(float(np.quantile(fn(nrm)[:, warm:], 0.99)), 1e-3) for k, fn in rules.items()}

    def detect(Xn, Xr):
        out = {}
        ds = [scores(m, Xn, chunk=4000) for m in models]
        for k, v in enumerate(ds):
            out[f'DeepQCD s{k}'] = v
        out['DeepQCD'] = np.mean(ds, 0)
        out['Gradient boosting (lags)'] = uc.gbm_score(bst, Xn, 5, cfg['wins'])
        out['Logistic (this step only)'] = uc.logistic_score(w_lr, Xn)
        zz = z(Xn)
        g = np.full(zz.shape, -np.inf)
        for t in range(K - 1, zz.shape[1]):
            r_ = np.zeros(len(zz))
            for i in range(t - K + 1, t + 1):
                r_ = np.maximum(0, r_ + zz[:, i] - mu / 2)
            g[:, t] = r_
        out['WL-CUSUM'] = g
        for k, fn in rules.items():
            out[k] = fn(Xr).astype(float)
        out['Rules combined'] = np.max([out[k] / scale[k] for k in rules], 0)
        return out

    def detect_last(Wn, Wr):
        out = {'DeepQCD': np.mean([scores(m, Wn, chunk=20000)[:, -1] for m in models], 0)}
        out['Gradient boosting (lags)'] = bst.predict(uc.lagged_last(Wn, 5, cfg['wins']), raw_score=True)
        out['Logistic (this step only)'] = Wn[:, -1] @ w_lr[:-1] + w_lr[-1]
        zz = z(Wn[:, -K:])
        r_ = np.zeros(len(zz))
        for i in range(K):
            r_ = np.maximum(0, r_ + zz[:, i] - mu / 2)
        out['WL-CUSUM'] = r_
        for k, fn in rules.items():
            out[k] = fn(Wr)[:, -1].astype(float)
        out['Rules combined'] = np.max([out[k] / scale[k] for k in rules], 0)
        return out

    res = {'settings': dict(cfg, seeds=list(SEEDS), epochs=EPOCHS, features=d['features'], targets=list(TARGETS), alphas=list(ALPHAS)),
           'counts': {'steps': int(N), 'entities': int(len(start)), 'episodes': {k: int(np.sum(per == i)) for i, k in
                                                                                 enumerate(('train', 'calibration', 'test', 'after'))},
                      'train_windows': int(len(xs)), 'cal_events': sets['cal']['n_ev'], 'test_events': sets['test']['n_ev'],
                      'cal_no': int(len(sets['cal']['X']) - sets['cal']['n_ev']), 'test_no': int(len(sets['test']['X']) - sets['test']['n_ev'])},
           'trees': int(bst.best_iteration)}
    D = {k: detect(s['Xn'], s['X']) for k, s in sets.items()}
    print(f'window sets scored ({time.time() - t0:.0f} s)', flush=True)
    res['windows'] = window_eval(D, sets, ek, efmt, warm)
    for name in ('DeepQCD', 'Gradient boosting (lags)', 'Logistic (this step only)', 'WL-CUSUM', 'Rules combined'):
        c = res['windows'][name]['calibrated']
        print(f'  {name:28s} calibrated: ' + '  '.join(f'a={a}: PD {c[a]["pd"]:.3f} [{c[a]["pd_ci"][0]:.2f}, {c[a]["pd_ci"][1]:.2f}] '
                                                      f'PFA {c[a]["pfa"]:.3f}' for a in c) +
              f'   oracle PD@0.05 {res["windows"][name]["oracle"]["pd_at"]["0.05"]:.3f}', flush=True)

    res['streaming'] = streaming(d, cfg, ent_of, within, e_ent, ea, espan, per, ek, norm, detect_last, rules, t0)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, f'{DATA}.json'), 'w'), default=float)
    print(f'done in {res["seconds"]} s', flush=True)


# ---------------------------------------------------------------- window-level evaluation
def maxima(d, s, warm):
    L = d.shape[1]
    d = np.where(s['valid'], d, -np.inf)
    t = np.arange(L)[None, :]
    m_pre = np.where((t >= warm) & (t < s['tau1'][:, None]), d, -np.inf).max(1)
    m_win = np.where((t >= s['tau1'][:, None]) & (t < s['tau2'][:, None]), d, -np.inf).max(1)
    return m_pre, m_win


def eq11(m_pre, m_win, n_ev, alpha, n_grid=1000):
    fin = np.r_[m_pre[np.isfinite(m_pre)], m_win[np.isfinite(m_win)]]
    h = np.unique(np.r_[np.quantile(fin, np.linspace(0, 1, n_grid)), np.inf])
    pfa = (m_pre[None, :] >= h[:, None]).mean(1)
    pd = ((m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])).mean(1)
    ok = pfa <= alpha
    return float(h[np.flatnonzero(ok)[np.argmax(pd[ok])]]) if ok.any() else float(np.inf)


def window_eval(D, sets, ek, efmt, warm):
    out = {}
    st, sc = sets['test'], sets['cal']
    n_ev, n = st['n_ev'], len(st['X'])
    r = np.random.default_rng(7)
    bi = [(r.integers(0, n_ev, n_ev), n_ev + r.integers(0, n - n_ev, n - n_ev)) for _ in range(B)]
    M = {k: maxima(v, st, warm) for k, v in D['test'].items()}
    groups = {'kind': ek[st['ev']]}
    if efmt is not None:
        groups['format'] = efmt[st['ev']]
    for name, dt in D['test'].items():
        o = {}
        oo, _ = uc.evaluate(dt, st['tau1'], st['tau2'], n_ev, warm, st['valid'])
        o['oracle'] = {'pd_at': oo['pd_at'], 'pfa': oo['pfa'], 'pd': oo['pd']}
        mc_pre, mc_win = maxima(D['cal'][name], sc, warm)
        m_pre, m_win = M[name]
        cal = {}
        for a in ALPHAS:
            h = eq11(mc_pre, mc_win, sc['n_ev'], a)
            hit = (m_pre[:n_ev] < h) & (m_win[:n_ev] >= h)
            fa = m_pre >= h
            bs = np.array([[hit[i].mean(), fa[np.r_[i, j]].mean()] for i, j in bi])
            g = {}
            for gname, lab_ in groups.items():
                g[gname] = {str(k): [float(hit[lab_ == k].mean()), int((lab_ == k).sum())] for k in np.unique(lab_)}
            cal[str(a)] = {'h': h, 'pd': float(hit.mean()), 'pfa': float(fa.mean()), 'pfa_cal': float((mc_pre >= h).mean()),
                           'pd_ci': np.quantile(bs[:, 0], [0.025, 0.975]).tolist(), 'pfa_ci': np.quantile(bs[:, 1], [0.025, 0.975]).tolist(),
                           'groups': g, '_hit': hit}
        o['calibrated'] = cal
        out[name] = o
    # paired differences against gradient boosting
    for name in out:
        for a in map(str, ALPHAS):
            h1, h2 = out[name]['calibrated'][a]['_hit'], out['Gradient boosting (lags)']['calibrated'][a]['_hit']
            diff = np.array([h1[i].mean() - h2[i].mean() for i, _ in bi])
            out[name]['calibrated'][a]['diff_vs_gb'] = [float(h1.mean() - h2.mean())] + np.quantile(diff, [0.025, 0.975]).tolist()
    for name in out:
        for a in out[name]['calibrated']:
            out[name]['calibrated'][a].pop('_hit')
    return out


# ---------------------------------------------------------------- streaming evaluation
def streaming(d, cfg, ent_of, within, e_ent, ea, espan, per, ek, norm, detect_last, rules, t0):
    X_all, start, ts, flag, usd = d['X'], d['start'], d['ts'], d['flag'], d['usd']
    N = len(ts)
    pre = cfg['pre']
    p0, p1, p2 = (np.datetime64(p, 's').astype(np.int64) for p in cfg['periods'])
    rate = cfg['rate'] * (0.2 if QUICK else 1.0)
    r = np.random.default_rng(11)
    involved = np.zeros(len(start), bool)
    involved[e_ent] = True
    pick = involved | (r.random(len(start)) < rate)
    w_ent = np.where(involved, 1.0, 1.0 / rate)
    rows = np.flatnonzero(pick[ent_of] & (ts >= p0) & (ts < p2) & (within >= pre))
    span_id = np.full(N, -1, np.int64)
    for i, (a, s_) in enumerate(zip(ea, espan)):
        if per[i] in (1, 2):
            span_id[a:s_] = i
    names = ['DeepQCD', 'Gradient boosting (lags)', 'Logistic (this step only)', 'WL-CUSUM', 'Rules combined'] + list(rules)
    S = {k: np.empty(len(rows), np.float32) for k in names}
    ch = 100_000
    off = np.arange(-pre, 1)
    for c in range(0, len(rows), ch):
        Wr = X_all[rows[c:c + ch, None] + off[None, :]]
        out = detect_last(norm(Wr), Wr)
        for k in names:
            S[k][c:c + ch] = out[k]
        if c // ch % 10 == 0:
            print(f'  streaming {c + len(Wr):,} / {len(rows):,} steps ({time.time() - t0:.0f} s)', flush=True)
    ent, day, sid = ent_of[rows], ts[rows] // 86400, span_id[rows]
    period = np.where(ts[rows] < p1, 1, 2)
    res = {'rows': int(len(rows)), 'rate': rate, 'entities': int(pick.sum()), 'involved': int(involved.sum())}
    # entity-days: active ones, and the highest score on rows outside flagged spans
    key = ent.astype(np.int64) * 100000 + (day - day.min())
    kstart = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    kw = w_ent[ent[kstart]]
    kper = period[kstart]
    # episodes scored from their first transaction: their positions in rows are contiguous
    pos = np.flatnonzero(sid >= 0)
    eb_ = np.r_[True, sid[pos][1:] != sid[pos][:-1]]
    e_lo = pos[eb_]
    e_hi = np.r_[pos[np.flatnonzero(eb_)[1:] - 1] + 1, pos[-1] + 1] if len(pos) else np.array([], int)
    e_id = sid[e_lo]
    ok = (within[ea[e_id]] >= pre) & (rows[e_lo] == ea[e_id])
    e_lo, e_hi, e_id = e_lo[ok], e_hi[ok], e_id[ok]
    test = per[e_id] == 2
    res['episodes'] = {'calibration': int((~test).sum()), 'test': int(test.sum())}
    fl_r = flag[rows]
    usd_f = np.where(fl_r, usd[rows], 0.0)
    out = {}
    for k in names:
        sc = S[k]
        M = np.maximum.reduceat(np.where(sid >= 0, -np.inf, sc), kstart)
        E = np.array([sc[lo:hi].max() for lo, hi in zip(e_lo, e_hi)]) if len(e_lo) else np.array([])
        o = {'active_days': {'calibration': float(kw[kper == 1].sum()), 'test': float(kw[kper == 2].sum())}}
        mk = kper == 1
        Ms, ws = M[mk], kw[mk]
        srt = np.argsort(-Ms)
        cw = np.cumsum(ws[srt])
        th = {}
        for R in TARGETS:
            j = int(np.searchsorted(cw, R / 1000 * ws.sum(), side='right'))
            th[R] = float(np.nextafter(Ms[srt[j]], np.inf)) if j < len(srt) else float(Ms[srt[-1]])
        mt = kper == 2
        Et = E[test]
        o['targets'] = {}
        for R, h in th.items():
            fa = float((kw[mt] * (M[mt] >= h)).sum() / kw[mt].sum() * 1000)
            det = Et >= h
            share, before = [], []
            for lo, hi in zip(e_lo[test], e_hi[test]):
                hit = np.flatnonzero(sc[lo:hi] >= h)
                tot = usd_f[lo:hi].sum()
                if len(hit):
                    share.append(float(usd_f[lo + hit[0]:hi].sum() / max(tot, 1e-9)))
                    before.append(int(fl_r[lo:lo + hit[0]].sum()))
                else:
                    share.append(0.0)
            bs = [det[np.random.default_rng(b).integers(0, len(det), len(det))].mean() for b in range(B)]
            o['targets'][str(R)] = {'h': h, 'false_alert_days_per_1000': fa, 'detected': float(det.mean()),
                                    'detected_ci': np.quantile(bs, [0.025, 0.975]).tolist(),
                                    'money_after_alert': float(np.mean(share)),
                                    'flagged_before_alert': float(np.median(before)) if before else None, '_det': det}
        fin = np.r_[M[mt][np.isfinite(M[mt])], Et[np.isfinite(Et)]]
        hs = np.unique(np.quantile(fin, np.linspace(0, 1, 300))) if len(fin) else np.array([np.inf])
        Mt, wt = M[mt], kw[mt]
        o['curve'] = {'fa': [float((wt * (Mt >= h)).sum() / wt.sum() * 1000) for h in hs], 'det': [float((Et >= h).mean()) for h in hs]}
        ekt = ek[e_id[test]]
        o['by_kind'] = {str(g): [float((Et[ekt == g] >= th[5]).mean()), int((ekt == g).sum())] for g in np.unique(ekt)}
        out[k] = o
    for k in names:
        for R in map(str, TARGETS):
            a, b = out[k]['targets'][R]['_det'], out['Gradient boosting (lags)']['targets'][R]['_det']
            bs = []
            for bb in range(B):
                i = np.random.default_rng(bb).integers(0, len(a), len(a))
                bs.append(a[i].mean() - b[i].mean())
            out[k]['targets'][R]['diff_vs_gb'] = [float(a.mean() - b.mean())] + np.quantile(bs, [0.025, 0.975]).tolist()
    for k in names:
        for R in map(str, TARGETS):
            out[k]['targets'][R].pop('_det')
        t5 = out[k]['targets']['5']
        print(f'  stream {k:28s} at 5 false alert-days per 1,000: realized {t5["false_alert_days_per_1000"]:.1f}, '
              f'detected {t5["detected"]:.3f} [{t5["detected_ci"][0]:.2f}, {t5["detected_ci"][1]:.2f}], money after alert '
              f'{t5["money_after_alert"]:.2f}', flush=True)
    res['detectors'] = out
    return res


# ---------------------------------------------------------------- learning curve
def learning_curve(xs, ys, norm, sets, cfg, train_windows, t0):
    res = {'fractions': {}}
    for frac in (0.1, 0.3, 1.0):
        xs_, ys_ = (xs, ys) if frac == 1.0 else train_windows(frac, np.random.default_rng(int(100 * frac)))
        xn = norm(xs_)
        nv = len(xs_) // 7
        perm = np.random.default_rng(0).permutation(len(xs_))
        va, tr = perm[:nv], perm[nv:]
        r = {'n_events': int(sum(y.max() > 0 for y in ys_)), 'n_windows': int(len(xs_))}
        D = {'DeepQCD': [], 'Gradient boosting (lags)': None}
        ds = {'cal': [], 'test': []}
        for seed in SEEDS[:2]:
            torch.manual_seed(seed)
            m = Model(xs.shape[2])
            fit(m, xn[tr], ys_[tr], xn[va], ys_[va], epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
            m.eval()
            for k in ds:
                ds[k].append(scores(m, sets[k]['Xn'], chunk=4000))
        bst = uc.gbm_fit(xn, ys_, lags=5, wins=cfg['wins'], rounds=4000)
        for name, dc, dt in (('DeepQCD', np.mean(ds['cal'], 0), np.mean(ds['test'], 0)),
                             ('Gradient boosting (lags)', uc.gbm_score(bst, sets['cal']['Xn'], 5, cfg['wins']),
                              uc.gbm_score(bst, sets['test']['Xn'], 5, cfg['wins']))):
            mc = maxima(dc, sets['cal'], cfg['warm'])
            mt = maxima(dt, sets['test'], cfg['warm'])
            n_ev = sets['test']['n_ev']
            r[name] = {}
            for a in ALPHAS:
                h = eq11(*mc, sets['cal']['n_ev'], a)
                r[name][str(a)] = {'pd': float(((mt[0][:n_ev] < h) & (mt[1][:n_ev] >= h)).mean()), 'pfa': float((mt[0] >= h).mean())}
        res['fractions'][str(frac)] = r
        print(f'  fraction {frac}: {r["n_events"]} training events; DeepQCD PD@0.05 {r["DeepQCD"]["0.05"]["pd"]:.3f}, '
              f'GB {r["Gradient boosting (lags)"]["0.05"]["pd"]:.3f} ({time.time() - t0:.0f} s)', flush=True)
    json.dump(res, open(os.path.join(OUT, f'{DATA}_curve.json'), 'w'))


if __name__ == '__main__':
    main()

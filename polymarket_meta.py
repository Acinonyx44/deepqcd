"""
Metaorder detection for a prediction-market maker, with ground truth from on-chain wallets.

A market maker loses money when one trader works a large order in pieces (a metaorder): every child order it fills is
followed by more pressure in the same direction. On anonymous venues nobody can label metaorders, which is why
metaorder detection is usually studied with proprietary broker data. Polymarket settles on chain, so every trade
carries its wallet: metaorders can be labelled exactly in public data, and a detector that sees only anonymous flow
(side, size, time, price) can be trained and scored against them. A detector that works here can then run on
anonymous venues (Kalshi, futures), where the labels do not exist.

Labels (polymarket_fetch.py --full: every trade of 750 random two-outcome markets with $1M-$30M volume, 2025-26):
    metaorder = one wallet, one direction, consecutive trades at most 10 minutes apart, at least 5 trades, at least
    $2,000 in total, lasting at least 1 minute. tau1 = its first minute, tau2 = its last minute + 1.
Anonymous 1-minute features (P = 9):
    0 trades      log(1 + number of trades)
    1 flow        signed $ toward outcome 0, sign(f) log(1 + |f| / $100)
    2 gross       log(1 + $ / 100)
    3 maxsize     log(1 + largest trade / $100)
    4 oneside     |net trade count| / trade count over the last 5 minutes (1 = all trades on one side)
    5 samesize    share of trades in the last 10 minutes within 5% in $ of an earlier same-side trade in that span
    6 regular     coefficient of variation of the gaps between trades over the last 10 minutes (0 = clockwork)
    7 dprice      change of the outcome-0 price in the minute
    8 active10    log(1 + number of trades in the last 10 minutes)

Test protocol (the paper's transient test on native streams): one stream per metaorder that starts after 180 quiet
minutes (no metaorder of any wallet active); the stream is the 180 minutes before tau1 and the metaorder until tau2
(at most 60 minutes). Plus no-change streams: 240-minute windows with no metaorder, at random. PFA = P(alarm before
tau1); PD = P(first alarm in [tau1, tau2)). Alarms count from minute 10 (warm-up). Split by market and time: markets
that closed first train (60%), the rest test.
Value: the price still to come in the metaorder's direction when the alarm sounds, from the alarm to 5 minutes after
the metaorder ends (what a market maker that steps aside at the alarm avoids selling into).

Run:  .venv/bin/python polymarket_meta.py [--quick]      -> runs/polymarket_meta/results.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

from original_qcd import Model, fit, normalization, scores, window_dec_stat_streams
from transient_real import projection
import usecases_common as uc

QUICK = '--quick' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'polymarket')
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'polymarket_meta')
CLOCK = sys.argv[sys.argv.index('--clock') + 1] if '--clock' in sys.argv else 'minute'
TARGET = sys.argv[sys.argv.index('--target') + 1] if '--target' in sys.argv else 'metaorder'
OUT = os.path.join(OUT, CLOCK if TARGET == 'metaorder' else f'{TARGET}_{CLOCK}')
GAP, MIN_N, MIN_USD = 600, 5, 2000
if CLOCK == 'minute':
    PRE, MAXLEN, L_NO, WARM, WL_K = 180, 60, 240, 10, 10
else:  # one step per taker order
    PRE, MAXLEN, L_NO, WARM, WL_K = 150, 60, 210, 10, 10
GBM = '--gbm' in sys.argv                    # only add the gradient-boosting baseline to the saved results
SEEDS = () if GBM else (0,) if QUICK else (0, 1, 2)
EPOCHS = 2 if QUICK else 20
N_TRAIN_WIN = 800 if QUICK else 6000
ALPHAS = (0.01, 0.05, 0.1)
SIZES = {'$2k-5k': (2e3, 5e3), '$5k-20k': (5e3, 2e4), '$20k+': (2e4, 1e12)}
FEATURES = (['trades', 'flow', 'gross', 'maxsize', 'oneside', 'samesize', 'regular', 'dprice', 'active10'] if CLOCK == 'minute' else
            ['side', 'size', 'gap', 'samesize', 'dprice', 'active10', 'imbalance10', 'run'])


def sgnlog(f, s=100):
    return np.sign(f) * np.log1p(np.abs(f) / s)


def roll(a, k):
    """Sum over the last k minutes (inclusive) along axis 0."""
    c = np.cumsum(a, 0)
    out = c.copy()
    out[k:] -= c[:-k]
    return out


def churn_sessions(t, w, s, usd, t0):
    """Round trips by one wallet: trades at most 10 minutes apart, at least 6 trades and 4 changes of side, at least
    $2,000 gross, net at most 20% of gross, lasting at least 1 minute (a wash trader or a scalper churning the market)."""
    o = np.lexsort((t, w))
    t2, w2, s2, u2 = t[o], w[o], s[o], usd[o]
    brk = np.r_[True, (w2[1:] != w2[:-1]) | (t2[1:] - t2[:-1] > GAP)]
    st = np.flatnonzero(brk)
    en = np.r_[st[1:], len(t2)]
    out = []
    for a, b in zip(st, en):
        if b - a < 6:
            continue
        ss, uu = s2[a:b], u2[a:b]
        gross = uu.sum()
        if (ss[1:] != ss[:-1]).sum() >= 4 and gross >= MIN_USD and abs((ss * uu).sum()) <= 0.2 * gross and t2[b - 1] - t2[a] >= 60:
            out.append((int(t2[a] // 60 - t0), int(t2[b - 1] // 60 - t0) + 1, 0, float(gross), int(b - a), int(t2[b - 1] // 60 - t0) + 6, None))
    return out


def market(cid):
    """Minute features, metaorder list and union label of one market."""
    z = np.load(os.path.join(DATA, 'full', cid + '.npz'))
    t, w, s, usd, p0 = z['t'], z['w'], z['s'].astype(float), z['usd'].astype(float), z['p0'].astype(float)
    if len(t) < 500:
        return None
    t0 = t[0] // 60
    mi = t // 60 - t0
    n = int(mi[-1]) + 1
    # ---- labels from wallets
    o = np.lexsort((t, w))
    to, wo, so, uo, mo = t[o], w[o], s[o], usd[o], mi[o]
    brk = np.r_[True, (wo[1:] != wo[:-1]) | (so[1:] != so[:-1]) | (to[1:] - to[:-1] > GAP)]
    gid = np.cumsum(brk) - 1
    cnt, amt = np.bincount(gid), np.bincount(gid, weights=uo)
    starts = np.flatnonzero(brk)
    first_t, last_t = to[starts], np.maximum.reduceat(to, starts)
    keep = (cnt >= MIN_N) & (amt >= MIN_USD) & (last_t - first_t >= 60)
    metas = [(int(first_t[k] // 60 - t0), int(last_t[k] // 60 - t0) + 1, int(so[starts[k]]), float(amt[k]), int(cnt[k]),
              int(last_t[k] // 60 - t0) + 6, None) for k in np.flatnonzero(keep)]
    if TARGET == 'churn':
        metas = churn_sessions(t, w, s, usd, t0)
    lab = np.zeros(n, np.float32)
    for a, b, *_ in metas:
        lab[a:b] = 1
    # ---- anonymous features
    nt = np.bincount(mi, minlength=n).astype(float)
    flow = np.bincount(mi, weights=s * usd, minlength=n)
    gross = np.bincount(mi, weights=usd, minlength=n)
    mx = np.zeros(n)
    np.maximum.at(mx, mi, usd)
    nside = np.bincount(mi, weights=s, minlength=n)
    one = np.where(roll(nt, 5) > 0, np.abs(roll(nside, 5)) / np.maximum(roll(nt, 5), 1), 0)
    same = np.zeros(len(t), bool)                           # child orders of similar size on the same side
    for lag in range(1, 51):
        same[lag:] |= (t[lag:] - t[:-lag] <= GAP) & (s[lag:] == s[:-lag]) & (np.abs(usd[lag:] - usd[:-lag]) <= 0.05 * usd[lag:])
    samesh = np.where(roll(nt, 10) > 0, roll(np.bincount(mi, weights=same, minlength=n), 10) / np.maximum(roll(nt, 10), 1), 0)
    gap = np.r_[0.0, np.diff(t).astype(float)]
    g1, g2, gc = (roll(np.bincount(mi[1:], weights=v, minlength=n), 10) for v in (gap[1:], gap[1:] ** 2, np.ones(len(t) - 1)))
    var = np.maximum(g2 / np.maximum(gc, 1) - (g1 / np.maximum(gc, 1)) ** 2, 0)
    reg = np.where(gc >= 3, np.sqrt(var) / np.maximum(g1 / np.maximum(gc, 1), 1), 1.0)
    last = np.full(n, np.nan)
    last[mi] = p0                                           # last trade of each minute (trades are sorted by time)
    last = last[np.maximum.accumulate(np.where(np.isnan(last), 0, np.arange(n)))]
    dp = np.r_[0, np.diff(last)]
    x = np.stack([np.log1p(nt), sgnlog(flow), np.log1p(gross / 100), np.log1p(mx / 100), one, samesh, np.minimum(reg, 3), dp,
                  np.log1p(roll(nt, 10))], 1).astype(np.float32)
    return {'x': x, 'lab': lab, 'metas': metas, 'price': last.astype(np.float32), 'time': (t0 + np.arange(n)) * 60.0}


def market_orders(cid):
    """The same on the order clock: one step per taker order (fills of one order, same second, wallet and side, merged)."""
    z = np.load(os.path.join(DATA, 'full', cid + '.npz'))
    t, w, s, usd, p0 = z['t'], z['w'], z['s'].astype(int), z['usd'].astype(float), z['p0'].astype(float)
    o = np.lexsort((s, w, t))
    t, w, s, usd, p0 = t[o], w[o], s[o], usd[o], p0[o]
    new = np.r_[True, (t[1:] != t[:-1]) | (w[1:] != w[:-1]) | (s[1:] != s[:-1])]
    idx = np.cumsum(new) - 1
    st = np.flatnonzero(new)
    t, w, s = t[st], w[st], s[st].astype(float)
    p0 = np.r_[p0[st[1:] - 1], p0[-1]]                      # outcome-0 price after the order's last fill
    usd = np.bincount(idx, weights=usd)
    n = len(t)
    if n < 500:
        return None
    # ---- labels from wallets (steps are orders)
    o = np.lexsort((t, w))
    to, wo, so, uo = t[o], w[o], s[o], usd[o]
    brk = np.r_[True, (wo[1:] != wo[:-1]) | (so[1:] != so[:-1]) | (to[1:] - to[:-1] > GAP)]
    gid = np.cumsum(brk) - 1
    cnt, amt = np.bincount(gid), np.bincount(gid, weights=uo)
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(o)] - 1
    keep = (cnt >= MIN_N) & (amt >= MIN_USD) & (to[ends] - to[starts] >= 60)
    metas = []
    lab = np.zeros(n, np.float32)
    for k in np.flatnonzero(keep):
        steps = o[starts[k]:ends[k] + 1]                    # order indices of the children
        a, b = int(steps.min()), int(steps.max()) + 1
        e = int(np.searchsorted(t, to[ends[k]] + 300, side='right')) - 1
        metas.append((a, b, int(so[starts[k]]), float(amt[k]), int(cnt[k]), e, steps.tolist()))
        lab[a:b] = 1
    # ---- anonymous features
    gap = np.r_[60.0, np.diff(t).astype(float)]
    same = np.zeros(n, bool)
    for lag in range(1, 31):
        same[lag:] |= (t[lag:] - t[:-lag] <= GAP) & (s[lag:] == s[:-lag]) & (np.abs(usd[lag:] - usd[:-lag]) <= 0.05 * usd[lag:])
    active = np.arange(n) - np.searchsorted(t, t - 600, side='left') + 1
    imb = roll(s, 10) / np.minimum(np.arange(n) + 1, 10)
    run = np.ones(n)
    for i in range(1, n):
        if s[i] == s[i - 1]:
            run[i] = run[i - 1] + 1
    dp = np.r_[0, np.diff(p0)] * 100
    x = np.stack([s, np.log1p(usd / 100), np.log1p(gap), same.astype(float), np.clip(dp, -10, 10), np.log1p(active), imb,
                  np.log1p(run)], 1).astype(np.float32)
    return {'x': x, 'lab': lab, 'metas': metas, 'price': p0.astype(np.float32), 'time': t.astype(float)}


def onset_streams(mk):
    """Test streams: metaorders that start after PRE quiet minutes. Returns [(x, tau1, tau2, meta, start)]."""
    out = []
    for meta in mk['metas']:
        a, b, e = meta[0], meta[1], meta[5]
        if a >= PRE and mk['lab'][a - PRE:a].max() == 0 and e < len(mk['x']):
            out.append((a - PRE, a, min(b, a + MAXLEN), meta))
    return out


def quiet_windows(mk, rng, k):
    """Up to k random L_NO-minute windows with no metaorder."""
    n = len(mk['x'])
    if n <= L_NO:
        return []
    busy = np.r_[0, np.cumsum(mk['lab'])]
    st = rng.integers(0, n - L_NO, size=4 * k)
    ok = [s for s in st if busy[s + L_NO] - busy[s] == 0]
    return ok[:k]


def first_alarm(d, h, start=WARM):
    a = d[start:] >= h
    return int(np.argmax(a)) + start if a.any() else -1


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    ms = sorted(json.load(open(os.path.join(DATA, 'full_markets.json'))), key=lambda m: m['closed'])
    ms = [m for m in ms if os.path.exists(os.path.join(DATA, 'full', m['cid'] + '.npz'))]
    if QUICK:
        ms = ms[::5]
    mks = {}
    for m in ms:
        r = market(m['cid']) if CLOCK == 'minute' else market_orders(m['cid'])
        if r is not None:
            mks[m['cid']] = r
    cids = [m['cid'] for m in ms if m['cid'] in mks]
    ntr = int(0.6 * len(cids))
    tr_c, te_c = cids[:ntr], cids[ntr:]
    print(f'{len(cids)} markets, {sum(len(mks[c]["metas"]) for c in cids)} metaorders; train {len(tr_c)} markets, test '
          f'{len(te_c)} ({time.time() - t0:.0f} s)', flush=True)

    # ---- training windows (union labels), enriched with metaorder onsets
    L_TR = PRE + MAXLEN
    xs, ys = [], []
    for c in tr_c:
        mk = mks[c]
        n = len(mk['x'])
        if n <= L_TR:
            continue
        for a, b, *_ in mk['metas']:                       # windows containing an onset at a random position
            for _ in range(2):
                s = a - int(rng.integers(30, PRE + 30))
                if 0 <= s and s + L_TR <= n:
                    xs.append(mk['x'][s:s + L_TR])
                    ys.append(mk['lab'][s:s + L_TR])
        for s in rng.integers(0, n - L_TR, size=max(1, N_TRAIN_WIN // (2 * len(tr_c)))):
            xs.append(mk['x'][s:s + L_TR])
            ys.append(mk['lab'][s:s + L_TR])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0] or list(xs))
    xs_n = norm(xs)
    perm = rng.permutation(len(xs))
    nv = len(perm) // 7
    va, tr = perm[:nv], perm[nv:]
    print(f'training windows: {len(xs)} ({(ys.max(1) > 0).mean():.0%} contain a metaorder; {ys.mean():.1%} of minutes labelled 1)',
          flush=True)

    # ---- test streams
    ev, no = [], []
    for c in te_c:
        mk = mks[c]
        for s, a, b, meta in onset_streams(mk):
            ev.append((c, s, a, b, meta))
        for s in quiet_windows(mk, rng, 12):
            no.append((c, s))
    L = PRE + MAXLEN
    X = np.zeros((len(ev) + len(no), L, len(FEATURES)), np.float32)
    tau1 = np.full(len(X), L + 1)
    tau2 = np.full(len(X), L + 1)
    valid = np.zeros((len(X), L), bool)
    for i, (c, s, a, b, meta) in enumerate(ev):
        seg = mks[c]['x'][s:b]
        X[i, :len(seg)] = seg
        valid[i, :len(seg)] = True
        tau1[i], tau2[i] = a - s, b - s
    for k, (c, s) in enumerate(no):
        i = len(ev) + k
        X[i, :L_NO] = mks[c]['x'][s:s + L_NO][:L]
        valid[i, :min(L_NO, L)] = True
    Xn = norm(X)
    n_ev = len(ev)
    print(f'test: {n_ev} metaorder onsets after {PRE} quiet minutes, {len(no)} no-change windows', flush=True)

    # ---- detectors
    det = {}
    pre = [xs_n[i][ys[i] == 0] for i in range(len(xs_n)) if (ys[i] == 0).any()]
    post = [xs_n[i][ys[i] == 1] for i in range(len(xs_n)) if (ys[i] == 1).any()]
    z, mu = projection(pre, post)
    zz = z(Xn)
    det['WL-CUSUM'] = np.c_[np.full((len(zz), WL_K), -np.inf), window_dec_stat_streams(zz[:, WL_K:], zz[:, :WL_K], mu)]
    flow_usd = np.sign(X[:, :, 1]) * np.expm1(np.abs(X[:, :, 1])) * 100 if CLOCK == 'minute' else None
    if CLOCK == 'minute':
        det['Flow, last 5 min'] = np.abs(roll(flow_usd.T, 5).T)
        det['Trades, last 10 min'] = X[:, :, 8].astype(float)
    else:
        size_usd = np.expm1(X[:, :, 1]) * 100
        det['Flow, last 10 orders'] = np.abs(roll((X[:, :, 0] * size_usd).T, 10).T)
        det['Same-side run'] = X[:, :, 7].astype(float)
        det['Similar sizes, last 10 orders'] = roll(X[:, :, 3].T, 10).T
    if GBM:
        det['Gradient boosting (lags)'] = uc.gbm_lags(xs_n, ys, Xn, wins=(10, 30))
    hist, deep = {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Model(len(FEATURES))
        hist[f's{seed}'] = fit(model, xs_n[tr], ys[tr], xs_n[va], ys[va], epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
        model.eval()
        deep.append(scores(model, Xn))
    for k, d in enumerate(deep):
        det[f'DeepQCD s{k}'] = d

    res = {'settings': {'gap_s': GAP, 'min_trades': MIN_N, 'min_usd': MIN_USD, 'pre': PRE, 'maxlen': MAXLEN, 'warm': WARM,
                        'wl_k': WL_K, 'L_no': L_NO, 'features': FEATURES},
           'n_markets': len(cids), 'n_train_markets': len(tr_c), 'n_test_markets': len(te_c),
           'n_metaorders': int(sum(len(mks[c]['metas']) for c in cids)), 'n_train_windows': int(len(xs)),
           'n_test_events': n_ev, 'n_test_no': len(no), 'detectors': {}, 'history': hist}
    t_idx = np.arange(L)[None, :]
    for name, d in det.items():
        d = np.where(valid, d, -np.inf)
        m_pre = np.where((t_idx >= WARM) & (t_idx < tau1[:, None]), d, -np.inf).max(1)
        m_win = np.where((t_idx >= tau1[:, None]) & (t_idx < tau2[:, None]), d, -np.inf).max(1)
        fin = np.r_[m_pre[np.isfinite(m_pre)], m_win[np.isfinite(m_win)]]
        h = np.unique(np.r_[np.quantile(fin, np.linspace(0, 1, 800)), np.inf])
        pfa = (m_pre[None, :] >= h[:, None]).mean(1)
        pd = ((m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])).mean(1)
        out = {'pd_at': {}, 'delay_at': {}, 'impact_at': {}, 'by_size': {}}
        hit_all = (m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])
        usd_ev = np.array([e[4][3] for e in ev])
        for a in ALPHAS:
            ok = pfa <= a
            j = int(np.flatnonzero(ok)[np.argmax(pd[ok])]) if ok.any() else len(h) - 1
            out['pd_at'][str(a)] = float(pd[j])
            out['by_size'][str(a)] = {k: float(hit_all[j, (usd_ev >= lo) & (usd_ev < hi)].mean())
                                      for k, (lo, hi) in SIZES.items() if ((usd_ev >= lo) & (usd_ev < hi)).any()}
            dl, rem, tot, frac = [], [], [], []
            for i, (c, s, aa, b, meta) in enumerate(ev):
                fa = first_alarm(d[i], h[j])
                if fa < 0 or fa < tau1[i] or fa >= tau2[i]:
                    continue
                pr, sgn = mks[c]['price'], meta[2]
                end = meta[5]
                dl.append((mks[c]['time'][s + fa] - mks[c]['time'][aa]) / 60)
                rem.append(100 * sgn * (pr[end] - pr[s + fa]))
                tot.append(100 * sgn * (pr[end] - pr[aa - 1]))
                frac.append((meta[1] - (s + fa)) / max(meta[1] - aa, 1))
            out['delay_at'][str(a)] = float(np.median(dl)) if dl else None
            out['impact_at'][str(a)] = {'n': len(rem), 'remaining_cents': float(np.mean(rem)) if rem else None,
                                        'remaining_se': float(np.std(rem, ddof=1) / np.sqrt(len(rem))) if len(rem) > 1 else None,
                                        'total_cents': float(np.mean(tot)) if tot else None,
                                        'per_event_cents': float(np.sum(rem) / n_ev),          # misses count as 0
                                        'remaining_share': float(np.mean(frac)) if frac else None,
                                        'caught_share': float(np.sum(frac) / n_ev)}
        o = np.argsort(pfa)
        out['pfa'], out['pd'] = np.round(pfa[o], 4).tolist(), np.round(pd[o], 4).tolist()
        res['detectors'][name] = out
        im = out['impact_at']['0.05']
        print(f'{name:22s} PD at PFA ' + '  '.join(f'{a}: {out["pd_at"][a]:.3f}' for a in out['pd_at']) +
              '  by size at 0.05: ' + ', '.join(f'{k} {v:.2f}' for k, v in out['by_size']['0.05'].items()) +
              f'   delay {out["delay_at"]["0.05"]} min, remaining share {im["remaining_share"]}, caught share {im["caught_share"]:.3f}, '
              f'cents per event {im["per_event_cents"]:.3f}', flush=True)
    if GBM:
        uc.add_detector(os.path.join(OUT, 'results.json'), 'Gradient boosting (lags)', res['detectors']['Gradient boosting (lags)'])
        return
    ds = [res['detectors'][f'DeepQCD s{k}'] for k in range(len(SEEDS))]
    res['detectors']['DeepQCD'] = dict(ds[0], pd_at={a: float(np.mean([x['pd_at'][a] for x in ds])) for a in ds[0]['pd_at']},
                                       pd_at_seeds={a: [x['pd_at'][a] for x in ds] for a in ds[0]['pd_at']},
                                       by_size={a: {k: float(np.mean([x['by_size'][a][k] for x in ds])) for k in ds[0]['by_size'][a]}
                                                for a in ds[0]['by_size']})
    # examples for the page
    exi = [i for i in range(n_ev)][:: max(1, n_ev // 8)][:8]
    def example(i):
        c, s0, a, b, meta = ev[i]
        signed = X[i, :tau2[i], 1] if CLOCK == 'minute' else X[i, :tau2[i], 0] * X[i, :tau2[i], 1]
        child = [] if meta[6] is None else [k - s0 for k in meta[6] if s0 <= k < s0 + tau2[i]]
        return {'tau1': int(tau1[i]), 'tau2': int(tau2[i]), 'usd': meta[3], 'n': meta[4], 'side': meta[2], 'child': child,
                'signed': np.round(signed, 3).tolist(), 'd': np.round(deep[0][i, :tau2[i]], 4).tolist(),
                'g': np.round(np.where(np.isfinite(det['WL-CUSUM'][i, :tau2[i]]), det['WL-CUSUM'][i, :tau2[i]], 0), 3).tolist(),
                'minutes': np.round((mks[c]['time'][s0:s0 + tau2[i]] - mks[c]['time'][a]) / 60, 2).tolist()}
    res['examples'] = [example(i) for i in exi]
    fm = {'before': np.concatenate([X[i, WARM:tau1[i]] for i in range(n_ev)]).mean(0),
          'metaorder': np.concatenate([X[i, tau1[i]:tau2[i]] for i in range(n_ev)]).mean(0),
          'no metaorder': X[n_ev:, WARM:L_NO].reshape(-1, len(FEATURES)).mean(0)}
    res['feature_means'] = {k: np.round(v, 3).tolist() for k, v in fm.items()}
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

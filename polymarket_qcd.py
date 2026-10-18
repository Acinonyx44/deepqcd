"""
Polymarket feasibility: does money move before the news? Transient QCD on deadline markets ("X by <date>?").

A deadline market that resolves YES does so when the event happens, at a random time before the deadline; the price
then jumps to about 1. If someone knew first, their buying could show up in the hours before the jump. A market that
resolves NO never has the event. So each market is a stream that either contains a change or does not, and the
question is the paper's transient one: can a detector alarm in the hours just before the public move, without
alarming elsewhere?

Data (polymarket_fetch.py): hourly YES price and every taker trade of at least $1,000 with its wallet.
Hourly features (P = 8):
    0 p          YES price
    1 dlogit     change of logit(p) from the previous hour
    2 flow       signed large-trade flow toward YES, sign(f) log(1 + |f| / $1,000)
    3 gross      log(1 + gross large-trade volume / $1,000)
    4 ntrades    log(1 + number of large trades)
    5 newflow    signed flow of wallets making their first large trade in this market
    6 freshflow  signed flow of wallets first seen in any of our markets within the last 7 days
    7 conc       largest wallet's share of gross large-trade volume over the last 24 hours

Labels (hindsight, from prices only):
    tau_pub = the first hour after which the YES price stays at or above 0.8 until the market closes (YES markets).
    A test stream is the L = 168 hours before tau_pub; the change window is its last W = 24 hours (tau1 = tau_pub - W,
    tau2 = tau_pub). Only "surprises" are used: the price stayed below 0.5 for the first L - W hours.
    No-change streams: consecutive L-hour windows of NO markets, and the earlier windows of YES markets.
PFA = P(alarm before tau1) over all test streams; PD = P(first alarm in [tau1, tau2)) over change streams;
alarms are counted from hour K = 24 (the first K hours start the CUSUM window, as in N cell 22).

Split by time: markets that closed before SPLIT train, the rest test.
Detectors: DeepQCD (the original network and training, all features; and without the price features), the
window-limited CUSUM (window_dec_stat) on the Gaussian-LLR projection of all features and of the flow features only,
and two price rules (price level, rise over the last 24 hours).

Run:  .venv/bin/python polymarket_qcd.py [--quick]      -> runs/polymarket/results.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from original_qcd import Model, fit, normalization, scores, window_dec_stat_streams
from transient_real import curve, projection

QUICK = '--quick' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'polymarket')
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'polymarket')
L, W, K = 168, 24, 24
P_PUB, P_SURPRISE = 0.8, 0.5
SPLIT = '2026-02-01'
SEEDS = (0,) if QUICK else (0, 1, 2)
EPOCHS = 2 if QUICK else 20
ALPHAS = (0.01, 0.05, 0.1)
FEATURES = ['p', 'dlogit', 'flow', 'gross', 'ntrades', 'newflow', 'freshflow', 'conc']
FLOW = [2, 3, 4, 5, 6, 7]
DEADLINE = re.compile(r'\b(by|before)\s+(the\s+)?(end\s+of\s+|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|'
                      r'20\d\d|\d{1,2}/|next|this|monday|tuesday|wednesday|thursday|friday|saturday|sunday|election|'
                      r'inauguration|year|week|month|halloween|christmas|thanksgiving)', re.I)


def sgnlog(f):
    return np.sign(f) * np.log1p(np.abs(f) / 1000)


def load():
    """Hourly feature streams of every downloaded deadline market. Returns a list of dicts."""
    ms = [m for m in json.load(open(os.path.join(DATA, 'deadline_markets.json')))
          if m['end'] is not None and DEADLINE.search(m['q']) and 'between' not in m['q'].lower()
          and os.path.exists(os.path.join(DATA, 'm', m['cid'] + '.json'))]
    recs = {m['cid']: json.load(open(os.path.join(DATA, 'm', m['cid'] + '.json'))) for m in ms}
    first_seen = {}                                      # wallet -> first hour seen in any market
    for r in recs.values():
        for t, w, _ in r['trades']:
            h = t // 3600
            if h < first_seen.get(w, 1 << 40):
                first_seen[w] = h
    out = []
    for m in ms:
        r = recs[m['cid']]
        if len(r['prices']) < 48:
            continue
        pt = np.array(r['prices'], float)
        h0, h1 = int(pt[0, 0] // 3600), int(m['closed'] // 3600)
        if h1 - h0 < 48:
            continue
        n = h1 - h0
        p = np.full(n, np.nan)
        idx = (pt[:, 0] // 3600).astype(int) - h0
        ok = (idx >= 0) & (idx < n)
        p[idx[ok]] = pt[ok, 1]
        if np.isnan(p[0]):
            p[0] = pt[0, 1]
        p = p[np.maximum.accumulate(np.where(np.isnan(p), 0, np.arange(n)))]     # forward fill
        flow, gross, ntr, newf, fresh = (np.zeros(n) for _ in range(5))
        seen, tr = set(), []
        for t, w, usd in r['trades']:
            i = t // 3600 - h0
            if not 0 <= i < n:
                continue
            flow[i] += usd
            gross[i] += abs(usd)
            ntr[i] += 1
            if w not in seen:
                seen.add(w)
                newf[i] += usd
            if t // 3600 - first_seen[w] <= 7 * 24:
                fresh[i] += usd
            tr.append((i, w, abs(usd)))
        g24 = np.convolve(gross, np.ones(24))[:n]
        conc, win, j, k = np.zeros(n), defaultdict(float), 0, 0          # sliding 24-hour window of wallet volumes
        for i in np.flatnonzero(g24 > 0):
            while j < len(tr) and tr[j][0] <= i:
                win[tr[j][1]] += tr[j][2]
                j += 1
            while tr[k][0] <= i - 24:
                win[tr[k][1]] -= tr[k][2]
                k += 1
            conc[i] = max(win.values()) / g24[i]
        lp = np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))
        x = np.stack([p, np.r_[0, np.diff(lp)], sgnlog(flow), np.log1p(gross / 1000), np.log1p(ntr), sgnlog(newf),
                      sgnlog(fresh), conc], 1).astype(np.float32)
        tau_pub = None
        if m['yes']:
            below = np.flatnonzero(p < P_PUB)
            tau_pub = int(below[-1] + 1) if len(below) else 0
            if tau_pub >= n:
                tau_pub = None                                # never settled at 0.8 before closing
        out.append({'q': m['q'], 'cid': m['cid'], 'yes': m['yes'], 'closed': m['closed'], 'h0': h0, 'x': x,
                    'tau_pub': tau_pub, 'volume': m['volume']})
    return out


def windows(mk):
    """Change streams (x, tau1) and no-change streams (x, None) of one market, each L hours long."""
    x, ch, no = mk['x'], [], []
    end = len(x)
    if mk['yes'] and mk['tau_pub'] is not None:
        t = mk['tau_pub']
        if t >= L and x[t - L:t - W, 0].max() < P_SURPRISE:
            ch.append((x[t - L:t], L - W))
        end = t - W                                         # earlier windows of a YES market carry no change
    for s in range(0, end - L + 1, L):
        no.append((x[s:s + L], None))
    return ch, no


def tr_aug(mk):
    """Extra training change streams: the same event with the stream ending 3, 6, ..., 21 hours before tau_pub."""
    x, t, out = mk['x'], mk['tau_pub'], []
    for s in range(3, W, 3):
        if t - s - L >= 0 and x[t - s - L:t - W, 0].max() < P_SURPRISE:
            out.append((x[t - s - L:t - s], L - W + s))
    return out


def stack(ws):
    xs = np.stack([w[0] for w in ws])
    tau1 = np.array([w[1] if w[1] is not None else L + 1 for w in ws])
    return xs, tau1


def labels(tau1):
    t = np.arange(L)[None, :]
    return (t >= tau1[:, None]).astype(np.float32)


def rates(d, tau1, h):
    """PFA over all streams (alarm in [K, tau1)), PD over change streams (first alarm in [tau1, L))."""
    t = np.arange(L)[None, :]
    pre = np.where((t >= K) & (t < tau1[:, None]), d, -np.inf).max(1)
    win = np.where((t >= tau1[:, None]), d, -np.inf).max(1)
    has = tau1 <= L
    pfa = (pre[None, :] >= h[:, None]).mean(1)
    pd = ((pre[None, has] < h[:, None]) & (win[None, has] >= h[:, None])).mean(1)
    return pfa, pd


def first_alarm(d, h):
    """Index of the first alarm at or after K, or L when none."""
    a = d[:, K:] >= h
    return np.where(a.any(1), a.argmax(1) + K, L)


def summarize(d, tau1):
    h = np.unique(np.r_[np.quantile(d[:, K:], np.linspace(0, 1, 600)), np.inf])
    pfa, pd = rates(d, tau1, h)
    has = tau1 <= L
    out = {'pd_at': {}, 'lead_at': {}, 'h_at': {}}
    for a in ALPHAS:
        ok = pfa <= a
        j = int(np.flatnonzero(ok)[np.argmax(pd[ok])]) if ok.any() else len(h) - 1
        out['pd_at'][str(a)] = float(pd[j])
        out['h_at'][str(a)] = float(h[j])
        fa = first_alarm(d[has], h[j])
        hit = (fa >= tau1[has]) & (fa < L)
        out['lead_at'][str(a)] = float(np.median(L - fa[hit])) if hit.any() else None
    o = np.argsort(pfa)
    out['pfa'], out['pd'] = np.round(pfa[o], 4).tolist(), np.round(pd[o], 4).tolist()
    return out


def price_rise(x):
    p = x[:, :, 0]
    mins = np.stack([p[:, max(0, t - 24):t + 1].min(1) for t in range(p.shape[1])], 1)
    return p - mins


def clusters(mks):
    """Markets whose surprise events went public within 12 hours of each other are one real-world event
    (for example the ladder "US strikes Iran by March 1 / March 15 / March 31"). Sets m['event']."""
    ev = sorted((m['h0'] + m['tau_pub'], i) for i, m in enumerate(mks) if windows(m)[0])
    k, last = -1, None
    for t, i in ev:
        if last is None or t - last > 12:
            k += 1
        mks[i]['event'] = k
        last = t
    return k + 1


def rolling_scores(model, xs, burn=L - 24, step=24, chunk=4000):
    """DeepQCD over whole markets the way it was trained, on L-hour streams: the statistic for hours
    [s, s + 24) comes from a run started at s - burn (so it always has 144 to 168 hours of history)."""
    segs, where = [], []
    for k, x in enumerate(xs):
        for s in range(burn, len(x), step):
            seg = x[s - burn:s + step]
            if len(seg) < burn + step:
                seg = np.r_[seg, np.repeat(seg[-1:], burn + step - len(seg), 0)]
            segs.append(seg)
            where.append((k, s))
    d = scores(model, np.stack(segs), chunk=chunk)
    out = [np.full(len(x), -np.inf) for x in xs]
    for (k, s), r in zip(where, d):
        n = min(step, len(xs[k]) - s)
        out[k][s:s + n] = r[burn:burn + n]
    return out


def backtest(stats, mks, h, burn, cost=0.01):
    """Buy YES at the first alarm (statistic >= h after `burn` hours) while the price is still below P_PUB, hold to
    resolution. Returns the profit per trade (outcome - price - cost) of every market that traded."""
    out = []
    for d, m in zip(stats, mks):
        p = m['x'][:, 0]
        ok = np.flatnonzero((np.arange(len(p)) >= burn) & (d >= h) & (p < P_PUB))
        if len(ok):
            out.append(m['yes'] - p[ok[0]] - cost)
    return np.array(out)


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    mks = load()
    n_clusters = clusters(mks)
    split = time.mktime(time.strptime(SPLIT, '%Y-%m-%d'))
    tr_m = [m for m in mks if m['closed'] < split]
    te_m = [m for m in mks if m['closed'] >= split]
    tr_ch, tr_no, te_ch, te_no = [], [], [], []
    for m in tr_m:
        c, n = windows(m)
        tr_ch += c + (tr_aug(m) if c else [])
        tr_no += n
    best = {}                                   # test: one change stream per real-world event, its largest market
    for m in te_m:
        c, n = windows(m)
        te_no += n
        if c and m['volume'] > best.get(m['event'], (0,))[0]:
            best[m['event']] = (m['volume'], c[0], m)
    te_ch = [(w, m) for _, w, m in sorted(best.values(), key=lambda b: b[2]['h0'] + b[2]['tau_pub'])]
    n_ev = {'train_markets': sum(1 for m in tr_m if windows(m)[0]), 'train_events': len({m['event'] for m in tr_m if windows(m)[0]}),
            'test_markets': sum(1 for m in te_m if windows(m)[0]), 'test_events': len(te_ch), 'all_events': n_clusters}
    print(f'{len(mks)} markets ({sum(m["yes"] for m in mks)} YES), {n_clusters} surprise events; train {len(tr_m)} markets, '
          f'{n_ev["train_events"]} events in {n_ev["train_markets"]} markets ({len(tr_ch)} change streams with shifts), '
          f'{len(tr_no)} no-change; test {len(te_m)} markets, {len(te_ch)} events, {len(te_no)} no-change', flush=True)

    # the pre-change distribution f0: hours of no-change training streams; normalization as in the IoT script
    norm = normalization([w[0] for w in tr_no])
    xs_tr, tau_tr = stack(tr_ch + tr_no)
    xs_tr = norm(xs_tr)
    y_tr = labels(tau_tr)
    raw_te, tau_te = stack([w for w, _ in te_ch] + te_no)
    xs_te = norm(raw_te)
    n_ch = len(te_ch)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(xs_tr))
    nv = len(perm) // 7
    va, tr = perm[:nv], perm[nv:]
    res = {'L': L, 'W': W, 'K': K, 'split': SPLIT, 'p_pub': P_PUB, 'p_surprise': P_SURPRISE, 'features': FEATURES,
           'n_markets': len(mks), 'n_yes': int(sum(m['yes'] for m in mks)), 'n_train_markets': len(tr_m),
           'n_test_markets': len(te_m), 'n_events': n_ev, 'n_train_streams': len(xs_tr), 'n_test_no': len(te_no),
           'detectors': {}}

    # Gaussian-LLR projections for window_dec_stat (fitted on training streams: pre = before tau1, post = window)
    def proj(cols):
        pre = [xs_tr[i, :min(tau_tr[i], L)][:, cols] for i in range(len(xs_tr))]
        post = [xs_tr[i, tau_tr[i]:][:, cols] for i in range(len(xs_tr)) if tau_tr[i] < L]
        return projection(pre, post)

    whole = [norm(m['x']) for m in te_m]                      # whole test markets, for the backtest
    Tmax = max(len(x) for x in whole)
    det, full = {}, {}
    for name, cols in (('WL-CUSUM, all features', list(range(len(FEATURES)))), ('WL-CUSUM, flow only', FLOW)):
        z, mu = proj(cols)
        zz = z(xs_te[:, :, cols])
        det[name] = np.c_[np.full((len(zz), K), -np.inf), window_dec_stat_streams(zz[:, K:], zz[:, :K], mu)]
        zw = np.stack([np.r_[z(x[:, cols]), np.zeros(Tmax - len(x), np.float32)] for x in whole])
        gw = window_dec_stat_streams(zw[:, K:], zw[:, :K], mu)
        full[name] = [np.r_[np.full(K, -np.inf), gw[k, :len(x) - K]] for k, x in enumerate(whole)]
        res['detectors'].setdefault(name, {})['mu_post'] = mu
    det['Price level'] = raw_te[:, :, 0].astype(float)
    full['Price level'] = [m['x'][:, 0].astype(float) for m in te_m]
    det['Price rise over 24 h'] = price_rise(raw_te).astype(float)
    full['Price rise over 24 h'] = [price_rise(m['x'][None])[0].astype(float) for m in te_m]
    hist = {}
    for name, cols in (('DeepQCD, all features', list(range(len(FEATURES)))), ('DeepQCD, flow only', FLOW)):
        ds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = Model(len(cols))
            hist[f'{name} s{seed}'] = fit(model, xs_tr[tr][:, :, cols], y_tr[tr], xs_tr[va][:, :, cols], y_tr[va],
                                          epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
            model.eval()
            d = scores(model, xs_te[:, :, cols])
            d[:, :K] = -np.inf
            ds.append(d)
            if seed == SEEDS[0]:
                full[name] = rolling_scores(model, [x[:, cols] for x in whole])
        det[name] = ds

    burn = L - 24
    for name, d in det.items():
        if isinstance(d, list):
            runs = [summarize(di, tau_te) for di in d]
            out = runs[0]
            out['pd_at_seeds'] = {a: [r['pd_at'][a] for r in runs] for a in out['pd_at']}
            out['curves_other_seeds'] = [{'pfa': r['pfa'], 'pd': r['pd']} for r in runs[1:]]
            out['pd_at'] = {a: float(np.mean(v)) for a, v in out['pd_at_seeds'].items()}
        else:
            out = summarize(d, tau_te)
        bt = {}
        for a in map(str, ALPHAS):
            pr = backtest(full[name], te_m, out['h_at'][a], burn)
            bt[a] = {'n': int(len(pr)), 'mean': float(pr.mean()) if len(pr) else None,
                     'se': float(pr.std(ddof=1) / np.sqrt(len(pr))) if len(pr) > 1 else None,
                     'hit': float((pr > 0).mean()) if len(pr) else None}
        out['backtest'] = bt
        res['detectors'].setdefault(name, {}).update(out)
        b = bt['0.05']
        print(f'{name:24s} PD at PFA ' + '  '.join(f'{a}: {out["pd_at"][a]:.3f}' for a in out['pd_at']) +
              f'   lead (h) at 0.05: {out["lead_at"]["0.05"]}   backtest at 0.05: n {b["n"]}, '
              f'{b["mean"] if b["mean"] is None else round(b["mean"], 3)} +- {b["se"] if b["se"] is None else round(b["se"], 3)}', flush=True)
    # reference: buy YES at a random hour (price below P_PUB) of every test market
    rnd = []
    for m in te_m:
        p = m['x'][:, 0]
        ok = np.flatnonzero((np.arange(len(p)) >= burn) & (p < P_PUB))
        if len(ok):
            rnd.append(m['yes'] - p[rng.choice(ok)] - 0.01)
    rnd = np.array(rnd)
    res['backtest_random'] = {'n': int(len(rnd)), 'mean': float(rnd.mean()), 'se': float(rnd.std(ddof=1) / np.sqrt(len(rnd)))}
    print(f'random entry: n {len(rnd)}, {rnd.mean():.3f} +- {res["backtest_random"]["se"]:.3f}', flush=True)
    res['history'] = hist
    res['pd_se'] = {a: float(np.sqrt(max(v, 1e-9) * (1 - v) / max(n_ch, 1)))
                    for a, v in res['detectors']['DeepQCD, all features']['pd_at'].items()}

    # test events for the page: price, flow and the statistics in the 7 days before tau_pub
    d0 = det['DeepQCD, all features'][0]
    g0 = det['WL-CUSUM, all features']
    res['events'] = [{'q': m['q'], 'volume': m['volume'], 'p': np.round(raw_te[i, :, 0], 4).tolist(),
                      'flow': np.round(raw_te[i, :, 2], 3).tolist(), 'newflow': np.round(raw_te[i, :, 5], 3).tolist(),
                      'd': np.round(np.where(np.isfinite(d0[i]), d0[i], 0), 4).tolist(),
                      'g': np.round(np.where(np.isfinite(g0[i]), g0[i], 0), 3).tolist(),
                      'tau_pub_utc': int((m['h0'] + m['tau_pub']) * 3600)} for i, (w, m) in enumerate(te_ch)]
    # what changes in the last day before the public move (raw features; test change streams vs no-change streams)
    res['feature_means'] = {
        'event, days 2-7 before': np.round(np.concatenate([raw_te[i, K:tau_te[i]] for i in range(n_ch)]).mean(0), 3).tolist(),
        'event, last 24 h': np.round(np.concatenate([raw_te[i, tau_te[i]:] for i in range(n_ch)]).mean(0), 3).tolist(),
        'no event': np.round(raw_te[n_ch:, K:].reshape(-1, raw_te.shape[2]).mean(0), 3).tolist()}
    # the same in plain units: per hour, how busy and how one-sided the large trades were
    unlog = lambda f: np.sign(f) * 1000 * np.expm1(np.abs(f))
    groups = {'event, days 2-7 before': np.concatenate([raw_te[i, K:tau_te[i]] for i in range(n_ch)]),
              'event, last 24 h': np.concatenate([raw_te[i, tau_te[i]:] for i in range(n_ch)]),
              'no event': raw_te[n_ch:, K:].reshape(-1, raw_te.shape[2])}
    res['activity'] = {k: {'hours': int(len(g)), 'price': float(g[:, 0].mean()),
                           'share_hours_with_trade': float((g[:, 4] > 0).mean()), 'trades_per_hour': float(np.expm1(g[:, 4]).mean()),
                           'usd_per_hour': float(unlog(g[:, 3]).mean()),
                           'net_toward_yes': float(unlog(g[:, 2]).sum() / max(unlog(g[:, 3]).sum(), 1e-9)),
                           'new_wallets_net_toward_yes': float(unlog(g[:, 5]).sum() / max(unlog(g[:, 3]).sum(), 1e-9)),
                           'fresh_wallets_net_toward_yes': float(unlog(g[:, 6]).sum() / max(unlog(g[:, 3]).sum(), 1e-9))}
                       for k, g in groups.items()}
    for k, v in res['activity'].items():
        print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

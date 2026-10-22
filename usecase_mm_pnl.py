"""
Use case 3, market makers on prediction markets: turn the Polymarket alarms (polymarket_meta.py) into P&L.

Replay of the held-out markets (the same 745 markets and time split as polymarket_meta.py). We are a market maker that
takes the other side of every taker trade while it quotes (the same fraction of every trade, so it cancels). A fill's
P&L is its markout: the outcome-0 price 5 minutes later against the trade price, signed for our side, in cents per
share and weighted by the trade's dollars. Quoting decisions for minute m use only data up to the end of minute m - 1.

Strategies:
    always quote
    step back when busy           pull both sides while trades over 10 minutes are in their top 2% (training data)
    step back on metaorder alarm  pull the side being hit (the direction of the last 5 minutes' flow) while the
                                  WL-CUSUM metaorder statistic is above its threshold
    ... unless churn              the same, but keep quoting while the DeepQCD churn alarm is on
    step back on churn alarm      pull both sides while the DeepQCD churn alarm is on
    step back on either alarm     the union of the two
    oracle                        pull the hit side exactly during labelled metaorders (the ceiling)
Thresholds: set on training markets so that each alarm is on during 2% of the minutes without an episode.
Detectors are trained as in polymarket_meta.py: WL-CUSUM for metaorders (it saved the most there), DeepQCD (seed 0)
for churn (it won there).

Run:  .venv/bin/python usecase_mm_pnl.py [--quick]      -> runs/usecases/mm_pnl.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

QUICK = '--quick' in sys.argv
sys.argv = [sys.argv[0], '--clock', 'minute']
import polymarket_meta as pm                                     # noqa: E402
import usecases_common as uc                                     # noqa: E402
from original_qcd import normalization, scores, window_dec_stat_streams   # noqa: E402
from transient_real import projection                           # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'usecases')
H = 300            # markout horizon, seconds
ALARM_RATE = 0.02
L_TR = pm.PRE + pm.MAXLEN


def windows(mks, cids, key, rng, per_market=12):
    xs, ys = [], []
    for c in cids:
        mk = mks[c]
        n = len(mk['x'])
        if n <= L_TR:
            continue
        for meta in mk[key + '_metas']:
            s = meta[0] - int(rng.integers(30, pm.PRE + 30))
            if 0 <= s and s + L_TR <= n:
                xs.append(mk['x'][s:s + L_TR]); ys.append(mk[key][s:s + L_TR])
        for s in rng.integers(0, n - L_TR, per_market):
            xs.append(mk['x'][s:s + L_TR]); ys.append(mk[key][s:s + L_TR])
    return np.stack(xs), np.stack(ys)


def whole(stat_fn, X):
    """A statistic over a whole market, computed in overlapping 240-minute segments started 180 minutes earlier (as the
    detectors were trained on 240-minute windows)."""
    n, out = len(X), np.full(len(X), -np.inf)
    segs, where = [], []
    for s in range(pm.PRE, n, pm.MAXLEN):
        seg = X[s - pm.PRE:s + pm.MAXLEN]
        if len(seg) < L_TR:
            seg = np.r_[seg, np.zeros((L_TR - len(seg), X.shape[1]), np.float32)]
        segs.append(seg); where.append(s)
    if not segs:
        return out
    d = stat_fn(np.stack(segs))
    for s, r in zip(where, d):
        k = min(pm.MAXLEN, n - s)
        out[s:s + k] = r[pm.PRE:pm.PRE + k]
    return out


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    ms = sorted(json.load(open(os.path.join(pm.DATA, 'full_markets.json'))), key=lambda m: m['closed'])
    if QUICK:
        ms = ms[::12]
    mks, trades = {}, {}
    for m in ms:
        p = os.path.join(pm.DATA, 'full', m['cid'] + '.npz')
        if not os.path.exists(p):
            continue
        r = pm.market(m['cid'])
        if r is None:
            continue
        z = np.load(p)
        t, w, s, usd, p0 = z['t'], z['w'], z['s'].astype(float), z['usd'].astype(float), z['p0'].astype(float)
        t0m = t[0] // 60
        ch = pm.churn_sessions(t, w, s, usd, t0m)
        r['meta'], r['meta_metas'] = r['lab'], r['metas']
        r['churn'] = np.zeros(len(r['x']), np.float32)
        for a, b, *_ in ch:
            r['churn'][a:b] = 1
        r['churn_metas'] = ch
        mks[m['cid']] = r
        trades[m['cid']] = (t, s, usd, p0, t0m)
    cids = [m['cid'] for m in ms if m['cid'] in mks]
    ntr = int(0.6 * len(cids))
    tr_c, te_c = cids[:ntr], cids[ntr:]
    print(f'{len(cids)} markets ({len(te_c)} held out) loaded in {time.time() - t0:.0f} s', flush=True)

    # ---- metaorder detector: WL-CUSUM on the projection fitted to metaorder labels
    xm, ym = windows(mks, tr_c, 'meta', rng)
    norm = normalization([x for x, y in zip(xm, ym) if y.max() == 0])
    xm_n = norm(xm)
    z, mu = projection([x[y == 0] for x, y in zip(xm_n, ym)], [x[y == 1] for x, y in zip(xm_n, ym) if y.max() > 0])
    cusum = lambda S: np.c_[np.full((len(S), pm.WL_K), -np.inf), window_dec_stat_streams(z(norm(S))[:, pm.WL_K:], z(norm(S))[:, :pm.WL_K], mu)]
    # ---- churn detector: DeepQCD trained on churn labels
    xc, yc = windows(mks, tr_c, 'churn', rng)
    _, models = uc.deepqcd(norm(xc), yc, norm(xc[:2]), seeds=(0,), epochs=1 if QUICK else 20, rng=rng)
    deep = lambda S: scores(models[0], norm(S))
    print(f'detectors trained ({time.time() - t0:.0f} s)', flush=True)

    # ---- statistics over whole markets; thresholds from training markets (alarm on 2% of episode-free minutes)
    stats = {}
    for c in cids:
        X = mks[c]['x']
        stats[c] = {'meta': whole(cusum, X), 'churn': whole(deep, X), 'busy': X[:, 8].astype(float)}
    h = {}
    rate = {}
    for k, lab in (('meta', 'meta'), ('churn', 'churn'), ('busy', 'meta')):
        vals = np.concatenate([stats[c][k][(mks[c][lab] == 0) & np.isfinite(stats[c][k])] for c in tr_c])
        h[k] = float(np.quantile(vals, 1 - ALARM_RATE))
        rate[k] = float((vals > h[k]).mean())                          # realized share of episode-free minutes alarmed
    print('thresholds', h, 'alarm rates on training', rate, flush=True)

    # ---- replay
    strategies = ['always quote', 'step back when busy', 'step back on metaorder alarm', 'step back on alarm unless churn',
                  'step back on churn alarm', 'step back on either alarm', 'oracle']
    tot = {k: {'pnl': 0.0, 'usd': 0.0, 'n': 0} for k in strategies}
    by_regime = {k: {'metaorder': [0.0, 0.0], 'churn': [0.0, 0.0], 'other': [0.0, 0.0]} for k in ['always quote']}
    for c in te_c:
        t, s, usd, p0, t0m = trades[c]
        mi = t // 60 - t0m
        fut = np.searchsorted(t, t + H, side='right') - 1                 # last trade within 5 minutes
        ok = t + H <= t[-1]
        mark = -s * (p0[fut] - p0) * 100                                  # our side is -s; cents per share
        S, mk = stats[c], mks[c]
        n = len(mk['x'])
        prev = np.clip(mi - 1, 0, n - 1)                                  # decisions use the previous minute
        meta_on = (S['meta'][prev] > h['meta']) & (mi > 0)             # strict: the CUSUM sits at 0 most of the time
        churn_on = (S['churn'][prev] > h['churn']) & (mi > 0)
        busy_on = (S['busy'][prev] > h['busy']) & (mi > 0)
        flow5 = pm.roll(np.bincount(mi, weights=s * usd, minlength=n), 5)
        hit_side = np.sign(flow5[prev])
        meta_true = mk['meta'][np.clip(mi, 0, n - 1)] > 0
        side_true = np.zeros(len(t))
        for a, b, sg, *_ in mk['meta_metas']:
            sel = (mi >= a) & (mi < b)
            side_true[sel] = sg
        quote = {'always quote': np.ones(len(t), bool),
                 'step back when busy': ~busy_on,
                 'step back on metaorder alarm': ~(meta_on & (s == hit_side)),
                 'step back on alarm unless churn': ~(meta_on & ~churn_on & (s == hit_side)),
                 'step back on churn alarm': ~churn_on,
                 'step back on either alarm': ~((meta_on & (s == hit_side)) | churn_on),
                 'oracle': ~(meta_true & (s == side_true))}
        for k, q in quote.items():
            sel = q & ok
            tot[k]['pnl'] += float(np.sum(mark[sel] * usd[sel]))
            tot[k]['usd'] += float(np.sum(usd[sel]))
            tot[k]['n'] += int(sel.sum())
        churn_true = mk['churn'][np.clip(mi, 0, n - 1)] > 0
        for reg, sel in (('metaorder', meta_true & ok), ('churn', churn_true & ok), ('other', ~meta_true & ~churn_true & ok)):
            by_regime['always quote'][reg][0] += float(np.sum(mark[sel] * usd[sel]))
            by_regime['always quote'][reg][1] += float(np.sum(usd[sel]))
    base = tot['always quote']
    res = {'markets_test': len(te_c), 'horizon_s': H, 'alarm_rate': ALARM_RATE, 'thresholds': h, 'realized_alarm_rates': rate,
           'strategies': {}}
    for k, v in tot.items():
        res['strategies'][k] = {'pnl_cents_x_usd': v['pnl'], 'usd_filled': v['usd'], 'share_of_volume': v['usd'] / base['usd'],
                                'cents_per_share': v['pnl'] / v['usd'], 'pnl_per_100usd_of_all_volume': 100 * v['pnl'] / base['usd'],
                                'loss_cut': (v['pnl'] - base['pnl']) / abs(base['pnl'])}
        print(f'{k:34s} filled {v["usd"] / base["usd"]:.1%} of volume, markout {v["pnl"] / v["usd"]:+.3f} c/share on fills, '
              f'P&L {100 * v["pnl"] / base["usd"]:+.2f} c per $100 of all volume', flush=True)
    res['by_regime'] = {reg: {'cents_per_share': v[0] / max(v[1], 1), 'share_of_volume': v[1] / base['usd']}
                        for reg, v in by_regime['always quote'].items()}
    print('markouts by regime (always quote):', {k: (round(v['cents_per_share'], 3), round(v['share_of_volume'], 3))
                                                 for k, v in res['by_regime'].items()}, flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'mm_pnl.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

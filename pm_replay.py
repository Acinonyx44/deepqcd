"""
Market making on Polymarket at scale: the alarms of pm_final.py turned into P&L on the held-out markets.

We are a market maker that takes the other side of every taker trade while it quotes (the same fraction of every trade).
A fill's P&L is its markout: the outcome-0 price H later against the trade price, signed for our side, in cents per
share, weighted by the trade's dollars (H = 5 minutes, and 30 minutes as a check). Decisions for minute m use data up
to the end of minute m - 1.

Alarms (models saved by pm_final.py; each scored on whole markets in 240-minute segments that start 180 minutes before
the 60 minutes they score, as the detectors were trained): metaorder, churn, informed flow (anonymous and with wallet
features: Polymarket's wallets are public, so a market maker there can compute them), and price jump; DeepQCD (the
five-seed ensemble) and gradient boosting for each. Thresholds come from the calibration markets: each alarm is on for
a given share (2% or 10%) of the decision minutes (minutes before a trade) outside its own episodes.

Strategies: always quote; step back when busy (trades over 10 minutes in their top share); for each alarm, pull the
side being hit (the direction of the last 5 minutes' flow) - or both sides for churn and jumps - while it is on; the
union of the DeepQCD alarms; and two oracles that pull the hit side exactly during true metaorders / informed episodes.
Intervals: 1,000 bootstrap resamples of the test markets.

Run:  .venv/bin/python pm_replay.py [--quick]      -> runs/pm/replay.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

QUICK = '--quick' in sys.argv
sys.argv = [sys.argv[0], '--target', 'informed', '--wallets'] + (['--quick'] if QUICK else [])
import pm_final as pf                                               # noqa: E402
import usecases_common as uc                                        # noqa: E402
from original_qcd import Model, scores                              # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = pf.OUT
RATES = (0.02, 0.10)
HS = (300, 1800)
ALARMS = [('metaorder', False, 'hit'), ('churn', False, 'both'), ('informed', False, 'hit'), ('informed', True, 'hit'),
          ('jump', False, 'both')]


def load_models(target, wallets):
    tag = target + ('_wallets' if wallets else '')
    d = torch.load(os.path.join(OUT, f'{tag}_models.pt'), weights_only=False)
    ms = []
    for st in d['state']:
        m = Model(len(d['features']))
        m.load_state_dict(st)
        m.eval()
        ms.append(m)
    import lightgbm as lgb
    bst = lgb.Booster(model_str=d['gbm'])
    mu, sd = d['mu'], d['sd']
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    return tag, ms, bst, norm, len(d['features'])


def whole(models, bst, norm, X, need):
    """DeepQCD ensemble and gradient boosting of a market at the minutes in need (DeepQCD on the 60-minute blocks that
    hold them, each with the 180 minutes before)."""
    n = len(X)
    deep = np.full(n, -np.inf)
    Xn = norm(X)
    gb = bst.predict(uc.lagged_last(np.stack([Xn[max(0, k - 30):k + 1] if k >= 30 else
                                                np.r_[np.zeros((30 - k, X.shape[1]), np.float32), Xn[:k + 1]] for k in need]), 5, (10, 30)),
                     raw_score=True)
    blocks = np.unique(need[need >= pf.PRE] // pf.MAXLEN)
    segs, where = [], []
    for b in blocks:
        s = b * pf.MAXLEN
        seg = Xn[s - pf.PRE:s + pf.MAXLEN]
        if len(seg) < pf.L:
            seg = np.r_[seg, np.zeros((pf.L - len(seg), X.shape[1]), np.float32)]
        segs.append(seg); where.append(s)
    if segs:
        S = np.stack(segs)
        d = np.mean([scores(m, S, chunk=2000) for m in models], 0)
        for s, r in zip(where, d):
            k = min(pf.MAXLEN, n - s)
            deep[s:s + k] = r[pf.PRE:pf.PRE + k]
    return deep[need], gb


def main():
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get('QCD_THREADS', 4)))
    ms = pf.markets()
    Wt = pf.wallet_table(ms)
    gid = {a: i for i, a in enumerate(Wt['addr'])}
    n_m = len(ms)
    split = np.where(np.arange(n_m) < 0.6 * n_m, 0, np.where(np.arange(n_m) < 0.75 * n_m, 1, 2))
    M = {key: load_models(*key[:2]) for key in ALARMS}
    targets = ['metaorder', 'churn', 'informed', 'jump']
    per = {1: [], 2: []}
    for mi_, m in enumerate(ms):
        if split[mi_] == 0:
            continue
        gm = {mi_: np.array([gid[a] for a in np.load(os.path.join(pf.DATA, 'full', m['cid'] + '.npz'))['wallets']], np.int64)}
        mk = pf.market(m, mi_, Wt, gm, targets=targets, wallets=True)
        if mk is None:
            continue
        t, w, s, usd, p0, t0m = mk['trades']
        mi = t // 60 - t0m
        n = mk['n']
        prev = np.clip(mi - 1, 0, n - 1)
        need, inv = np.unique(prev, return_inverse=True)
        rec = {'t': t, 's': s, 'usd': usd, 'p0': p0, 'first': mi > 0, 'busy': mk['x'][prev, 8].astype(float),
               'flow5': pmm_roll(np.bincount(mi, weights=s * usd, minlength=n), 5)[prev], 'lab_prev': {}, 'side_now': {}}
        for target in targets:
            lab = np.zeros(n, np.float32)
            side = np.zeros(n, np.float32)
            for a, b, sg, *_ in mk['eps_all'][target]:
                lab[max(a, 0):min(b, n)] = 1
                side[max(a, 0):min(b, n)] = sg
            rec['lab_prev'][target] = lab[prev] > 0
            rec['side_now'][target] = side[np.clip(mi, 0, n - 1)]
        for key in ALARMS:
            tag, mods, bst, norm, P = M[key]
            d_, g_ = whole(mods, bst, norm, mk['x'][:, :P], need)
            rec[tag] = (d_[inv].astype(np.float32), g_[inv].astype(np.float32))
        per[split[mi_]].append(rec)
        if len(per[1]) + len(per[2]) in (1, 50, 200, 500, 1000):
            print(f'  {len(per[1]) + len(per[2])} markets scored ({time.time() - t0:.0f} s)', flush=True)
    print(f'calibration {len(per[1])} markets, test {len(per[2])} ({time.time() - t0:.0f} s)', flush=True)

    # thresholds: each alarm on for a share of the decision minutes outside its own episodes (calibration markets)
    H = {}
    for key in ALARMS:
        tag = key[0] + ('_wallets' if key[1] else '')
        for j, det in enumerate(('DeepQCD', 'GB')):
            vals = []
            for rec in per[1]:
                v = rec[tag][j][~rec['lab_prev'][key[0]]]
                vals.append(v[np.isfinite(v)])
            vals = np.concatenate(vals)
            for r in RATES:
                H[(tag, det, r)] = float(np.quantile(vals, 1 - r))
    vals = np.concatenate([rec['busy'] for rec in per[1]])
    for r in RATES:
        H[('busy', '', r)] = float(np.quantile(vals, 1 - r))

    res = {'markets': {'calibration': len(per[1]), 'test': len(per[2])}, 'rates': list(RATES), 'horizons_s': list(HS), 'results': {}}
    for r in RATES:
        for hz in HS:
            rows = {}                                      # strategy -> per-market (pnl, usd)
            for rec in per[2]:
                t, s, usd, p0, first = rec['t'], rec['s'], rec['usd'], rec['p0'], rec['first']
                fut = np.searchsorted(t, t + hz, side='right') - 1
                ok = t + hz <= t[-1]
                mark = -s * (p0[fut] - p0) * 100
                hit = s == np.sign(rec['flow5'])
                q = {'always quote': np.ones(len(t), bool), 'step back when busy': ~((rec['busy'] > H[('busy', '', r)]) & first)}
                union = np.zeros(len(t), bool)
                for key in ALARMS:
                    tag = key[0] + ('_wallets' if key[1] else '')
                    for j, det in enumerate(('DeepQCD', 'GB')):
                        on = (rec[tag][j] > H[(tag, det, r)]) & first
                        pull = on & hit if key[2] == 'hit' else on
                        q[f'{tag} alarm ({det})'] = ~pull
                        if det == 'DeepQCD':
                            union |= pull
                q['any DeepQCD alarm'] = ~union
                for target in ('metaorder', 'informed'):
                    side = rec['side_now'][target]
                    q[f'oracle: {target}'] = ~((side != 0) & (s == side))
                for k, v in q.items():
                    sel = v & ok
                    rows.setdefault(k, []).append((float(np.sum(mark[sel] * usd[sel])), float(np.sum(usd[sel])), float(np.sum(usd[ok]))))
            out = {}
            base = np.array(rows['always quote'])
            rng = np.random.default_rng(5)
            bi = [rng.integers(0, len(base), len(base)) for _ in range(pf.B)]
            for k, v in rows.items():
                v = np.array(v)
                pnl = 100 * v[:, 0].sum() / v[:, 2].sum()
                bs = [100 * v[i, 0].sum() / v[i, 2].sum() for i in bi]
                dif = [100 * (v[i, 0].sum() - base[i, 0].sum()) / v[i, 2].sum() for i in bi]
                out[k] = {'pnl_per_100usd': float(pnl), 'ci': np.quantile(bs, [0.025, 0.975]).tolist(),
                          'vs_always_ci': np.quantile(dif, [0.025, 0.975]).tolist(), 'share_filled': float(v[:, 1].sum() / v[:, 2].sum()),
                          'cents_per_share_on_fills': float(v[:, 0].sum() / max(v[:, 1].sum(), 1))}
            res['results'][f'{r}_{hz}'] = out
            print(f'alarm share {r}, markout {hz // 60} min:', flush=True)
            for k, v in out.items():
                print(f'  {k:34s} P&L {v["pnl_per_100usd"]:+7.2f} c per $100 [{v["ci"][0]:+.2f}, {v["ci"][1]:+.2f}], filled '
                      f'{v["share_filled"]:.0%}', flush=True)
    res['thresholds'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in H.items()}
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'replay.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


def pmm_roll(a, k):
    c = np.cumsum(a)
    out = c.copy()
    out[k:] -= c[:-k]
    return out


if __name__ == '__main__':
    main()

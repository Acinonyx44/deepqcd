"""
Use case 4, hedge funds: early warning of violent rallies (squeezes) in heavily shorted US stocks.

A short seller loses most in a squeeze: the price runs up, shorts are forced to buy back, and the buying pushes it
higher. If the start is detected early, the position can be cut while most of the run is still ahead. Data (all
public, squeeze_fetch.py): Yahoo daily OHLCV and FINRA Reg SHO daily short-sale volume, 2021-2026, for the stocks with
a median short-volume share of at least 40% and median volume of at least 300,000 shares.

Label (hindsight, plentiful across many stocks): a rally = the close rises at least 50% from a local low within at
most 10 trading days, with volume over those days at least 3 times the stock's 60-day median. tau1 = the first day
after the low, tau2 = the day after the peak.
Daily features (P = 10): log return, log volume against the 60-day median, short-volume share and its change against
the 20-day mean, intraday range, overnight gap, 5-day and 20-day returns, 20-day volatility, distance below the
52-week high.
Test (native): rallies that start in 2025 or later after 60 days without one: those 60 days and the rally; plus 8,000
random 60-day windows without a rally. False alarms are counted on the same 60 days (after 5 days of warm-up).
Training: before 2025. Detectors: DeepQCD (three seeds), WL-CUSUM over 3 days, and rules: 2-day return, volume spike,
short-share jump, and all combined.
Value: the part of the rally still to come at the alarm (% of the price at the alarm), which a short seller that
covers at the alarm's close avoids.

Run:  .venv/bin/python usecase_squeeze.py      -> runs/usecases/squeeze.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

import usecases_common as uc
from original_qcd import normalization

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, 'data', 'squeeze')
OUT = os.path.join(ROOT, 'runs', 'usecases')
RISE, MAXDAYS, VOLX = 0.5, 10, 3.0
PRE, WARM, WL_K = 60, 5, 3
L = PRE + MAXDAYS + 1
SPLIT = np.datetime64('2025-01-01')
FEATURES = ['ret', 'vol_ratio', 'short_share', 'short_jump', 'range', 'gap', 'ret5', 'ret20', 'vol20', 'below_high']


def short_share(syms):
    """Daily FINRA short-volume share per symbol: DataFrame indexed by date, one column per symbol."""
    keep = set(syms)
    parts = []
    for f in sorted(os.listdir(os.path.join(D, 'finra'))):
        t = pd.read_csv(os.path.join(D, 'finra', f), sep='|', usecols=[0, 1, 2, 4], dtype={'Symbol': str}).dropna()
        t = t[t.Symbol.isin(keep) & (t.TotalVolume > 0)]
        parts.append(pd.DataFrame({'date': pd.to_datetime(t.Date.astype(int).astype(str)), 'sym': t.Symbol,
                                   'share': t.ShortVolume / t.TotalVolume}))
    s = pd.concat(parts)
    return s.pivot_table(index='date', columns='sym', values='share')


def stock(sym, ss):
    j = json.load(open(os.path.join(D, 'yahoo', f'{sym}.json')))['chart']['result']
    if not j or 'timestamp' not in j[0]:
        return None
    q = j[0]['indicators']['quote'][0]
    df = pd.DataFrame({k: q.get(k) for k in ('open', 'high', 'low', 'close', 'volume')},
                      index=pd.to_datetime(j[0]['timestamp'], unit='s').normalize()).dropna()
    df = df[(df.close > 0) & (df.volume > 0)]
    df = df[~df.index.duplicated()]
    if len(df) < 300:
        return None
    sh = ss[sym].reindex(df.index).ffill().fillna(0.45).values if sym in ss else np.full(len(df), 0.45)
    c, v = df.close.values, df.volume.values
    lr = np.r_[0, np.diff(np.log(c))]
    medv = pd.Series(v).rolling(60, min_periods=20).median().bfill().values
    vr = np.log(v / np.maximum(medv, 1))
    sh20 = pd.Series(sh).rolling(20, min_periods=5).mean().bfill().values
    ret5 = np.log(c / pd.Series(c).shift(5).bfill().values)
    ret20 = np.log(c / pd.Series(c).shift(20).bfill().values)
    vol20 = pd.Series(lr).rolling(20, min_periods=5).std().bfill().values
    hi52 = pd.Series(c).rolling(252, min_periods=20).max().bfill().values
    gap = np.log(df.open.values / pd.Series(c).shift(1).bfill().values)
    rng_ = np.log(df.high.values / df.low.values)
    x = np.stack([np.clip(lr, -1, 1), np.clip(vr, -3, 5), sh, np.clip(sh - sh20, -0.5, 0.5), np.clip(rng_, 0, 1), np.clip(gap, -1, 1),
                  np.clip(ret5, -2, 2), np.clip(ret20, -2, 2), np.clip(vol20, 0, 0.5), np.clip(np.log(c / hi52), -4, 0)], 1).astype(np.float32)
    # rallies: from a low, +50% within 10 days with 3x volume; greedy, non-overlapping
    lab = np.zeros(len(c), np.float32)
    ev = []
    i = 1
    while i < len(c) - 1:
        j_end = min(len(c), i + MAXDAYS + 1)
        k = i + int(np.argmax(c[i:j_end]))
        if c[k] >= (1 + RISE) * c[i - 1] and v[i:k + 1].sum() >= VOLX * medv[i - 1] * (k - i + 1) and c[i - 1] <= c[i - 1:i + 2].min():
            lab[i:k + 1] = 1
            ev.append((i, k + 1))
            i = k + 1
        else:
            i += 1
    return {'x': x, 'lab': lab, 'ev': ev, 'dates': df.index.values, 'close': c}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    syms = [s for s in json.load(open(os.path.join(D, 'universe.json'))) if os.path.exists(os.path.join(D, 'yahoo', f'{s}.json'))]
    ss = short_share(syms)
    S = {}
    for s in syms:
        try:
            r = stock(s, ss)
        except (KeyError, ValueError, TypeError):
            r = None
        if r is not None:
            S[s] = r
    n_ev = sum(len(r['ev']) for r in S.values())
    print(f'{len(S)} stocks, {n_ev} rallies ({time.time() - t0:.0f} s)', flush=True)

    xs, ys = [], []
    for r in S.values():
        x, lab, dates = r['x'], r['lab'], r['dates']
        for a, b in r['ev']:
            if dates[a] >= SPLIT:
                continue
            for _ in range(3):
                st = a - int(rng.integers(10, PRE + 1))
                if st >= 0 and st + L <= len(x) and dates[st + L - 1] < SPLIT:
                    xs.append(x[st:st + L]); ys.append(lab[st:st + L])
        ok = np.flatnonzero(dates[L:] < SPLIT)
        for st in rng.choice(ok, min(len(ok), 12), replace=False) if len(ok) else []:
            xs.append(x[st:st + L]); ys.append(lab[st:st + L])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)

    X, tau1, tau2, info = [], [], [], []
    for sym, r in S.items():
        x, lab, dates, c = r['x'], r['lab'], r['dates'], r['close']
        for a, b in r['ev']:
            if dates[a] >= SPLIT and a >= PRE and lab[a - PRE:a].max() == 0:
                seg = np.zeros((L, x.shape[1]), np.float32)
                seg[:b - a + PRE] = x[a - PRE:b]
                X.append(seg); tau1.append(PRE); tau2.append(b - a + PRE)
                info.append((sym, a, b))
    n_test = len(X)
    pool = [(s, st) for s, r in S.items() for st in np.flatnonzero(r['dates'][:-PRE] >= SPLIT)[::7]]
    for k in rng.permutation(len(pool)):
        s, st = pool[k]
        lab = S[s]['lab']
        if lab[max(0, st - 20):st + PRE].max() == 0:
            seg = np.zeros((L, xs.shape[2]), np.float32)
            seg[:PRE] = S[s]['x'][st:st + PRE]
            X.append(seg); tau1.append(PRE); tau2.append(PRE)
            if len(X) - n_test == 8000:
                break
    X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
    valid = np.arange(L)[None, :] < np.where(np.arange(len(X)) < n_test, tau2, PRE)[:, None]
    Xn = norm(X)
    print(f'training windows {len(xs)} ({(ys.max(1) > 0).mean():.0%} with a rally); test {n_test} rallies, {len(X) - n_test} '
          f'quiet windows', flush=True)

    F = {f: i for i, f in enumerate(FEATURES)}
    rule_fns = {'Rule: 2-day return': lambda A: A[:, :, F['ret']] + np.c_[np.zeros((len(A), 1)), A[:, :-1, F['ret']]],
                'Rule: volume spike': lambda A: A[:, :, F['vol_ratio']],
                'Rule: short-share jump': lambda A: A[:, :, F['short_jump']]}
    nrm = xs[[y.max() == 0 for y in ys]]
    comb = np.max([fn(X) / max(np.quantile(fn(nrm)[:, WARM:], 0.99), 1e-3) for fn in rule_fns.values()], 0)
    det = {'Logistic (this step only)': uc.logistic_step(xs_n, ys, Xn), 'WL-CUSUM': uc.wl_cusum(xs_n, ys, Xn, WL_K), **{n: fn(X).astype(float) for n, fn in rule_fns.items()},
           'Rules combined': comb.astype(float)}
    deep, _ = uc.deepqcd(xs_n, ys, Xn, seeds=(0, 1, 2), epochs=20, rng=rng)
    res = {'n_stocks': len(S), 'n_rallies': n_ev, 'n_train_windows': int(len(xs)), 'n_test_events': n_test,
           'n_test_no': int(len(X) - n_test), 'features': FEATURES, 'detectors': {}}

    def value(dd, h):
        rem, delay = [], []
        for i, (sym, a, b) in enumerate(info):
            fa = uc.first_alarm(np.where(valid[i], dd[i], -np.inf), h, WARM, tau2[i])
            if fa < tau1[i]:
                continue
            c = S[sym]['close']
            rem.append(100 * (c[b - 1] / c[a - PRE + fa] - 1))
            delay.append(fa - tau1[i])
        tot = [100 * (S[s]['close'][b - 1] / S[s]['close'][a - 1] - 1) for s, a, b in info]
        return {'n': len(rem), 'remaining_pct': float(np.mean(rem)) if rem else None, 'rally_pct': float(np.mean(tot)),
                'avoided_pct_per_rally': float(np.sum(rem) / n_test), 'delay_days': float(np.median(delay)) if delay else None}
    outs = []
    for k, dd in enumerate(deep):
        o, _ = uc.evaluate(dd, tau1, tau2, n_test, WARM, valid)
        o['value'] = {a: value(dd, o['h_at'][a]) for a in o['h_at']}
        res['detectors'][f'DeepQCD s{k}'] = o
        outs.append(o)
    res['detectors']['DeepQCD'] = uc.merge_seeds(outs)
    res['detectors']['DeepQCD']['value_seeds'] = [o['value'] for o in outs]
    uc.report('DeepQCD', res['detectors']['DeepQCD'])
    for name, dd in det.items():
        o, _ = uc.evaluate(dd, tau1, tau2, n_test, WARM, valid)
        o['value'] = {a: value(dd, o['h_at'][a]) for a in o['h_at']}
        res['detectors'][name] = o
        uc.report(name, o)
    for name in ['DeepQCD s0', 'DeepQCD s1', 'DeepQCD s2', 'WL-CUSUM', 'Rule: 2-day return', 'Rule: volume spike', 'Rules combined']:
        v = res['detectors'][name]['value']['0.05']
        print(f'  value {name:22s} delay {v["delay_days"]} d, remaining {v["remaining_pct"]} of {v["rally_pct"]:.0f}%, '
              f'avoided per rally {v["avoided_pct_per_rally"]:.1f}%', flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'squeeze.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

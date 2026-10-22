"""
Use case 1, market makers: a toxic-flow alarm on crypto perpetual futures (Binance ETHUSDT, public aggregated trades).

A market maker quoting both sides loses when the price runs in one direction: every quote it leaves on the run's side
is filled just before the price moves through it (adverse selection). If it sees a run starting it can pull or skew
those quotes. Runs are labelled in hindsight and are plentiful: every day has dozens.

Labels: per-second last price; a zigzag with reversal 5 bps finds the swings; a fast run is a swing of at least 15 bps
completed within 60 seconds. tau1 = the first second of the swing, tau2 = the second it ends.
Anonymous per-second features (P = 10): trades, signed notional, gross notional, largest trade, price levels traded,
1-second return, imbalance over 5 s, trades over 10 s, realized volatility over 30 s, return over 5 s.
Test (native): every fast run after 300 seconds without one, 300 s before tau1 plus the run; 6,000 random 360-second
windows without a run. PFA = P(alarm before tau1), PD = P(first alarm during the run). Warm-up 30 s.
Split by time: 2026-07-01..09-30 train (92 days), 2026-10-01..10-20 test (20 days).
Detectors: DeepQCD (the original network, three seeds), WL-CUSUM (window_dec_stat over 10 s), and three rules: price
move over 5 s, signed notional over 5 s, trades over 10 s.
Value: the part of the run still to come at the alarm (bps), which a market maker that pulls its quotes on that side
avoids; averaged over all runs (misses count 0) it is the adverse selection avoided per run.

Run:  .venv/bin/python usecase_toxic.py [--quick]      -> runs/usecases/toxic.json
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import usecases_common as uc
from original_qcd import normalization

QUICK = '--quick' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'usecases')
THETA, X_BPS, MAXDUR = 5.0, 15.0, 60
PRE, L_NO, WARM, WL_K = 300, 360, 30, 10
L = PRE + MAXDUR
FEATURES = ['trades', 'flow', 'gross', 'maxsize', 'levels', 'ret1', 'imb5', 'trades10', 'vol30', 'ret5']


def roll(a, k):
    c = np.cumsum(a, 0)
    out = c.copy()
    out[k:] -= c[:-k]
    return out


def day(date):
    t = pd.read_csv(os.path.join(ROOT, 'data', 'binance', f'ETHUSDT-aggTrades-{date}.zip'),
                    usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'])
    t0 = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp())
    sec = (t.transact_time.values // 1000 - t0).astype(int)
    keep = (sec >= 0) & (sec < 86400)
    sec, p, q = sec[keep], t.price.values[keep], t.quantity.values[keep]
    side = np.where(t.is_buyer_maker.values[keep], -1.0, 1.0)            # taker buy = +1
    notional = p * q
    n = 86400
    nt = np.bincount(sec, minlength=n).astype(float)
    flow = np.bincount(sec, weights=side * notional, minlength=n)
    gross = np.bincount(sec, weights=notional, minlength=n)
    mx = np.zeros(n)
    np.maximum.at(mx, sec, notional)
    lv = pd.DataFrame({'s': sec, 'p': p}).drop_duplicates().s.value_counts().reindex(range(n), fill_value=0).values
    last = np.full(n, np.nan)
    last[sec] = p
    if np.isnan(last[0]):
        last[0] = p[0]
    last = last[np.maximum.accumulate(np.where(np.isnan(last), 0, np.arange(n)))]
    lp = np.log(last) * 1e4
    r1 = np.r_[0, np.diff(lp)]
    vol30 = np.sqrt(roll(r1 ** 2, 30))
    x = np.stack([np.log1p(nt), np.sign(flow) * np.log1p(np.abs(flow) / 1e4), np.log1p(gross / 1e4), np.log1p(mx / 1e4),
                  np.log1p(lv), np.clip(r1, -20, 20), roll(flow, 5) / np.maximum(roll(gross, 5), 1), np.log1p(roll(nt, 10)),
                  np.minimum(vol30, 50), np.clip(roll(r1, 5), -50, 50)], 1).astype(np.float32)
    return x, lp


def runs(lp):
    """Zigzag swings (reversal THETA bps); fast runs = swings of >= X_BPS within MAXDUR seconds: [(tau1, tau2, sign)]."""
    piv, ext, ext_i, direction = [0], lp[0], 0, 1                     # start as an up-leg from second 0
    for i in range(1, len(lp)):
        v = lp[i]
        if direction >= 0 and v > ext:
            ext, ext_i = v, i
        elif direction <= 0 and v < ext:
            ext, ext_i = v, i
        if direction >= 0 and v <= ext - THETA and ext_i != piv[-1]:
            piv.append(ext_i); direction, ext, ext_i = -1, v, i
        elif direction <= 0 and v >= ext + THETA and ext_i != piv[-1]:
            piv.append(ext_i); direction, ext, ext_i = 1, v, i
    out = []
    for a, b in zip(piv[:-1], piv[1:]):
        amp = lp[b] - lp[a]
        if abs(amp) >= X_BPS and b - a <= MAXDUR:
            out.append((a + 1, b + 1, int(np.sign(amp))))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    dates = [str(dt.date(2026, 7, 1) + dt.timedelta(k)) for k in range(112)]          # 2026-07-01 .. 2026-10-20
    if QUICK:
        dates = dates[:6] + dates[92:95]
    train_days = [d for d in dates if d < '2026-10-01']
    D = {}
    for d in dates:
        x, lp = day(d)
        rr = runs(lp)
        lab = np.zeros(len(x), np.float32)
        for a, b, _ in rr:
            lab[a:b] = 1
        D[d] = (x, lp, rr, lab)
    n_runs = {d: len(D[d][2]) for d in dates}
    print(f'{len(dates)} days read ({time.time() - t0:.0f} s); fast runs per day: median {np.median(list(n_runs.values())):.0f}',
          flush=True)

    xs, ys = [], []
    for d in train_days:
        x, lp, rr, lab = D[d]
        for a, b, _ in rr:
            s = a - int(rng.integers(60, PRE + 30))
            if s >= 0 and s + L <= len(x):
                xs.append(x[s:s + L]); ys.append(lab[s:s + L])
        for s in rng.integers(0, len(x) - L, 100 if not QUICK else 50):
            xs.append(x[s:s + L]); ys.append(lab[s:s + L])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)

    ev, X, tau1, tau2 = [], [], [], []
    for d in dates:
        if d in train_days:
            continue
        x, lp, rr, lab = D[d]
        for a, b, sg in rr:
            if a >= PRE and lab[a - PRE:a].max() == 0:
                seg = np.zeros((L, x.shape[1]), np.float32)
                seg[:b - a + PRE] = x[a - PRE:b]
                X.append(seg); tau1.append(PRE); tau2.append(b - a + PRE)
                ev.append((d, a, b, sg))
    n_ev = len(ev)
    test_days = [d for d in dates if d not in train_days]
    for k in range(1000 if QUICK else 6000):
        d = test_days[k % len(test_days)]
        x, lp, rr, lab = D[d]
        for _ in range(20):
            s = int(rng.integers(0, len(x) - L_NO))
            if lab[max(0, s - 60):s + L_NO].max() == 0:
                seg = np.zeros((L, x.shape[1]), np.float32)
                seg[:L_NO] = x[s:s + L_NO]
                X.append(seg); tau1.append(L + 1); tau2.append(L + 1)
                break
    X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
    tau1 = np.where(tau1 > L, L_NO, tau1)                   # no-change streams: false alarms anywhere in their 660 s
    tau2 = np.where(np.arange(len(X)) >= n_ev, L_NO, tau2)
    valid = np.arange(L)[None, :] < np.where(np.arange(len(X)) < n_ev, tau2, L_NO)[:, None]
    Xn = norm(X)
    print(f'training windows {len(xs)}; test: {n_ev} fast runs after {PRE} quiet seconds, {len(X) - n_ev} quiet windows', flush=True)

    det = {'Logistic (this step only)': uc.logistic_step(xs_n, ys, Xn), 'WL-CUSUM': uc.wl_cusum(xs_n, ys, Xn, WL_K),
           'Rule: price move, 5 s': np.abs(X[:, :, 9]).astype(float),
           'Rule: signed flow, 5 s': np.abs(roll(np.sign(X[:, :, 1]) * np.expm1(np.abs(X[:, :, 1])), 5)),
           'Rule: trades, 10 s': X[:, :, 7].astype(float)}
    deep, _ = uc.deepqcd(xs_n, ys, Xn, seeds=(0,) if QUICK else (0, 1, 2), epochs=2 if QUICK else 20, rng=rng)
    res = {'settings': {'theta_bps': THETA, 'x_bps': X_BPS, 'maxdur': MAXDUR, 'pre': PRE, 'warm': WARM, 'wl_k': WL_K,
                        'features': FEATURES, 'train_days': len(train_days), 'test_days': len(test_days)},
           'runs_per_day': n_runs, 'n_train_windows': int(len(xs)), 'n_test_events': n_ev, 'n_test_no': int(len(X) - n_ev),
           'detectors': {}}
    sizes = np.array([abs(D[d][1][b - 1] - D[d][1][a - 1]) for d, a, b, sg in ev])
    groups = {'15-20 bps': sizes < 20, '20-30 bps': (sizes >= 20) & (sizes < 30), '30+ bps': sizes >= 30}

    def value(dstat, out):
        val = {}
        for a in map(str, uc.ALPHAS):
            h = out['h_at'][a]
            rem, delay = [], []
            for i, (d, s, b, sg) in enumerate(ev):
                fa = uc.first_alarm(dstat[i], h, WARM, tau2[i])
                if fa < tau1[i]:
                    continue
                lp = D[d][1]
                rem.append(sg * (lp[b - 1] - lp[s - PRE + fa]))
                delay.append(fa - tau1[i])
            val[a] = {'n': len(rem), 'remaining_bps': float(np.mean(rem)) if rem else None,
                      'avoided_bps_per_run': float(np.sum(rem) / n_ev), 'delay_s': float(np.median(delay)) if delay else None,
                      'run_bps': float(np.mean(sizes))}
        return val
    runs_out = []
    for k, dd in enumerate(deep):
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, WARM, valid, groups)
        o['value'] = value(np.where(valid, dd, -np.inf), o)
        runs_out.append(o)
        res['detectors'][f'DeepQCD s{k}'] = o
    res['detectors']['DeepQCD'] = uc.merge_seeds(runs_out)
    res['detectors']['DeepQCD']['value_seeds'] = [r['value'] for r in runs_out]
    uc.report('DeepQCD', res['detectors']['DeepQCD'])
    for name, dd in det.items():
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, WARM, valid, groups)
        o['value'] = value(np.where(valid, dd, -np.inf), o)
        res['detectors'][name] = o
        uc.report(name, o)
    for name in ['DeepQCD s0', 'WL-CUSUM', 'Rule: price move, 5 s', 'Rule: signed flow, 5 s', 'Rule: trades, 10 s']:
        v = res['detectors'][name]['value']['0.05']
        print(f'  value {name:26s} delay {v["delay_s"]} s, remaining {v["remaining_bps"]} of {v["run_bps"]:.1f} bps, '
              f'avoided per run {v["avoided_bps_per_run"]:.2f} bps', flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'toxic.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

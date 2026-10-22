"""
Use case 6, banks: card fraud / account takeover on IBM TabFormer (Padhi et al., ICASSP 2021): 24.4 million synthetic
credit-card transactions of 2,000 people (6,139 cards), 1991-2020, with 29,757 fraudulent transactions. A stolen card
is used in a burst and then blocked or abandoned: a transient change in the card's transaction stream.

Each card is a stream with one step per transaction (how issuers score). Features (P = 15): log amount, refund,
online, chip, new merchant, new city, new state or country, foreign, new merchant category, any error, log hours since
the previous transaction, hour of day (sin, cos), amount against the card's history (z-score so far), transactions in
the last 24 hours.
Change: a burst = fraudulent transactions of one card at most 20 transactions apart; from its first to one after its
last (at most 30 + 1).
Test (native): bursts that start in 2016 or later after 30 fraud-free transactions: those 30 and the burst; plus
8,000 windows of 30 fraud-free transactions (from 2016) of any card. False alarms are counted on the same 30
transactions for both (after a warm-up of 5). Training: bursts and windows before 2016.
Detectors: DeepQCD (three seeds), WL-CUSUM (window_dec_stat over 5 transactions), and issuer-style rules: online and
above the usual amount, new state or country, new merchant and above the usual amount, transaction errors, velocity
(transactions in 24 hours), above the usual amount, and all combined.

Run:  .venv/bin/python usecase_fraud.py [--quick]      -> runs/usecases/fraud.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import usecases_common as uc
from original_qcd import normalization

QUICK = '--quick' in sys.argv
GBM = '--gbm' in sys.argv                    # only add the gradient-boosting baseline to the saved results
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'usecases')
PRE, MAXCH, SLACK, WARM, WL_K, GAPTX = 30, 30, 1, 5, 5, 20
L = PRE + MAXCH + SLACK
SPLIT_YEAR = 2016
FEATURES = ['log_amount', 'refund', 'online', 'chip', 'new_merchant', 'new_city', 'new_state', 'foreign', 'new_mcc', 'error',
            'log_gap_h', 'hour_sin', 'hour_cos', 'amount_z', 'tx_24h']
US = set('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA '
         'RI SC SD TN TX UT VT VA WA WV WI WY DC AA AE AP'.split())


def load():
    t = pd.read_csv(os.path.join(ROOT, 'data', 'tabformer', 'card_transaction.v1.csv'),
                    dtype={'User': 'int32', 'Card': 'int8', 'Year': 'int16', 'Month': 'int8', 'Day': 'int8', 'Use Chip': 'category',
                           'Merchant Name': 'int64', 'Merchant City': 'category', 'Merchant State': 'category', 'MCC': 'int16',
                           'Errors?': 'category', 'Is Fraud?': 'category'}, usecols=lambda c: c != 'Zip')
    if QUICK:
        t = t[t.User < 300]
    hm = t.Time.str.split(':', expand=True).astype(int)
    ts = pd.to_datetime(dict(year=t.Year, month=t.Month, day=t.Day, hour=hm[0], minute=hm[1]))
    card = t.User.astype(np.int64) * 10 + t.Card
    d = pd.DataFrame({'card': card.values, 'ts': ts.values, 'year': t.Year.values,
                      'amount': t.Amount.str.replace('$', '', regex=False).astype(float).values,
                      'chip': t['Use Chip'].astype(str).values, 'merchant': t['Merchant Name'].values,
                      'city': t['Merchant City'].astype(str).values, 'state': t['Merchant State'].astype(str).values,
                      'mcc': t.MCC.values, 'error': t['Errors?'].notna().values, 'fraud': (t['Is Fraud?'] == 'Yes').values,
                      'hour': hm[0].values})
    return d.sort_values(['card', 'ts'], kind='stable').reset_index(drop=True)


def features(d):
    first = lambda cols: ~d.duplicated(['card'] + cols).values
    g = d.groupby('card')
    gap = g.ts.diff().dt.total_seconds().fillna(7 * 86400).clip(0, 30 * 86400).values / 3600
    la = np.log1p(np.abs(d.amount.values))
    n = g.cumcount().values + 1
    s1 = g['amount'].transform(lambda a: np.log1p(np.abs(a)).cumsum()).values - la
    s2 = g['amount'].transform(lambda a: (np.log1p(np.abs(a)) ** 2).cumsum()).values - la ** 2
    m = np.where(n > 1, s1 / np.maximum(n - 1, 1), la)
    sd = np.sqrt(np.maximum(np.where(n > 2, s2 / np.maximum(n - 1, 1) - m ** 2, 1.0), 0.04))
    key = d.card.values.astype(np.int64) * 10 ** 10 + d.ts.values.astype('datetime64[s]').astype(np.int64)
    tx24 = np.arange(len(d)) - np.searchsorted(key, key - 86400, side='left')
    online = d.chip.values == 'Online Transaction'
    st = d.state.values
    foreign = (~online) & (st != 'nan') & ~np.isin(st, list(US))
    x = np.stack([la, d.amount.values < 0, online, d.chip.values == 'Chip Transaction', first(['merchant']), first(['city']),
                  first(['state']), foreign, first(['mcc']), d.error.values, np.log1p(gap), np.sin(2 * np.pi * d.hour.values / 24),
                  np.cos(2 * np.pi * d.hour.values / 24), np.clip((la - m) / sd, -5, 5), np.log1p(tx24)], 1)
    return x.astype(np.float32)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    d = load()
    X_all = features(d)
    card, fraud, year = d.card.values, d.fraud.values, d.year.values
    starts = np.r_[0, np.flatnonzero(card[1:] != card[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]
    lab = np.zeros(len(d), np.float32)
    near = np.zeros(len(d), bool)                      # within GAPTX transactions of a fraud
    bursts = []
    for s, e in zip(starts, ends):
        k = np.flatnonzero(fraud[s:e]) + s
        if not len(k):
            continue
        for a, b in zip(np.r_[k[0], k[1:][np.diff(k) > GAPTX]], np.r_[k[:-1][np.diff(k) > GAPTX], k[-1]]):
            b2 = min(b + 1 + SLACK, a + MAXCH + SLACK, e)
            lab[a:b2] = 1
            near[max(s, a - GAPTX):min(e, b + GAPTX)] = True
            bursts.append((s, e, a, b2))
    print(f'{len(d):,} transactions, {len(starts):,} cards, {len(bursts):,} fraud bursts ({time.time() - t0:.0f} s)', flush=True)

    xs, ys = [], []
    tr_b = [b for b in bursts if year[b[2]] < SPLIT_YEAR]
    te_b = [b for b in bursts if year[b[2]] >= SPLIT_YEAR and b[2] - b[0] >= PRE and lab[b[2] - PRE:b[2]].max() == 0]
    for s, e, a, b in tr_b:
        for _ in range(3):
            st = a - int(rng.integers(5, PRE + 1))
            if st >= s and st + L <= e:
                xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
    cand = np.flatnonzero((year < SPLIT_YEAR) & ~near)
    for st in rng.choice(cand, 6000 if QUICK else 25000, replace=False):
        if st + L <= len(d) and card[st] == card[st + L - 1]:
            xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)

    X, tau1, tau2, kind = [], [], [], []
    for s, e, a, b in te_b:
        seg = np.zeros((L, X_all.shape[1]), np.float32)
        seg[:b - a + PRE] = X_all[a - PRE:b]
        X.append(seg); tau1.append(PRE); tau2.append(b - a + PRE)
        kind.append(bool(X_all[a, 2]))
    n_ev = len(X)
    cand = np.flatnonzero((year >= SPLIT_YEAR) & ~near)
    k = 0
    for st in rng.choice(cand, 40000, replace=False):
        if st + PRE <= len(d) and card[st] == card[st + PRE - 1] and not near[st:st + PRE].any():
            seg = np.zeros((L, X_all.shape[1]), np.float32)
            seg[:PRE] = X_all[st:st + PRE]
            X.append(seg); tau1.append(PRE); tau2.append(PRE)
            k += 1
            if k == (2000 if QUICK else 8000):
                break
    X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
    valid = np.arange(L)[None, :] < np.where(np.arange(len(X)) < n_ev, tau2, PRE)[:, None]
    Xn = norm(X)
    print(f'training windows {len(xs)} ({(ys.max(1) > 0).mean():.0%} with fraud); test {n_ev} bursts, {len(X) - n_ev} clean windows',
          flush=True)

    F = {f: i for i, f in enumerate(FEATURES)}
    rule_fns = {'Rule: online, above usual amount': lambda A: A[:, :, F['online']] * A[:, :, F['amount_z']],
                'Rule: new state or country': lambda A: np.maximum(A[:, :, F['new_state']], A[:, :, F['foreign']]),
                'Rule: new merchant, above usual amount': lambda A: A[:, :, F['new_merchant']] * A[:, :, F['amount_z']],
                'Rule: transaction errors': lambda A: A[:, :, F['error']],
                'Rule: velocity (24 h)': lambda A: A[:, :, F['tx_24h']],
                'Rule: above usual amount': lambda A: A[:, :, F['amount_z']]}
    nrm = xs[[y.max() == 0 for y in ys]]
    comb = np.max([fn(X) / max(np.quantile(fn(nrm)[:, WARM:], 0.99), 1e-3) for fn in rule_fns.values()], 0)
    kind = np.array(kind)
    groups = {'starts online': kind, 'starts in person': ~kind}
    if GBM:
        dd = uc.gbm_lags(xs_n, ys, Xn, wins=(5, 15))
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, WARM, valid, groups)
        uc.add_detector(os.path.join(OUT, 'fraud.json'), 'Gradient boosting (lags)', o)
        return
    det = {'Logistic (this step only)': uc.logistic_step(xs_n, ys, Xn), 'WL-CUSUM': uc.wl_cusum(xs_n, ys, Xn, WL_K), **{n: fn(X).astype(float) for n, fn in rule_fns.items()},
           'Rules combined': comb.astype(float)}
    deep, _ = uc.deepqcd(xs_n, ys, Xn, seeds=(0,) if QUICK else (0, 1, 2), epochs=2 if QUICK else 20, rng=rng)
    res = {'settings': {'pre': PRE, 'maxch': MAXCH, 'slack': SLACK, 'warm': WARM, 'wl_k': WL_K, 'gap_tx': GAPTX,
                        'split_year': SPLIT_YEAR, 'features': FEATURES},
           'n_transactions': int(len(d)), 'n_cards': int(len(starts)), 'n_bursts': len(bursts), 'n_train_bursts': len(tr_b),
           'n_train_windows': int(len(xs)), 'n_test_events': n_ev, 'n_test_no': int(len(X) - n_ev),
           'group_sizes': {g: int(m.sum()) for g, m in groups.items()}, 'detectors': {}}
    outs = [uc.evaluate(dd, tau1, tau2, n_ev, WARM, valid, groups)[0] for dd in deep]
    for k, o in enumerate(outs):
        res['detectors'][f'DeepQCD s{k}'] = o
    res['detectors']['DeepQCD'] = uc.merge_seeds(outs)
    uc.report('DeepQCD', res['detectors']['DeepQCD'])
    for name, dd in det.items():
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, WARM, valid, groups)
        res['detectors'][name] = o
        uc.report(name, o)
    # how many fraudulent transactions get through before the first alarm (at PFA 0.05)
    lost = {}
    for name in ['DeepQCD s0', 'WL-CUSUM', 'Rules combined']:
        dd = deep[0] if name == 'DeepQCD s0' else det[name]
        h = res['detectors'][name]['h_at']['0.05']
        n_before = []
        for i in range(n_ev):
            fa = uc.first_alarm(np.where(valid[i], dd[i], -np.inf), h, WARM, tau2[i])
            if fa >= tau1[i]:
                n_before.append(fa - tau1[i])
        lost[name] = {'median_tx_before_alarm': float(np.median(n_before)) if n_before else None}
    res['fraud_before_alarm'] = lost
    print(lost, flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'fraud.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

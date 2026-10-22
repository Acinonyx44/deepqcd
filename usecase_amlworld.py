"""
Use case 2, banks: money-laundering alarms on IBM AMLworld (Altman et al., NeurIPS 2023 datasets track; HI-Small:
5.1 million transfers between 515,000 accounts at many banks, 1-18 September 2022, every laundering transfer labelled,
with eight typologies). It is the successor of AMLSim with far richer normal behaviour: many payment formats, 15
currencies, salaries, bills, investments.

Each account is a stream with one step per transaction it takes part in (how banks score: transaction by
transaction). Features (P = 15): direction out / in, log USD amount, payment format (ACH, cheque, credit card, cash,
wire, bitcoin; reinvestment is all zero), cross-currency, cross-bank, self-transfer, new counterparty, log hours since
the account's previous transaction, and the counterparty's number of distinct counterparties so far (hub contact).
Change: from the account's first laundering transaction to one transaction after its last.
Test (native): laundering accounts whose first laundering transaction is after 2022-09-06 12:00 and that have at least
15 earlier transactions: the 15 transactions before and the change (at most 10 + 1); plus 8,000 windows of 15
transactions of accounts never involved, after the same date (false alarms are counted on 15 transactions for both). Training: before 2022-09-06 00:00. Warm-up 3.
Detectors: DeepQCD (three seeds), WL-CUSUM (window_dec_stat over 5 transactions), and rules: ACH to/from a new
counterparty, amount against the account's usual amount, new counterparties in the last 5, hub contact,
cross-currency, and all rules combined.

Run:  .venv/bin/python usecase_amlworld.py [--quick]      -> runs/usecases/amlworld.json
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
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'usecases')
PRE, MAXCH, SLACK, WARM, WL_K = 15, 10, 1, 3, 5
L = PRE + MAXCH + SLACK
SPLIT_TRAIN, SPLIT_TEST = pd.Timestamp('2022-09-06 00:00'), pd.Timestamp('2022-09-06 12:00')
FORMATS = ['ACH', 'Cheque', 'Credit Card', 'Cash', 'Wire', 'Bitcoin']
FEATURES = ['out', 'in', 'log_usd'] + FORMATS + ['cross_cur', 'cross_bank', 'self', 'new_cp', 'log_gap_h', 'cp_degree']


def load():
    t = pd.read_csv(os.path.join(ROOT, 'data', 'amlworld', 'HI-Small_Trans.csv'), dtype=str)
    t.columns = ['ts', 'fb', 'fa', 'tb', 'ta', 'amt_r', 'cur_r', 'amt_p', 'cur_p', 'fmt', 'lau']
    t['ts'] = pd.to_datetime(t.ts, format='%Y/%m/%d %H:%M')
    t['amt_r'], t['amt_p'] = t.amt_r.astype(float), t.amt_p.astype(float)
    # USD rates from cross-currency transfers paid in USD (or received in USD)
    rate = {'US Dollar': 1.0}
    a = t[(t.cur_p == 'US Dollar') & (t.cur_r != 'US Dollar') & (t.amt_p > 0)]
    rate.update((a.amt_r / a.amt_p).groupby(a.cur_r).median().to_dict())
    b = t[(t.cur_r == 'US Dollar') & (t.cur_p != 'US Dollar') & (t.amt_r > 0)]
    for c, v in (b.amt_p / b.amt_r).groupby(b.cur_p).median().items():
        rate.setdefault(c, v)
    t['usd'] = t.amt_p / t.cur_p.map(rate).fillna(1.0)
    t['src'], t['dst'] = t.fb + '_' + t.fa, t.tb + '_' + t.ta
    return t


def rows(t):
    """One row per (account, transaction), sorted by account and time, with the features and the laundering flag."""
    base = {'ts': t.ts.values, 'usd': t.usd.values, 'fmt': t.fmt.values, 'cross_cur': (t.cur_r != t.cur_p).values,
            'cross_bank': (t.fb != t.tb).values, 'self': (t.src == t.dst).values, 'lau': (t.lau == '1').values}
    out = pd.DataFrame(dict(base, acct=t.src.values, cp=t.dst.values, out=True))
    inn = pd.DataFrame(dict(base, acct=t.dst.values, cp=t.src.values, out=False))
    r = pd.concat([out, inn[~inn['self']]], ignore_index=True)
    r = r.sort_values(['acct', 'ts'], kind='stable').reset_index(drop=True)
    r['new_cp'] = ~r.duplicated(['acct', 'cp'])
    r['deg'] = r.groupby('acct').new_cp.cumsum()
    gap = r.groupby('acct').ts.diff().dt.total_seconds() / 3600
    r['gap'] = gap.fillna(24 * 7).clip(0, 24 * 7)
    # counterparty's distinct counterparties so far
    left = r[['ts', 'cp']].reset_index().sort_values('ts')
    right = r[['ts', 'acct', 'deg']].rename(columns={'acct': 'cp', 'deg': 'cp_deg'}).sort_values('ts')
    m = pd.merge_asof(left, right, on='ts', by='cp', direction='backward')
    r.loc[m['index'].values, 'cp_deg'] = m.cp_deg.fillna(0).values
    return r


def features(r):
    f = [r.out.values, ~r.out.values, np.log1p(r.usd.values / 100)]
    f += [(r.fmt.values == k) for k in FORMATS]
    f += [r.cross_cur.values, r.cross_bank.values, r['self'].values, r.new_cp.values, np.log1p(r.gap.values),
          np.log1p(r.cp_deg.values)]
    return np.stack([np.asarray(v, np.float32) for v in f], 1)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    t = load()
    r = rows(t)
    X_all = features(r)
    acct = r.acct.values
    starts = np.r_[0, np.flatnonzero(acct[1:] != acct[:-1]) + 1]
    ends = np.r_[starts[1:], len(r)]
    lau = r.lau.values
    ts = r.ts.values
    print(f'{len(t):,} transfers, {len(starts):,} accounts, {len(r):,} account-transactions ({time.time() - t0:.0f} s)', flush=True)

    # per account: first and last laundering row
    lab = np.zeros(len(r), np.float32)
    ev_tr, ev_te, normal = [], [], []
    for s, e in zip(starts, ends):
        k = np.flatnonzero(lau[s:e])
        if len(k):
            a, b = s + k[0], s + k[-1] + 1 + SLACK
            b = min(b, e, a + MAXCH + SLACK)
            lab[a:b] = 1
            if ts[a] < SPLIT_TRAIN:
                ev_tr.append((s, e, a, b))
            elif ts[a] >= SPLIT_TEST and a - s >= PRE:
                ev_te.append((s, e, a, b))
        elif e - s >= L:
            normal.append((s, e))
    print(f'laundering accounts: {len(ev_tr)} train, {len(ev_te)} test; normal accounts with >= {L} transactions: {len(normal):,}',
          flush=True)

    xs, ys = [], []
    for s, e, a, b in ev_tr:
        for _ in range(3):
            st = a - int(rng.integers(5, PRE + 1))
            if st >= s and st + L <= e:
                xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
    # normal windows that end before the training cut: from any never-involved account with enough history by then
    tr_norm = [(s, s + int(np.searchsorted(ts[s:e], np.datetime64(SPLIT_TRAIN)))) for s, e in normal]
    tr_norm = [(s, e) for s, e in tr_norm if e - s >= L]
    for k in rng.choice(len(tr_norm), min(len(tr_norm), 3000 if QUICK else 20000), replace=False):
        s, e = tr_norm[k]
        st = int(rng.integers(s, e - L + 1))
        xs.append(X_all[st:st + L]); ys.append(lab[st:st + L])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)

    X, tau1, tau2, info = [], [], [], []
    for s, e, a, b in ev_te:
        seg = np.zeros((L, X_all.shape[1]), np.float32)
        seg[:b - a + PRE] = X_all[a - PRE:b]
        X.append(seg); tau1.append(PRE); tau2.append(b - a + PRE)
        k = np.flatnonzero(lau[s:e])
        info.append({'fmt': r.fmt.values[a], 'n_laundering': int(len(k)), 'out': bool(r.out.values[a])})
    n_ev = len(X)
    te_norm = [(s, e) for s, e in normal if (ts[s:e] >= SPLIT_TEST).sum() >= L]
    for k in rng.choice(len(te_norm), min(len(te_norm), 2000 if QUICK else 8000), replace=False):
        s, e = te_norm[k]
        s2 = s + int(np.argmax(ts[s:e] >= SPLIT_TEST))
        st = int(rng.integers(s2, e - L + 1))
        X.append(X_all[st:st + L]); tau1.append(PRE); tau2.append(PRE)     # false alarms counted on the same 15 steps
    X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
    valid = np.arange(L)[None, :] < np.where(np.arange(len(X)) < n_ev, tau2, PRE)[:, None]
    Xn = norm(X)
    print(f'training windows {len(xs)} ({(ys.max(1) > 0).mean():.0%} with laundering); test {n_ev} laundering accounts, '
          f'{len(X) - n_ev} normal windows', flush=True)

    F = {f: i for i, f in enumerate(FEATURES)}
    roll = lambda a, k: np.cumsum(a, 1) - np.c_[np.zeros((len(a), k)), np.cumsum(a, 1)[:, :-k]]

    def above_usual(A):
        usual = np.cumsum(A[:, :, F['log_usd']], 1) / np.arange(1, A.shape[1] + 1)[None, :]
        return A[:, :, F['log_usd']] - np.c_[usual[:, :1], usual[:, :-1]]
    rule_fns = {'Rule: ACH with a new counterparty': lambda A: A[:, :, F['ACH']] * A[:, :, F['new_cp']],
                'Rule: amount above usual': above_usual,
                'Rule: new counterparties, last 5': lambda A: roll(A[:, :, F['new_cp']], 5),
                'Rule: hub contact': lambda A: A[:, :, F['cp_degree']],
                'Rule: cross-currency': lambda A: A[:, :, F['cross_cur']]}
    rules = {n: fn(X).astype(float) for n, fn in rule_fns.items()}
    nrm = xs[[y.max() == 0 for y in ys]]                     # raw normal training windows set each rule's scale
    comb = np.max([fn(X) / max(np.quantile(fn(nrm)[:, WARM:], 0.99), 1e-3) for fn in rule_fns.values()], 0)
    det = {'Logistic (this step only)': uc.logistic_step(xs_n, ys, Xn), 'WL-CUSUM': uc.wl_cusum(xs_n, ys, Xn, WL_K), **rules, 'Rules combined': comb.astype(float)}
    deep, _ = uc.deepqcd(xs_n, ys, Xn, seeds=(0,) if QUICK else (0, 1, 2), epochs=2 if QUICK else 20, rng=rng)
    fmts = np.array([i['fmt'] for i in info])
    multi = np.array([i['n_laundering'] > 1 for i in info])
    groups = {'ACH': fmts == 'ACH', 'not ACH': fmts != 'ACH', 'one laundering transfer': ~multi, 'several': multi}
    res = {'settings': {'pre': PRE, 'maxch': MAXCH, 'slack': SLACK, 'warm': WARM, 'wl_k': WL_K, 'features': FEATURES},
           'n_transfers': int(len(t)), 'n_accounts': int(len(starts)), 'n_train_events': len(ev_tr),
           'n_train_windows': int(len(xs)), 'n_test_events': n_ev, 'n_test_no': int(len(X) - n_ev),
           'group_sizes': {g: int(m.sum()) for g, m in groups.items()}, 'detectors': {}}
    outs = [uc.evaluate(d, tau1, tau2, n_ev, WARM, valid, groups)[0] for d in deep]
    for k, o in enumerate(outs):
        res['detectors'][f'DeepQCD s{k}'] = o
    res['detectors']['DeepQCD'] = uc.merge_seeds(outs)
    uc.report('DeepQCD', res['detectors']['DeepQCD'])
    for name, d in det.items():
        o, _ = uc.evaluate(d, tau1, tau2, n_ev, WARM, valid, groups)
        res['detectors'][name] = o
        uc.report(name, o)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'amlworld.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

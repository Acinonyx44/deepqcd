"""
Money laundering as transient change detection, on IBM AMLSim banks (fetch_amlsim.sh).

Each account is a daily stream. When it is drawn into a laundering scheme (one of AMLSim's eight typologies: fan-in,
fan-out, cycle, bipartite, stack, random, scatter-gather, gather-scatter) its behaviour changes for the 5-20 days the
scheme runs and then returns to normal: the paper's transient change. AMLSim labels every laundering transaction, so
the labels are exact and plentiful (about 2,200 accounts per bank), and laundering amounts are drawn from the same
range as normal transfers, so size alone does not give it away.

Daily features of an account (P = 9), the kind a bank's monitoring computes:
    0 n_in, 1 n_out        log(1 + number of incoming / outgoing transfers)
    2 amt_in, 3 amt_out    log(1 + amount / 100)
    4 new_in, 5 new_out    log(1 + number of counterparties seen for the first time)
    6 deg7                 log(1 + distinct counterparties over the last 7 days)
    7 pass3                log(1 + min(money in, money out) over the last 3 days / 100): money passing through
    8 hub                  log(1 + largest deg7 among today's counterparties): contact with an account that suddenly
                           deals with many others (the only feature that looks beyond the account itself)
Change: from the account's first laundering transaction (tau1) to 3 days after its last (tau2); a bank that flags an
account within 3 days of its last laundering transfer still catches it while the money is moving.
Test (native streams, banks 5 and 6, never seen in training): one stream per laundering account whose first
laundering transaction comes after day 120: the 120 days before tau1 and the change; plus 4,000 windows of accounts
never involved. PFA = P(alarm in days 14..119) (14 days of warm-up), PD = P(first alarm in [tau1, tau2)).
Detectors: DeepQCD (the paper's network and training, labels 1 during laundering; banks 1-4), the window-limited
CUSUM (window_dec_stat over 7 days on the Gaussian-LLR projection), and five standard AML rules: money passing
through (pass3), many counterparties (deg7), transaction velocity (transfers over 7 days), contact with a hub (hub) and
new counterparties (over 3 days), alone and combined (each scaled by its 99th percentile on normal days, then the largest).

Run:  .venv/bin/python aml_qcd.py [--quick]      -> runs/aml/results.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from original_qcd import Model, fit, normalization, scores, window_dec_stat_streams
from transient_real import projection

QUICK = '--quick' in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'amlsim')
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'aml')
TRAIN_BANKS, TEST_BANKS = (1, 2, 3, 4), (5, 6)
DAYS, PRE, MAXCH, SLACK, WARM, WL_K = 720, 120, 30, 3, 14, 7
L = PRE + MAXCH + SLACK
N_NO_TEST = 1000 if QUICK else 4000
SEEDS = (0,) if QUICK else (0, 1, 2)
EPOCHS = 2 if QUICK else 20
ALPHAS = (0.01, 0.05, 0.1)
FEATURES = ['n_in', 'n_out', 'amt_in', 'amt_out', 'new_in', 'new_out', 'deg7', 'pass3', 'hub']


def roll(a, k):
    c = np.cumsum(a, 1)
    out = c.copy()
    out[:, k:] -= c[:, :-k]
    return out


def dormant(A):
    """log(1 + days since the previous transfer) on a day with a transfer, else 0 (the "dormant account" scenario).
    Days before the start of the stream count as inactive."""
    act = (A[:, :, 0] + A[:, :, 1]) > 0
    n, T = act.shape
    last = np.where(act, np.arange(T)[None, :], -1)
    prev = np.maximum.accumulate(np.c_[np.full((n, 1), -1), last[:, :-1]], 1)
    gap = np.where(prev >= 0, np.arange(T)[None, :] - prev, np.arange(T)[None, :] + 30)
    return np.where(act, np.log1p(gap), 0.0)


def bank(seed):
    """Daily features (accounts, DAYS, P), union labels, per-account laundering spans, roles and typologies."""
    d = os.path.join(DATA, f'qcd10K_s{seed}')
    tx = pd.read_csv(os.path.join(d, 'tx_log.csv'), usecols=['step', 'amount', 'nameOrig', 'nameDest', 'isSAR', 'alertID'])
    tx = tx[tx.step < DAYS]
    am = pd.read_csv(os.path.join(d, 'alert_members.csv'))
    ids = np.unique(np.r_[tx.nameOrig.values, tx.nameDest.values])
    ix = {a: i for i, a in enumerate(ids)}
    A = len(ids)
    o, b = tx.nameOrig.map(ix).values, tx.nameDest.map(ix).values
    day, amt = tx.step.values.astype(int), tx.amount.values
    f = np.zeros((9, A, DAYS), np.float32)
    np.add.at(f[0], (b, day), 1)
    np.add.at(f[1], (o, day), 1)
    np.add.at(f[2], (b, day), amt)
    np.add.at(f[3], (o, day), amt)
    pair = pd.DataFrame({'o': o, 'b': b, 'day': day}).sort_values('day', kind='stable')
    first = ~pair.duplicated(['o', 'b'])
    np.add.at(f[4], (pair.b.values[first], pair.day.values[first]), 1)
    np.add.at(f[5], (pair.o.values[first], pair.day.values[first]), 1)
    # distinct counterparties over the last 7 days, via coverage intervals of each (account, counterparty) pair
    ev = pd.DataFrame({'a': np.r_[o, b], 'c': np.r_[b, o], 'day': np.r_[day, day]}).drop_duplicates().sort_values(['a', 'c', 'day'])
    av, cv, dv = ev.a.values, ev.c.values, ev.day.values
    prev = np.r_[-10 ** 6, dv[:-1]]
    same = np.r_[False, (av[1:] == av[:-1]) & (cv[1:] == cv[:-1])]
    start = np.where(same, np.maximum(dv, prev + 7), dv)
    ok = start <= dv + 6
    diff = np.zeros((A, DAYS + 8), np.float32)
    np.add.at(diff, (av[ok], start[ok]), 1)
    np.add.at(diff, (av[ok], dv[ok] + 7), -1)
    deg7 = np.cumsum(diff, 1)[:, :DAYS]
    f[6] = deg7
    f[7] = np.minimum(roll(f[2], 3), roll(f[3], 3))
    np.maximum.at(f[8], (o, day), deg7[b, day])
    np.maximum.at(f[8], (b, day), deg7[o, day])
    x = np.stack([np.log1p(f[0]), np.log1p(f[1]), np.log1p(f[2] / 100), np.log1p(f[3] / 100), np.log1p(f[4]), np.log1p(f[5]),
                  np.log1p(f[6]), np.log1p(f[7] / 100), np.log1p(f[8])], 2)
    vel7 = roll(f[0] + f[1], 7)
    # laundering spans per account (first to last SAR transaction day, all alerts) and the union label
    sar = tx.isSAR.values == 1
    spans = defaultdict(list)
    for acct, dd, al in zip(np.r_[o[sar], b[sar]], np.r_[day[sar], day[sar]], np.r_[tx.alertID.values[sar], tx.alertID.values[sar]]):
        spans[(acct, al)].append(dd)
    lab = np.zeros((A, DAYS), np.float32)
    span = {}
    for (acct, al), dd in spans.items():
        a0, a1 = min(dd), max(dd)
        lab[acct, a0:a1 + 1 + SLACK] = 1
        if acct not in span or a0 < span[acct][0]:
            span[acct] = (a0, a1, al)
    role = {}
    for r in am.itertuples():
        if r.accountID in ix:
            k = ix[r.accountID]
            role[(k, r.alertID)] = ('main' if str(r.isMain).lower() == 'true' else 'member', r.reason)
    return {'x': x, 'lab': lab, 'span': span, 'role': role, 'vel7': np.log1p(vel7), 'ever': lab.max(1) > 0}


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    B = {s: bank(s) for s in TRAIN_BANKS + TEST_BANKS}
    print(f'banks built ({time.time() - t0:.0f} s): ' + ', '.join(f'{s}: {len(b["x"])} accounts, {len(b["span"])} laundering'
                                                                      for s, b in B.items()), flush=True)

    # ---- training windows: each laundering account at three random offsets, plus normal windows
    xs, ys = [], []
    for s in TRAIN_BANKS:
        b = B[s]
        for acct, (a0, a1, al) in b['span'].items():
            for _ in range(1 if QUICK else 3):
                st = a0 - int(rng.integers(30, PRE + 10))
                if 0 <= st and st + L <= DAYS:
                    xs.append(b['x'][acct, st:st + L])
                    ys.append(b['lab'][acct, st:st + L])
        normal = np.flatnonzero(~b['ever'])
        for acct in rng.choice(normal, 300 if QUICK else 2500, replace=False):
            st = int(rng.integers(0, DAYS - L))
            xs.append(b['x'][acct, st:st + L])
            ys.append(b['lab'][acct, st:st + L])
    xs, ys = np.stack(xs), np.stack(ys)
    norm = normalization([x for x, y in zip(xs, ys) if y.max() == 0])
    xs_n = norm(xs)
    perm = rng.permutation(len(xs))
    nv = len(perm) // 7
    va, tr = perm[:nv], perm[nv:]
    print(f'training windows: {len(xs)} ({(ys.max(1) > 0).mean():.0%} with laundering)', flush=True)

    # ---- test streams
    X, tau1, tau2, info = [], [], [], []
    for s in TEST_BANKS:
        b = B[s]
        for acct, (a0, a1, al) in b['span'].items():
            if a0 < PRE or b['lab'][acct, a0 - PRE:a0].max() > 0:
                continue
            e = min(a1 + 1 + SLACK, a0 + MAXCH + SLACK, DAYS)
            seg = np.zeros((L, len(FEATURES)), np.float32)
            seg[:e - (a0 - PRE)] = b['x'][acct, a0 - PRE:e]
            X.append(seg)
            tau1.append(PRE)
            tau2.append(e - (a0 - PRE))
            r, typ = b['role'].get((acct, al), ('member', '?'))
            info.append({'bank': s, 'role': r, 'type': typ, 'span': int(a1 - a0 + 1),
                         'active_before': bool(b['x'][acct, a0 - PRE:a0, :2].sum() > 0)})
    n_ev = len(X)
    for k in range(N_NO_TEST):
        s = TEST_BANKS[k % len(TEST_BANKS)]
        b = B[s]
        acct = int(rng.choice(np.flatnonzero(~b['ever'])))
        st = int(rng.integers(0, DAYS - L))
        X.append(b['x'][acct, st:st + L])
        tau1.append(PRE)
        tau2.append(L + 1)
    X, tau1, tau2 = np.stack(X), np.array(tau1), np.array(tau2)
    Xn = norm(X)
    print(f'test: {n_ev} laundering accounts, {N_NO_TEST} normal windows', flush=True)

    # ---- detectors
    det = {}
    pre = [xs_n[i][ys[i] == 0] for i in range(len(xs_n))]
    post = [xs_n[i][ys[i] == 1] for i in range(len(xs_n)) if ys[i].max() > 0]
    z, mu = projection(pre, post)
    zz = z(Xn)
    det['WL-CUSUM'] = np.c_[np.full((len(zz), WL_K), -np.inf), window_dec_stat_streams(zz[:, WL_K:], zz[:, :WL_K], mu)]
    det['Rule: pass-through (3 days)'] = X[:, :, 7].astype(float)
    det['Rule: counterparties (7 days)'] = X[:, :, 6].astype(float)
    det['Rule: velocity (7 days)'] = np.log1p(roll((np.expm1(X[:, :, 0]) + np.expm1(X[:, :, 1])), 7)).astype(float)
    det['Rule: hub contact'] = X[:, :, 8].astype(float)
    newcp = lambda A: np.log1p(roll(np.expm1(A[:, :, 4]) + np.expm1(A[:, :, 5]), 3)).astype(float)
    det['Rule: new counterparties (3 days)'] = newcp(X)
    det['Rule: dormant account reactivated'] = dormant(X)
    det['Rule: dormant + new counterparty'] = dormant(X) * (newcp(X) > 0)
    # all five rules at once, as a bank runs its scenarios: each scaled by its 99th percentile on normal training days
    rule_fns = [lambda A: A[:, :, 7], lambda A: A[:, :, 6], lambda A: np.log1p(roll(np.expm1(A[:, :, 0]) + np.expm1(A[:, :, 1]), 7)),
                lambda A: A[:, :, 8], newcp, dormant, lambda A: dormant(A) * (newcp(A) > 0)]
    nrm = np.stack([x for x, y in zip(xs, ys) if y.max() == 0])
    det['Rules combined'] = np.max([fn(X) / max(np.quantile(fn(nrm)[:, WARM:], 0.99), 1e-6) for fn in rule_fns], 0).astype(float)
    hist, deep = {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Model(len(FEATURES))
        hist[f's{seed}'] = fit(model, xs_n[tr], ys[tr], xs_n[va], ys[va], epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
        model.eval()
        deep.append(scores(model, Xn))
        det[f'DeepQCD s{seed}'] = deep[-1]

    t_idx = np.arange(L)[None, :]
    roles = np.array([i['role'] for i in info])
    types = np.array([i['type'] for i in info])
    activeb = np.array([i['active_before'] for i in info])
    res = {'settings': {'pre': PRE, 'maxch': MAXCH, 'slack': SLACK, 'warm': WARM, 'wl_k': WL_K, 'features': FEATURES,
                        'train_banks': TRAIN_BANKS, 'test_banks': TEST_BANKS},
           'n_train_windows': int(len(xs)), 'n_test_events': n_ev, 'n_test_no': N_NO_TEST,
           'n_main': int((roles == 'main').sum()), 'n_member': int((roles == 'member').sum()),
           'types': {t: int((types == t).sum()) for t in np.unique(types)}, 'n_active_before': int(activeb.sum()),
           'detectors': {}, 'history': hist}
    for name, d in det.items():
        m_pre = np.where((t_idx >= WARM) & (t_idx < tau1[:, None]), d, -np.inf).max(1)
        m_win = np.where((t_idx >= tau1[:, None]) & (t_idx < tau2[:, None]), d, -np.inf).max(1)
        fin = np.r_[m_pre[np.isfinite(m_pre)], m_win[np.isfinite(m_win)]]
        h = np.unique(np.r_[np.quantile(fin, np.linspace(0, 1, 1000)), np.inf])
        pfa = (m_pre[None, :] >= h[:, None]).mean(1)
        hit = (m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])
        pd = hit.mean(1)
        out = {'pd_at': {}, 'by_role': {}, 'by_type': {}, 'delay_at': {}, 'active_before': {}}
        for a in ALPHAS:
            ok = pfa <= a
            j = int(np.flatnonzero(ok)[np.argmax(pd[ok])]) if ok.any() else len(h) - 1
            out['pd_at'][str(a)] = float(pd[j])
            out['by_role'][str(a)] = {r: float(hit[j, roles == r].mean()) for r in ('main', 'member')}
            out['by_type'][str(a)] = {t: float(hit[j, types == t].mean()) for t in np.unique(types)}
            out['active_before'][str(a)] = float(hit[j, activeb].mean())
            dl = [int(np.argmax(d[i, PRE:] >= h[j])) for i in np.flatnonzero(hit[j])]
            out['delay_at'][str(a)] = float(np.median(dl)) if dl else None
        o = np.argsort(pfa)
        out['pfa'], out['pd'] = np.round(pfa[o], 4).tolist(), np.round(pd[o], 4).tolist()
        res['detectors'][name] = out
        br = out['by_role']['0.05']
        print(f'{name:30s} PD at PFA ' + '  '.join(f'{a}: {out["pd_at"][a]:.3f}' for a in out['pd_at']) +
              f'   at 0.05: main {br["main"]:.2f}, member {br["member"]:.2f}, active before {out["active_before"]["0.05"]:.2f}, '
              f'delay {out["delay_at"]["0.05"]} d', flush=True)
    ds = [res['detectors'][f'DeepQCD s{k}'] for k in range(len(SEEDS))]
    res['detectors']['DeepQCD'] = dict(ds[0], pd_at={a: float(np.mean([x['pd_at'][a] for x in ds])) for a in ds[0]['pd_at']},
                                       pd_at_seeds={a: [x['pd_at'][a] for x in ds] for a in ds[0]['pd_at']},
                                       by_role={a: {r: float(np.mean([x['by_role'][a][r] for x in ds])) for r in ('main', 'member')}
                                                for a in ds[0]['by_role']},
                                       by_type={a: {t: float(np.mean([x['by_type'][a][t] for x in ds])) for t in ds[0]['by_type'][a]}
                                                for a in ds[0]['by_type']},
                                       active_before={a: float(np.mean([x['active_before'][a] for x in ds])) for a in ds[0]['active_before']})
    exi = [i for i in range(n_ev) if roles[i] == 'main'][:4] + [i for i in range(n_ev) if roles[i] == 'member'][:4]
    res['examples'] = [{'role': roles[i], 'type': types[i], 'tau2': int(tau2[i]), 'x': np.round(X[i, :tau2[i]], 3).tolist(),
                        'd': np.round(deep[0][i, :tau2[i]], 4).tolist()} for i in exi]
    res['feature_means'] = {
        'normal': np.round(X[n_ev:, WARM:PRE].reshape(-1, len(FEATURES)).mean(0), 3).tolist(),
        'laundering, main': np.round(np.concatenate([X[i, PRE:tau2[i]] for i in range(n_ev) if roles[i] == 'main']).mean(0), 3).tolist(),
        'laundering, member': np.round(np.concatenate([X[i, PRE:tau2[i]] for i in range(n_ev) if roles[i] == 'member']).mean(0), 3).tolist()}
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

"""
Use case 5, banks: early warning of credit-card default (UCI "Default of credit card clients", Yeh & Lien 2009:
30,000 Taiwanese card holders, April-September 2005, and whether each defaulted in October; 22% did).

Each client is a stream of 6 monthly steps (April -> September) with what the bank sees each month: repayment status
(-2 no use, -1 paid in full, 0 revolving, k = k months late), utilization (bill / limit), log bill, log payment, payment
against the previous bill, bill change, and the (fixed) log limit and age. The change is "on the way to default": the
Bayesian setting where it has already happened for a defaulter and never happens for the others, so DeepQCD's label
is 1 in every month of a defaulter. An alarm in any month flags the client; PD = share of defaulters flagged by
September, PFA = share of non-defaulters flagged. Lead time = months between the first alarm and October.
Detectors: DeepQCD (three seeds), WL-CUSUM over 3 months, a logistic scorecard on the current month (the industry
baseline: uc.logistic_step), and rules: months late now, utilization, low payment, combined.
Split: 70% of clients train, 30% test (no dates to split on).

Run:  .venv/bin/python usecase_credit.py      -> runs/usecases/credit.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

import usecases_common as uc
from original_qcd import normalization

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'runs', 'usecases')
FEATURES = ['status', 'utilization', 'log_bill', 'log_payment', 'pay_ratio', 'bill_change', 'log_limit', 'age']


def streams():
    d = pd.read_csv(os.path.join(ROOT, 'data', 'uci', 'default.csv'))
    status = d[['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']].values.astype(float)           # April .. September
    bill = d[[f'BILL_AMT{k}' for k in (6, 5, 4, 3, 2, 1)]].values.astype(float)
    pay = d[[f'PAY_AMT{k}' for k in (6, 5, 4, 3, 2, 1)]].values.astype(float)
    lim = d.LIMIT_BAL.values.astype(float)[:, None]
    prev = np.c_[bill[:, :1], bill[:, :-1]]
    x = np.stack([np.clip(status, -2, 8), np.clip(bill / lim, -1, 3), np.sign(bill) * np.log1p(np.abs(bill) / 1000),
                  np.log1p(pay / 1000), np.clip(pay / np.maximum(prev, 1), 0, 2), np.clip((bill - prev) / np.maximum(np.abs(prev), 1000), -3, 3),
                  np.repeat(np.log(lim), 6, 1), np.repeat(d.AGE.values[:, None] / 100.0, 6, 1)], 2).astype(np.float32)
    return x, d['default payment next month'].values.astype(bool)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    x, dflt = streams()
    perm = rng.permutation(len(x))
    ntr = int(0.7 * len(x))
    tr, te = perm[:ntr], perm[ntr:]
    ys = np.repeat(dflt[tr, None].astype(np.float32), 6, 1)
    norm = normalization(list(x[tr][~dflt[tr]]))
    xs_n = norm(x[tr])
    order = np.r_[te[dflt[te]], te[~dflt[te]]]                     # change streams first
    X = x[order]
    n_ev = int(dflt[te].sum())
    tau1 = np.where(np.arange(len(X)) < n_ev, 0, 6)
    tau2 = np.full(len(X), 6)
    Xn = norm(X)
    nrm = x[tr][~dflt[tr]]
    rule_fns = {'Rule: months late now': lambda A: A[:, :, 0], 'Rule: utilization': lambda A: A[:, :, 1],
                'Rule: low payment': lambda A: 1 - np.minimum(A[:, :, 4], 1)}
    comb = np.max([fn(X) / max(np.quantile(fn(nrm), 0.99), 1e-3) for fn in rule_fns.values()], 0)
    det = {'Logistic (this step only)': uc.logistic_step(xs_n, ys, Xn),
           'WL-CUSUM': uc.wl_cusum(xs_n, ys, Xn, 3), **{n: fn(X).astype(float) for n, fn in rule_fns.items()},
           'Rules combined': comb.astype(float)}
    # (the CUSUM uses months 1-3 to start its window, so it can alarm from month 4)
    deep, _ = uc.deepqcd(xs_n, ys, Xn, seeds=(0, 1, 2), epochs=20, rng=rng)
    res = {'n_clients': int(len(x)), 'default_rate': float(dflt.mean()), 'n_train': int(ntr), 'n_test_defaulters': n_ev,
           'n_test_others': int(len(X) - n_ev), 'features': FEATURES, 'detectors': {}}

    def lead(d, h):
        m = [uc.first_alarm(d[i], h, 0, 6) for i in range(n_ev)]
        m = [6 - k for k in m if k >= 0]                           # months from the alarm month to October
        return float(np.median(m)) if m else None
    outs = []
    for k, dd in enumerate(deep):
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, 0)
        o['lead_months'] = {a: lead(dd, o['h_at'][a]) for a in o['h_at']}
        res['detectors'][f'DeepQCD s{k}'] = o
        outs.append(o)
    res['detectors']['DeepQCD'] = uc.merge_seeds(outs)
    uc.report('DeepQCD', res['detectors']['DeepQCD'])
    for name, dd in det.items():
        o, _ = uc.evaluate(dd, tau1, tau2, n_ev, 0)
        o['lead_months'] = {a: lead(dd, o['h_at'][a]) for a in o['h_at']}
        res['detectors'][name] = o
        uc.report(name, o)
    # PD at PFA 0.2 (credit teams accept more false flags than fraud teams)
    for name, o in res['detectors'].items():
        pfa, pd = np.array(o['pfa']), np.array(o['pd'])
        o['pd_at_0.2'] = float(pd[pfa <= 0.2].max()) if (pfa <= 0.2).any() else 0.0
    print('PD at PFA 0.2:', {k: round(v['pd_at_0.2'], 3) for k, v in res['detectors'].items() if 's1' not in k and 's2' not in k})
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'credit.json'), 'w'))
    print(f'done in {res["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()

"""
Data for the short-squeeze use case (usecase_squeeze.py).

1. FINRA Reg SHO daily short-sale volume, consolidated NMS file (public): for every trading day from 2021-01-04 to
   2026-10-16, Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market -> data/squeeze/finra/CNMSshvol<date>.txt
2. The universe: common-stock-like symbols (1-4 capital letters) that traded on at least 80% of those days with a
   median total volume of at least 300,000 shares (FINRA's reported volume), and a median short-volume share of at
   least 40%.
3. Yahoo Finance daily adjusted OHLCV for the universe, 2020-10-01 to 2026-10-17 -> data/squeeze/yahoo/<SYM>.json

Run:  .venv/bin/python squeeze_fetch.py [finra|universe|yahoo|all]
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, 'data', 'squeeze')
START, END = dt.date(2021, 1, 4), dt.date(2026, 10, 16)


def get(url, tries=5, ua=False, binary=False):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0' if ua else 'research'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read() if binary else r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * 2 ** k)
        except Exception:
            time.sleep(3 * 2 ** k)
    return None


def finra():
    os.makedirs(os.path.join(D, 'finra'), exist_ok=True)
    days = [START + dt.timedelta(k) for k in range((END - START).days + 1)]
    days = [d for d in days if d.weekday() < 5]

    def one(d):
        p = os.path.join(D, 'finra', f'CNMSshvol{d:%Y%m%d}.txt')
        if os.path.exists(p):
            return 1
        txt = get(f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt')
        if txt is None:                     # holiday
            return 0
        open(p, 'w').write(txt)
        return 1
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, days))
    print(f'FINRA: {n} trading days', flush=True)


def universe():
    files = sorted(os.listdir(os.path.join(D, 'finra')))
    vol, ratio = defaultdict(list), defaultdict(list)
    for f in files:
        for line in open(os.path.join(D, 'finra', f)).read().splitlines()[1:]:
            parts = line.split('|')
            if len(parts) < 5 or not parts[1].isalpha() or not parts[1].isupper() or len(parts[1]) > 4:
                continue
            try:
                sv, tv = float(parts[2]), float(parts[4])
            except ValueError:
                continue
            if tv > 0:
                vol[parts[1]].append(tv)
                ratio[parts[1]].append(sv / tv)
    n = len(files)
    syms = sorted(s for s in vol if len(vol[s]) >= 0.8 * n and np.median(vol[s]) >= 3e5 and np.median(ratio[s]) >= 0.4)
    json.dump(syms, open(os.path.join(D, 'universe.json'), 'w'))
    print(f'universe: {len(syms)} symbols out of {len(vol)}', flush=True)


def yahoo():
    os.makedirs(os.path.join(D, 'yahoo'), exist_ok=True)
    syms = json.load(open(os.path.join(D, 'universe.json')))
    p1 = int(dt.datetime(2020, 10, 1).timestamp())
    p2 = int(dt.datetime(2026, 10, 17).timestamp())
    ok = 0
    for k, s in enumerate(syms):
        p = os.path.join(D, 'yahoo', f'{s}.json')
        if os.path.exists(p):
            ok += 1
            continue
        txt = get(f'https://query2.finance.yahoo.com/v8/finance/chart/{s}?period1={p1}&period2={p2}&interval=1d&events=split',
                  ua=True)
        if txt:
            open(p, 'w').write(txt)
            ok += 1
        time.sleep(0.4)
        if k % 100 == 0:
            print(f'yahoo {k}/{len(syms)}: {ok} ok', flush=True)
    print(f'yahoo: {ok} of {len(syms)}', flush=True)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('finra', 'all'):
        finra()
    if what in ('universe', 'all'):
        universe()
    if what in ('yahoo', 'all'):
        yahoo()

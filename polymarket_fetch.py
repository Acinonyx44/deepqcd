"""
Download hourly YES prices and large trades (>= $1,000, with wallets) for Polymarket deadline markets ("X by <date>?").

Input:  data/polymarket/deadline_markets.json: closed Yes/No markets whose question contains "by" or "before", open
        for at least 3 days and resolved to 0 or 1, with at least $100,000 volume (made by --list from the gamma API,
        keyset pagination, since offset paging stops at about 2,500).
Output: data/polymarket/m/<conditionId>.json = {'prices': [[t, p_yes], ...], 'trades': [[t, wallet, yes_flow_usd], ...]}
        yes_flow_usd > 0 is money moving the price toward YES (buy YES or sell NO), < 0 toward NO.

APIs (public, no key): clob /prices-history (windows of at most ~14 days; fidelity in minutes) and data-api /trades
(taker trades, newest first; paged backwards in time with `end`, since `offset` stops at 10,000).

Run:  .venv/bin/python polymarket_fetch.py --list               -> data/polymarket/deadline_markets.json
      .venv/bin/python polymarket_fetch.py [min_volume_usd]     -> data/polymarket/m/
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'polymarket')
OUT = os.path.join(DATA, 'm')
MIN_VOL = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '--list' else 1e6
MIN_CASH = 1000
WINDOW = 14 * 86400


def get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research'}), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def iso(s):
    if not s:
        return None
    s = s.replace('Z', '+00:00').replace(' ', 'T')
    if re.search(r'[+-]\d\d$', s):
        s += ':00'
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def list_markets(min_volume=1e5):
    """Closed markets by volume (keyset pages of 100) down to min_volume; keep the resolved Yes/No deadline questions."""
    allm, cur = [], None
    while True:
        u = 'https://gamma-api.polymarket.com/markets/keyset?closed=true&limit=100&order=volumeNum&ascending=false'
        if cur:
            u += '&after_cursor=' + urllib.parse.quote(cur)
        d = get(u)
        if not d.get('markets'):
            break
        allm += d['markets']
        cur = d.get('next_cursor')
        if not cur or float(d['markets'][-1].get('volumeNum') or 0) < min_volume:
            break
    json.dump(allm, open(os.path.join(DATA, 'closed_markets_raw.json'), 'w'))
    keep, seen = [], set()
    for m in allm:
        try:
            outs, pr = json.loads(m.get('outcomes') or '[]'), json.loads(m.get('outcomePrices') or '[]')
        except ValueError:
            continue
        q = m.get('question') or ''
        if m['conditionId'] in seen or outs != ['Yes', 'No'] or len(pr) != 2 or not re.search(r'\bby\b|\bbefore\b', q, re.I):
            continue
        s, c = iso(m.get('startDate')), iso(m.get('closedTime'))
        if not s or not c or c - s < 3 * 86400 or float(pr[0]) not in (0.0, 1.0):
            continue
        seen.add(m['conditionId'])
        keep.append({'q': q, 'cid': m['conditionId'], 'yes_token': json.loads(m['clobTokenIds'])[0], 'start': s, 'closed': c,
                     'end': iso(m.get('endDate')), 'yes': int(float(pr[0]) == 1.0), 'volume': float(m.get('volumeNum') or 0),
                     'negRisk': bool(m.get('negRisk')), 'slug': m.get('slug')})
    json.dump(keep, open(os.path.join(DATA, 'deadline_markets.json'), 'w'))
    print(f'{len(allm)} closed markets listed; {len(keep)} deadline markets kept, {sum(k["yes"] for k in keep)} resolved YES')


def prices(m):
    out, t = {}, int(m['start'])
    stop = int(m['closed']) + 86400
    while t < stop:
        h = get(f'https://clob.polymarket.com/prices-history?market={m["yes_token"]}&startTs={t}&endTs={min(t + WINDOW, stop)}'
                f'&fidelity=60')
        for row in h.get('history', []):
            out[row['t']] = row['p']
        t += WINDOW
    return sorted(out.items())


def trades(m):
    seen, rows, end = set(), [], int(m['closed']) + 86400
    while True:
        page = get(f'https://data-api.polymarket.com/trades?market={m["cid"]}&filterType=CASH&filterAmount={MIN_CASH}'
                   f'&limit=1000&end={end}')
        new = 0
        for r in page:
            key = (r['transactionHash'], r['proxyWallet'], r['asset'], r['side'], r['size'])
            if key in seen:
                continue
            seen.add(key)
            new += 1
            usd = r['size'] * r['price']
            toward_yes = (r['outcomeIndex'] == 0) == (r['side'] == 'BUY')
            rows.append([r['timestamp'], r['proxyWallet'], round(usd if toward_yes else -usd, 2)])
        if len(page) < 1000 or new == 0:
            break
        oldest = min(r['timestamp'] for r in page)
        end = oldest if new > 0 and oldest < end else end - 1
    rows.sort()
    return rows


def one(m):
    path = os.path.join(OUT, m['cid'] + '.json')
    if os.path.exists(path):
        return 'skip'
    try:
        rec = {'prices': prices(m), 'trades': trades(m)}
    except Exception as e:
        return f'error {e}'
    json.dump(rec, open(path, 'w'))
    return f'{len(rec["prices"])} prices, {len(rec["trades"])} trades'


def main():
    os.makedirs(OUT, exist_ok=True)
    if '--list' in sys.argv:
        return list_markets()
    ms = [m for m in json.load(open(os.path.join(DATA, 'deadline_markets.json')))
          if m['volume'] >= MIN_VOL and m['end'] is not None and m['closed'] - m['start'] >= 3 * 86400]
    print(f'{len(ms)} markets with volume >= ${MIN_VOL:,.0f} ({sum(m["yes"] for m in ms)} resolved YES)', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for k, (m, msg) in enumerate(zip(ms, ex.map(one, ms))):
            if k % 25 == 0 or msg.startswith('error'):
                print(f'{k:5d} {time.time() - t0:6.0f}s  {m["q"][:60]:60s} {msg}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()

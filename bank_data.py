"""
Loaders for the bank studies at scale (bank_final.py). Each returns one stream per entity (an account or a card),
one step per transaction, as plain arrays, and caches them in data/cache/<name>.npz:

    X       (N, P) float32   features, rows sorted by entity and time (raw: standardized later on normal training data)
    start   (E,)   int64     first row of each entity; its rows are start[e] .. start[e + 1] - 1
    ts      (N,)   int64     transaction time, seconds since 1970
    flag    (N,)   bool      laundering / fraudulent transaction
    usd     (N,)   float32   amount in USD
    kind    (N,)   int16     a category per row for breakdowns (payment format, online, laundering pattern ...)
    kinds   list of str      names of the kind codes

Datasets:
    aml-hi-small, aml-li-small, aml-hi-medium   IBM AMLworld (Altman et al., NeurIPS 2023). An account's step is a
        transfer it sends or receives. P = 15, as in usecase_amlworld.py: direction, log USD amount, payment format
        (6), cross-currency, cross-bank, self-transfer, new counterparty, log hours since the account's previous
        transfer, the counterparty's number of distinct counterparties so far. kind: the laundering pattern of the
        transfer from the <name>_Patterns.txt file (fan-in, fan-out, cycle, ...; 'unlisted' for laundering transfers in
        no listed attempt; 'normal' otherwise). Built with pyarrow and numpy so that HI-Medium (32 million transfers,
        64 million account steps) fits in memory.
    tabformer   IBM TabFormer cards (usecase_fraud.py features, P = 15). kind: online or in person.
    sparkov     Sparkov card simulator (B. Harris), the 1,048,575-row mirror on Hugging Face (2012-01 .. 2013-03). The
        mirror rounds card numbers, so a card is identified by its holder (name, birth date, street). P = 12: log
        amount, online, new merchant, new category, log km from home, log hours since the previous transaction, hour
        of day (sin, cos), night (22-04 h), amount against the card's history (z-score so far), transactions in the
        last 24 hours, category's fraud-prone flag (shopping_net, grocery_pos, misc_net). kind: online or in person.
"""
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'data', 'cache')
AML = {'aml-hi-small': ('amlworld', 'HI-Small'), 'aml-li-small': ('amlworld_li', 'LI-Small'),
       'aml-hi-medium': ('amlworld_hm', 'HI-Medium')}
AML_FEATURES = ['out', 'in', 'log_usd', 'ACH', 'Cheque', 'Credit Card', 'Cash', 'Wire', 'Bitcoin', 'cross_cur', 'cross_bank',
                'self', 'new_cp', 'log_gap_h', 'cp_degree']
FORMATS = ['ACH', 'Cheque', 'Credit Card', 'Cash', 'Wire', 'Bitcoin']
SPARKOV_FEATURES = ['log_amount', 'online', 'new_merchant', 'new_category', 'log_km_home', 'log_gap_h', 'hour_sin', 'hour_cos',
                    'night', 'amount_z', 'tx_24h', 'risky_category']


def load(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'{name}.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d['kinds'] = [str(k) for k in d['kinds']]
        d['features'] = [str(k) for k in d['features']]
        return d
    t0 = time.time()
    d = amlworld(*AML[name]) if name in AML else tabformer() if name == 'tabformer' else sparkov()
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})
    print(f'{name}: built and cached in {time.time() - t0:.0f} s', flush=True)
    return d


def _within(start, n):
    """Index of each row within its entity, and the entity of each row."""
    ent = np.repeat(np.arange(len(start)), np.diff(np.r_[start, n]))
    return np.arange(n) - start[ent], ent


def amlworld(folder, tag):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    t0 = time.time()
    names = ['ts', 'fb', 'fa', 'tb', 'ta', 'amt_r', 'cur_r', 'amt_p', 'cur_p', 'fmt', 'lau']
    types = {'ts': pa.string(), 'fb': pa.string(), 'fa': pa.string(), 'tb': pa.string(), 'ta': pa.string(), 'amt_r': pa.float64(),
             'cur_r': pa.string(), 'amt_p': pa.float64(), 'cur_p': pa.string(), 'fmt': pa.string(), 'lau': pa.int8()}
    tb = pcsv.read_csv(os.path.join(ROOT, 'data', folder, f'{tag}_Trans.csv'),
                       read_options=pcsv.ReadOptions(column_names=names, skip_rows=1, block_size=1 << 26),
                       convert_options=pcsv.ConvertOptions(column_types=types))
    n = tb.num_rows
    ts = pc.strptime(tb['ts'], format='%Y/%m/%d %H:%M', unit='s').cast(pa.int64()).to_numpy()
    src = pc.binary_join_element_wise(tb['fb'], tb['fa'], '_')
    dst = pc.binary_join_element_wise(tb['tb'], tb['ta'], '_')
    accts = pc.unique(pa.chunked_array(src.chunks + dst.chunks))
    s_id = pc.index_in(src, value_set=accts).to_numpy().astype(np.int32)
    d_id = pc.index_in(dst, value_set=accts).to_numpy().astype(np.int32)
    del src, dst

    def codes(cols):
        u = pc.unique(pa.chunked_array(sum((tb[c].chunks for c in cols), [])))
        return [pc.index_in(tb[c], value_set=u).to_numpy().astype(np.int16) for c in cols], [str(x) for x in u.to_pylist()]
    (cr, cp_), cur = codes(['cur_r', 'cur_p'])
    (fm,), fmts = codes(['fmt'])
    banks = pc.unique(pa.chunked_array(tb['fb'].chunks + tb['tb'].chunks))
    cross_bank = (pc.index_in(tb['fb'], value_set=banks).to_numpy() != pc.index_in(tb['tb'], value_set=banks).to_numpy())
    amt_r, amt_p = tb['amt_r'].to_numpy(), tb['amt_p'].to_numpy()
    lau = tb['lau'].to_numpy().astype(bool)
    print(f'  {tag}: {n:,} transfers, {len(accts):,} accounts read ({time.time() - t0:.0f} s)', flush=True)

    # USD rates from cross-currency transfers paid (or received) in US dollars, as in usecase_amlworld.py
    usd_c = cur.index('US Dollar')
    rate = np.ones(len(cur))
    have = np.zeros(len(cur), bool)
    m = (cp_ == usd_c) & (cr != usd_c) & (amt_p > 0)
    for c in np.unique(cr[m]):
        k = m & (cr == c)
        rate[c], have[c] = np.median(amt_r[k] / amt_p[k]), True
    m = (cr == usd_c) & (cp_ != usd_c) & (amt_r > 0)
    for c in np.unique(cp_[m]):
        if not have[c]:
            k = m & (cp_ == c)
            rate[c] = np.median(amt_p[k] / amt_r[k])
    usd = (amt_p / rate[cp_]).astype(np.float32)
    del amt_r

    # the laundering pattern of each laundering transfer
    pat_name, pat = ['normal', 'unlisted'], np.zeros(n, np.int16)
    pat[lau] = 1
    keys = {}
    cur_type = None
    for line in open(os.path.join(ROOT, 'data', folder, f'{tag}_Patterns.txt')):
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT'):
            cur_type = line.split('-', 1)[1].split(':')[0].strip()
            if cur_type not in pat_name:
                pat_name.append(cur_type)
        elif line and not line.startswith('END') and cur_type:
            f = line.split(',')
            keys[(f[0], f[1], f[2], f[3], f[4], round(float(f[7]), 2))] = pat_name.index(cur_type)
    li = np.flatnonzero(lau)
    sub = tb.take(pa.array(li))
    cols = [sub[c].to_pylist() for c in ('ts', 'fb', 'fa', 'tb', 'ta')]
    for j, i in enumerate(li):
        k = keys.get((cols[0][j], cols[1][j], cols[2][j], cols[3][j], cols[4][j], round(float(amt_p[i]), 2)))
        if k is not None:
            pat[i] = k
    print(f'  laundering transfers {lau.sum():,}, {(pat > 1).sum():,} in listed patterns', flush=True)
    del tb, sub, amt_p, cols

    # one row per (account, transfer): sent by the account, or received (self-transfers once). 32-bit indices and
    # early deletes keep HI-Medium (64 million rows) within memory.
    slf = s_id == d_id
    tx = np.r_[np.arange(n, dtype=np.int32), np.flatnonzero(~slf).astype(np.int32)]
    acct = np.r_[s_id, d_id[tx[n:]]]
    order = np.lexsort((ts[tx], acct))
    out = order < n
    tx, acct = tx[order], acct[order]
    del order
    cp = np.where(out, d_id[tx], s_id[tx])
    N = len(tx)
    t_r = ts[tx]
    start = np.r_[0, np.flatnonzero(acct[1:] != acct[:-1]) + 1].astype(np.int64)
    ent = np.repeat(np.arange(len(start), dtype=np.int32), np.diff(np.r_[start, N]))
    first_row = start[ent]
    key = acct.astype(np.int64) * len(accts) + cp
    _, first = np.unique(key, return_index=True)
    del key
    new_cp = np.zeros(N, bool)
    new_cp[first] = True
    del first
    cs = np.cumsum(new_cp, dtype=np.int32)
    deg = cs - np.r_[np.int32(0), cs][first_row]
    del cs
    gap = np.r_[np.float32(0), np.diff(t_r).astype(np.float32)] / 3600
    gap[np.arange(N) == first_row] = 24 * 7
    np.clip(gap, 0, 24 * 7, out=gap)
    del first_row, ent
    # the counterparty's distinct counterparties at the time of the transfer
    tm = (t_r - t_r.min()) // 60
    big = np.int64(1) << 22
    k2 = acct.astype(np.int64) * big + tm
    j = np.searchsorted(k2, cp.astype(np.int64) * big + tm, side='right') - 1
    del k2, tm
    np.maximum(j, 0, out=j)
    cp_deg = np.where(acct[j] == cp, deg[j], 0)
    del j, acct, cp, deg
    print(f'  {N:,} account steps built ({time.time() - t0:.0f} s)', flush=True)

    X = np.empty((N, len(AML_FEATURES)), np.float32)
    X[:, 0], X[:, 1] = out, ~out
    X[:, 2] = np.log1p(usd[tx] / 100)
    f = fm[tx]
    for k, name in enumerate(FORMATS):
        X[:, 3 + k] = (f == fmts.index(name)) if name in fmts else 0
    X[:, 9] = (cr != cp_)[tx]
    X[:, 10] = cross_bank[tx]
    X[:, 11] = slf[tx]
    X[:, 12] = new_cp
    X[:, 13] = np.log1p(gap)
    X[:, 14] = np.log1p(cp_deg)
    return {'X': X, 'start': start, 'ts': t_r, 'flag': lau[tx], 'usd': usd[tx], 'kind': pat[tx], 'kinds': pat_name,
            'features': AML_FEATURES, 'fmt': f, 'fmts': fmts, 'tx': tx}


def tabformer():
    import usecase_fraud as uf
    d = uf.load()
    X = uf.features(d)
    card = d.card.values
    start = np.r_[0, np.flatnonzero(card[1:] != card[:-1]) + 1].astype(np.int64)
    ts = d.ts.values.astype('datetime64[s]').astype(np.int64)
    return {'X': X, 'start': start, 'ts': ts, 'flag': d.fraud.values, 'usd': np.abs(d.amount.values).astype(np.float32),
            'kind': X[:, 2].astype(np.int16), 'kinds': ['in person', 'online'], 'features': uf.FEATURES}


def sparkov():
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, 'data', 'sparkov', 'credit_card_fraud_transactions.csv'), engine='pyarrow')
    d['card'] = d['first'] + '|' + d['last'] + '|' + d['dob'].astype(str) + '|' + d['street']
    d = d.sort_values(['card', 'unix_time'], kind='stable').reset_index(drop=True)
    first = lambda cols: ~d.duplicated(['card'] + cols).values
    g = d.groupby('card', sort=False)
    gap = g.unix_time.diff().fillna(7 * 86400).clip(0, 30 * 86400).values / 3600
    la = np.log1p(d.amt.values)
    n = g.cumcount().values + 1
    s1 = g['amt'].transform(lambda a: np.log1p(a).cumsum()).values - la
    s2 = g['amt'].transform(lambda a: (np.log1p(a) ** 2).cumsum()).values - la ** 2
    m = np.where(n > 1, s1 / np.maximum(n - 1, 1), la)
    sd = np.sqrt(np.maximum(np.where(n > 2, s2 / np.maximum(n - 1, 1) - m ** 2, 1.0), 0.04))
    codes = d.card.astype('category').cat.codes.values.astype(np.int64)
    key = codes * 10 ** 10 + d.unix_time.values
    tx24 = np.arange(len(d)) - np.searchsorted(key, key - 86400, side='left')
    hour = pd.to_datetime(d.unix_time, unit='s').dt.hour.values
    lat1, lon1, lat2, lon2 = (np.radians(d[c].values) for c in ('lat', 'long', 'merch_lat', 'merch_long'))
    km = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    online = d.category.str.endswith('_net').values
    X = np.stack([la, online, first(['merchant']), first(['category']), np.log1p(km), np.log1p(gap), np.sin(2 * np.pi * hour / 24),
                  np.cos(2 * np.pi * hour / 24), (hour >= 22) | (hour < 4), np.clip((la - m) / sd, -5, 5), np.log1p(tx24),
                  d.category.isin(['shopping_net', 'grocery_pos', 'misc_net']).values], 1).astype(np.float32)
    card = d.card.values
    start = np.r_[0, np.flatnonzero(card[1:] != card[:-1]) + 1].astype(np.int64)
    return {'X': X, 'start': start, 'ts': d.unix_time.values.astype(np.int64), 'flag': d.is_fraud.values.astype(bool),
            'usd': d.amt.values.astype(np.float32), 'kind': online.astype(np.int16), 'kinds': ['in person', 'online'],
            'features': SPARKOV_FEATURES}

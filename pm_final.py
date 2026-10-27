"""
Prediction markets at scale: Polymarket, every trade with its wallet, for the ~3,000 two-outcome markets with
$1M-$30M volume that closed in 2025-26 (polymarket_fetch.py --full). The final protocol of bank_final.py, applied to
four kinds of episode a market maker or the venue cares about.

Targets (--target):
    metaorder  one wallet works a large order in pieces (polymarket_meta.py: one direction, trades at most 10 minutes
               apart, at least 5 trades and $2,000, at least a minute)
    churn      one wallet trades back and forth (polymarket_meta.churn_sessions)
    informed   a wallet episode (one wallet, one direction, trades at most 10 minutes apart, at least $2,000) after which
               the price keeps going its way: the outcome-0 price 24 hours after the episode's last trade (the market's
               final value if it has closed) is at least 10 cents further in the episode's direction
    jump       the hour before a price jump: the outcome-0 price moves at least 15 cents within 30 minutes from a level
               between 5 and 95 cents, and is still at least 10 cents away two hours later
Views:
    anonymous  (default) the 9 one-minute features of polymarket_meta.py: what a market maker sees on any venue
    --wallets  plus 4 features only the venue can compute from wallets: signed flow from wallets first seen in the last
               7 days; the dollar-weighted track record of the wallets trading (mean 24-hour markout, in cents, of their
               earlier episodes that had matured by then); signed flow from proven wallets (3 or more matured episodes,
               mean markout 2 cents or more); and how concentrated the last 10 minutes' volume is in one wallet
Protocol: markets ordered by closing time, 60% train, 15% calibration, 25% test. One stream per episode that starts
after 180 minutes without one, capped at 60 minutes; plus 12 random 240-minute windows without one per market. Every
threshold is set by Eq. 11 on the calibration markets and applied to the test markets. DeepQCD: five seeds and their
ensemble. Also gradient boosting (lagged inputs), logistic, WL-CUSUM (window_dec_stat over 10 minutes) and rules.
Bootstrap intervals over test events and windows. The trained models are saved for the replay (pm_replay.py).

Run:  .venv/bin/python pm_final.py --target informed [--wallets] [--quick]      -> runs/pm/<target>[_wallets].json
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

TARGET = sys.argv[sys.argv.index('--target') + 1] if '--target' in sys.argv else 'informed'
WALLETS = '--wallets' in sys.argv
QUICK = '--quick' in sys.argv
_argv = sys.argv
sys.argv = [sys.argv[0], '--clock', 'minute']
import polymarket_meta as pmm                                       # noqa: E402
import usecases_common as uc                                        # noqa: E402
from original_qcd import Model, fit, scores                         # noqa: E402
from transient_real import projection                              # noqa: E402
sys.argv = _argv

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'polymarket')
OUT = os.path.join(ROOT, 'runs', 'quick' if QUICK else '', 'pm')
PRE, MAXLEN, WARM, WL_K, L_NO = 180, 60, 10, 10, 240
L = PRE + MAXLEN
SEEDS = (0,) if QUICK else (0, 1, 2, 3, 4)
EPOCHS = 2 if QUICK else 20
ALPHAS = (0.01, 0.05, 0.1)
B = 200 if QUICK else 1000
GAP, MIN_USD, INFORMED_C, HORIZON = 600, 2000.0, 10.0, 86400
MAX_WIN = 8000 if QUICK else 40000
SPORTS = re.compile(r'\bvs\.?\b|\bO/U\b|spread|\bwin on 20\d\d|\bNBA\b|\bNFL\b|\bNHL\b|\bMLB\b|\bUFC\b|\bLoL\b|Counter-Strike|'
                    r'Dota|\bBO[1-5]\b|Premier League|La Liga|Serie A|Bundesliga|\bEPL\b|Grand Prix|\bATP\b|\bWTA\b', re.I)
FEATURES = list(pmm.FEATURES) + (['fresh_flow', 'track_record', 'proven_flow', 'concentration'] if WALLETS else [])
TAG = TARGET + ('_wallets' if WALLETS else '')


def sgnlog(f, s=100):
    return np.sign(f) * np.log1p(np.abs(f) / s)


def markets():
    ms = sorted(json.load(open(os.path.join(DATA, 'full_markets.json'))), key=lambda m: m['closed'])
    ms = [m for m in ms if os.path.exists(os.path.join(DATA, 'full', m['cid'] + '.npz'))]
    return ms[::6] if QUICK else ms


def final_value(m):
    try:
        v = float(json.loads(m['outcome_prices'])[0])
    except (TypeError, ValueError, IndexError):
        return None
    return v if v in (0.0, 1.0) else None


# ---------------------------------------------------------------- pass 0: wallet episodes and their markouts
def wallet_table(ms):
    """Every wallet episode (one wallet, one direction, trades at most GAP apart, at least MIN_USD) of every market,
    with its 24-hour markout, and each wallet's first trade in any market. Cached in data/cache/pm_wallets.npz."""
    path = os.path.join(ROOT, 'data', 'cache', 'pm_wallets' + ('_quick' if QUICK else '') + '.npz')
    cids = np.array([m['cid'] for m in ms])
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if np.array_equal(z['cids'], cids):
            return {k: z[k] for k in z.files}
    gid, fs_g, fs_t = {}, [], []
    E = {k: [] for k in ('gid', 'mkt', 't0', 't1', 'side', 'usd', 'markout')}
    for mi_, m in enumerate(ms):
        z = np.load(os.path.join(DATA, 'full', m['cid'] + '.npz'))
        t, w, s, usd, p0 = z['t'], z['w'], z['s'].astype(int), z['usd'].astype(float), z['p0'].astype(float)
        g = np.array([gid.setdefault(a, len(gid)) for a in z['wallets']], np.int64)
        if len(t) == 0:
            continue
        lf = np.full(len(g), np.iinfo(np.int64).max)
        np.minimum.at(lf, w, t)
        fs_g.append(g); fs_t.append(lf)
        fv = final_value(m)
        o = np.lexsort((t, w))
        to, wo, so, uo = t[o], w[o], s[o], usd[o]
        brk = np.r_[True, (wo[1:] != wo[:-1]) | (so[1:] != so[:-1]) | (to[1:] - to[:-1] > GAP)]
        st = np.flatnonzero(brk)
        en = np.r_[st[1:], len(o)] - 1
        amt = np.add.reduceat(uo, st)
        keep = amt >= MIN_USD
        st, en, amt = st[keep], en[keep], amt[keep]
        t_end = to[en]
        p_end = p0[o[en]]
        j = np.searchsorted(t, t_end + HORIZON, side='right') - 1
        p_fut = p0[j]
        late = t_end + HORIZON > t[-1]
        if fv is not None:
            p_fut = np.where(late, fv, p_fut)
        E['gid'].append(g[wo[st]]); E['mkt'].append(np.full(len(st), mi_)); E['t0'].append(to[st]); E['t1'].append(t_end)
        E['side'].append(so[st]); E['usd'].append(amt); E['markout'].append(so[st] * (p_fut - p_end) * 100)
        if mi_ % 500 == 0:
            print(f'  wallet table: {mi_} markets, {len(gid):,} wallets', flush=True)
    out = {k: np.concatenate(v) for k, v in E.items()}
    fs = np.full(len(gid), np.iinfo(np.int64).max)
    np.minimum.at(fs, np.concatenate(fs_g), np.concatenate(fs_t))
    out['first_seen'] = fs
    out['addr'] = np.array(list(gid))
    out['cids'] = cids
    # matured track record: episodes sorted by (wallet, time the markout is known)
    known = out['t1'] + HORIZON
    o = np.lexsort((known, out['gid']))
    out['rk_gid'], out['rk_known'], out['rk_mark'] = out['gid'][o], known[o], out['markout'][o]
    out['rk_cum'] = np.cumsum(out['rk_mark'])
    out['rk_first'] = np.r_[0, np.flatnonzero(out['rk_gid'][1:] != out['rk_gid'][:-1]) + 1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def track(Wt, gids, ts):
    """Mean markout and number of a wallet's matured episodes at times ts."""
    key_g = Wt['rk_gid']
    big = np.int64(1) << 34
    keys = key_g * big + Wt['rk_known']
    j = np.searchsorted(keys, gids * big + ts, side='right') - 1
    ok = (j >= 0) & (key_g[np.maximum(j, 0)] == gids)
    j = np.maximum(j, 0)
    grp_start = Wt['rk_first'][np.searchsorted(Wt['rk_first'], j, side='right') - 1]
    n = np.where(ok, j - grp_start + 1, 0)
    tot = np.where(ok, Wt['rk_cum'][j] - np.r_[0, Wt['rk_cum']][grp_start], 0.0)
    return np.where(n > 0, tot / np.maximum(n, 1), 0.0), n


# ---------------------------------------------------------------- pass 1: one market
def market(m, mi_, Wt, gidmap, targets=None, wallets=None):
    """Minute features of one market (anonymous, plus the wallet features if wallets) and the episodes of each target."""
    targets = targets or [TARGET]
    wallets = WALLETS if wallets is None else wallets
    r = pmm.market(m['cid'])
    if r is None:
        return None
    z = np.load(os.path.join(DATA, 'full', m['cid'] + '.npz'))
    t, w, s, usd, p0 = z['t'], z['w'], z['s'].astype(float), z['usd'].astype(float), z['p0'].astype(float)
    t0 = t[0] // 60
    mi = t // 60 - t0
    n = len(r['x'])
    x = r['x']
    if wallets:
        g = gidmap[mi_][w]
        fresh = (t - Wt['first_seen'][g]) < 7 * 86400
        rep, cnt = track(Wt, g, t)
        f1 = np.bincount(mi, weights=s * usd * fresh, minlength=n)
        gross = np.bincount(mi, weights=usd, minlength=n)
        f2 = np.bincount(mi, weights=s * usd * np.clip(rep, -20, 20), minlength=n) / np.maximum(gross, 1)
        f3 = np.bincount(mi, weights=s * usd * ((cnt >= 3) & (rep >= 2)), minlength=n)
        # concentration: the largest single wallet's share of each minute's volume, averaged over 10 minutes
        o = np.lexsort((w, mi))
        k_ = mi[o].astype(np.int64) * (int(w.max()) + 1) + w[o]
        brk = np.r_[True, k_[1:] != k_[:-1]]
        per = np.add.reduceat(usd[o], np.flatnonzero(brk))
        mins = mi[o][brk]
        top = np.zeros(n)
        np.maximum.at(top, mins, per)
        conc = pmm.roll(top / np.maximum(gross, 1) * (gross > 0), 10) / 10
        x = np.c_[x, sgnlog(f1), f2, sgnlog(f3), conc].astype(np.float32)
    eps = {}
    for target in targets:
        if target == 'metaorder':
            e = [(a, b, sg, u, 0.0) for a, b, sg, u, *_ in r['metas']]
        elif target == 'churn':
            e = [(a, b, 0, u, 0.0) for a, b, _, u, *_ in pmm.churn_sessions(t, w, s, usd, t0)]
        elif target == 'informed':
            sel = (Wt['mkt'] == mi_) & (Wt['markout'] >= INFORMED_C)
            e = [(int(a // 60 - t0), int(b // 60 - t0) + 1, int(sg), float(u), float(mk)) for a, b, sg, u, mk in
                 zip(Wt['t0'][sel], Wt['t1'][sel], Wt['side'][sel], Wt['usd'][sel], Wt['markout'][sel])]
        else:
            pr = r['price'].astype(float)
            e, nxt = [], 60
            if n > 151:
                k = np.arange(n - 150)
                c = (pr[k] >= 0.05) & (pr[k] <= 0.95) & (np.abs(pr[k + 30] - pr[k]) >= 0.15) & (np.abs(pr[k + 150] - pr[k]) >= 0.10)
                for k in np.flatnonzero(c):
                    if k >= nxt:
                        e.append((int(k) - 60, int(k), int(np.sign(pr[k + 30] - pr[k])), 0.0, float(abs(pr[k + 30] - pr[k]) * 100)))
                        nxt = k + 360
        eps[target] = e
    out = {'x': x, 'price': r['price'], 'n': n, 'sports': bool(SPORTS.search(m.get('q') or '')), 'eps_all': eps,
           'trades': (t, w, s, usd, p0, t0)}
    e = eps[targets[0]]
    lab = np.zeros(n, np.float32)
    for a, b, *_ in e:
        lab[max(a, 0):b] = 1
    out['eps'], out['lab'] = e, lab
    return out


def windows_of(mk, rng, train):
    """Training windows (train=True) or test streams of one market."""
    n, x, lab = mk['n'], mk['x'], mk['lab']
    out = []
    if train:
        if n <= L:
            return out
        eps = mk['eps']
        if len(eps) > 30:
            eps = [eps[k] for k in rng.choice(len(eps), 30, replace=False)]
        for a, *_ in eps:
            for _ in range(2):
                st = a - int(rng.integers(30, PRE + 30))
                if 0 <= st and st + L <= n:
                    out.append((x[st:st + L], lab[st:st + L]))
        for st in rng.integers(0, n - L, 12):
            out.append((x[st:st + L], lab[st:st + L]))
        return out
    ev, no = [], []
    for e in mk['eps']:
        a, b = e[0], e[1]
        if a >= PRE and lab[a - PRE:a].max() == 0 and b <= n:
            seg = np.zeros((L, x.shape[1]), np.float32)
            bb = min(b, a + MAXLEN)
            seg[:bb - a + PRE] = x[a - PRE:bb]
            ev.append((seg, PRE, bb - a + PRE, e))
    if n > L_NO:
        busy = np.r_[0, np.cumsum(lab)]
        for st in rng.integers(0, n - L_NO, 48):
            if busy[st + L_NO] - busy[st] == 0:
                seg = np.zeros((L, x.shape[1]), np.float32)
                seg[:L_NO] = x[st:st + L_NO]
                no.append(seg)
                if len(no) == 12:
                    break
    return ev, no


def main():
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get('QCD_THREADS', 4)))
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    ms = markets()
    Wt = wallet_table(ms) if (WALLETS or TARGET == 'informed') else None
    if Wt is not None:
        print(f'wallet table: {len(Wt["gid"]):,} wallet episodes of $2,000+, {np.mean(Wt["markout"] >= INFORMED_C):.1%} informed '
              f'(markout >= {INFORMED_C} c); {len(Wt["first_seen"]):,} wallets ({time.time() - t0:.0f} s)', flush=True)
    gidmap = {}
    if WALLETS:
        gid = {a: i for i, a in enumerate(Wt['addr'])}
        for mi_, m in enumerate(ms):
            gidmap[mi_] = np.array([gid[a] for a in np.load(os.path.join(DATA, 'full', m['cid'] + '.npz'))['wallets']], np.int64)
        del gid
    n_m = len(ms)
    split = np.where(np.arange(n_m) < 0.6 * n_m, 0, np.where(np.arange(n_m) < 0.75 * n_m, 1, 2))
    tr_w, sets = [], {1: {'ev': [], 'no': [], 'meta': []}, 2: {'ev': [], 'no': [], 'meta': []}}
    n_eps = [0, 0, 0]
    for mi_, m in enumerate(ms):
        mk = market(m, mi_, Wt, gidmap)
        if mk is None:
            continue
        sp = split[mi_]
        n_eps[sp] += len(mk['eps'])
        if sp == 0:
            tr_w += windows_of(mk, rng, True)
        else:
            ev, no = windows_of(mk, rng, False)
            for seg, a, b, e in ev:
                sets[sp]['ev'].append((seg, a, b))
                sets[sp]['meta'].append({'mi': mi_, 'side': e[2], 'usd': e[3], 'extra': e[4], 'sports': mk['sports'],
                                         'a': int(e[0]), 'b': int(e[1])})
            sets[sp]['no'] += no
        if mi_ % 500 == 0:
            print(f'  {mi_} / {n_m} markets ({time.time() - t0:.0f} s)', flush=True)
    if len(tr_w) > MAX_WIN:
        keep = rng.choice(len(tr_w), MAX_WIN, replace=False)
        tr_w = [tr_w[k] for k in keep]
    xs = np.stack([a for a, _ in tr_w])
    ys = np.stack([b for _, b in tr_w])
    del tr_w
    nrm_rows = np.concatenate([x for x, y in zip(xs, ys) if y.max() == 0])
    mu_, sd_ = nrm_rows.mean(0), nrm_rows.std(0, ddof=1)
    sd_ = np.where(sd_ > 1e-8, sd_, 1.0)
    del nrm_rows
    norm = lambda a: ((a - mu_) / sd_).astype(np.float32)
    xs_n = norm(xs)
    S = {}
    for sp, name in ((1, 'cal'), (2, 'test')):
        ev, no = sets[sp]['ev'], sets[sp]['no']
        X = np.stack([e[0] for e in ev] + no)
        n_ev = len(ev)
        tau1 = np.r_[[e[1] for e in ev], np.full(len(no), PRE)].astype(int)
        tau2 = np.r_[[e[2] for e in ev], np.full(len(no), PRE)].astype(int)
        lim = np.r_[[e[2] for e in ev], np.full(len(no), L_NO)]
        valid = np.arange(L)[None, :] < lim[:, None]
        tau1 = np.where(np.arange(len(X)) < n_ev, tau1, L_NO)       # no-change windows: false alarms anywhere
        S[name] = dict(X=X, Xn=norm(X), tau1=tau1, tau2=tau2, n_ev=n_ev, valid=valid, meta=sets[sp]['meta'])
    del sets
    print(f'{TAG}: {n_m} markets, episodes train/cal/test {n_eps}; training windows {len(xs):,} ({(ys.max(1) > 0).mean():.0%} with '
          f'an episode); calibration {S["cal"]["n_ev"]} events, test {S["test"]["n_ev"]} events ({time.time() - t0:.0f} s)', flush=True)

    # ---------------------------------------------------------------- detectors
    perm = np.random.default_rng(0).permutation(len(xs))
    nv = len(xs) // 7
    va, tr = perm[:nv], perm[nv:]
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        mdl = Model(xs.shape[2])
        fit(mdl, xs_n[tr], ys[tr], xs_n[va], ys[va], epochs=EPOCHS, seed=seed, log=lambda *a, **k: None)
        mdl.eval()
        models.append(mdl)
        print(f'  DeepQCD seed {seed} trained ({time.time() - t0:.0f} s)', flush=True)
    bst = uc.gbm_fit(xs_n, ys, lags=5, wins=(10, 30), rounds=4000)
    w_lr = uc.logistic_fit(xs_n, ys)
    z, mu = projection([x[y == 0] for x, y in zip(xs_n, ys) if (y == 0).any()], [x[y == 1] for x, y in zip(xs_n, ys) if (y == 1).any()])
    F = {f: i for i, f in enumerate(FEATURES)}
    flow_usd = lambda A: np.sign(A[:, :, F['flow']]) * np.expm1(np.abs(A[:, :, F['flow']])) * 100
    rollw = lambda a, k: np.cumsum(a, 1) - np.c_[np.zeros((len(a), k)), np.cumsum(a, 1)[:, :-k]]
    rules = {'Flow, last 5 min': lambda A: np.abs(rollw(flow_usd(A), 5)), 'Trades, last 10 min': lambda A: A[:, :, F['active10']].astype(float)}
    if TARGET == 'jump':
        rules = {'Price move, last 30 min': lambda A: np.abs(rollw(A[:, :, F['dprice']].astype(float), 30)),
                 'Flow, last 30 min': lambda A: np.abs(rollw(flow_usd(A), 30)), **rules}
    if WALLETS:
        fu = lambda A, f: np.sign(A[:, :, F[f]]) * np.expm1(np.abs(A[:, :, F[f]])) * 100
        rules['Fresh-wallet flow, last 30 min'] = lambda A: np.abs(rollw(fu(A, 'fresh_flow'), 30))
        rules['Proven-wallet flow, last 30 min'] = lambda A: np.abs(rollw(fu(A, 'proven_flow'), 30))

    def detect(Xn, Xr):
        out = {}
        ds = [scores(mdl, Xn, chunk=2000) for mdl in models]
        for k, v in enumerate(ds):
            out[f'DeepQCD s{k}'] = v
        out['DeepQCD'] = np.mean(ds, 0)
        out['Gradient boosting (lags)'] = uc.gbm_score(bst, Xn, 5, (10, 30))
        out['Logistic (this step only)'] = uc.logistic_score(w_lr, Xn)
        zz = z(Xn)
        out['WL-CUSUM'] = np.c_[np.full((len(zz), WL_K), -np.inf), pmm.window_dec_stat_streams(zz[:, WL_K:], zz[:, :WL_K], mu)]
        for k, fn in rules.items():
            out[k] = fn(Xr).astype(float)
        return out

    D = {k: detect(s['Xn'], s['X']) for k, s in S.items()}
    print(f'scored ({time.time() - t0:.0f} s)', flush=True)
    res = {'settings': {'target': TARGET, 'wallets': WALLETS, 'features': FEATURES, 'pre': PRE, 'maxlen': MAXLEN, 'warm': WARM,
                        'wl_k': WL_K, 'seeds': list(SEEDS), 'informed_cents': INFORMED_C, 'min_usd': MIN_USD},
           'counts': {'markets': n_m, 'episodes': {'train': n_eps[0], 'calibration': n_eps[1], 'test': n_eps[2]},
                      'train_windows': int(len(xs)), 'cal_events': S['cal']['n_ev'], 'test_events': S['test']['n_ev'],
                      'test_no': int(len(S['test']['X']) - S['test']['n_ev'])},
           'trees': int(bst.best_iteration), 'detectors': {}}
    st, sc = S['test'], S['cal']
    n_ev, n = st['n_ev'], len(st['X'])
    meta = st['meta']
    sizes = np.array([e['usd'] for e in meta])
    sports = np.array([e['sports'] for e in meta])
    groups = {'sports': sports, 'other': ~sports}
    if TARGET in ('metaorder', 'informed'):
        groups.update({'$2k-5k': sizes < 5e3, '$5k-20k': (sizes >= 5e3) & (sizes < 2e4), '$20k+': sizes >= 2e4})
    r = np.random.default_rng(7)
    bi = [(r.integers(0, n_ev, n_ev), n_ev + r.integers(0, n - n_ev, n - n_ev)) for _ in range(B)]
    hits = {}
    for name, dt in D['test'].items():
        o = {}
        oo, _ = uc.evaluate(dt, st['tau1'], st['tau2'], n_ev, WARM, st['valid'])
        o['oracle'] = {'pd_at': oo['pd_at'], 'pfa': oo['pfa'], 'pd': oo['pd']}
        mc = maxima(D['cal'][name], sc)
        mt = maxima(dt, st)
        o['calibrated'] = {}
        for a in ALPHAS:
            h = eq11(*mc, sc['n_ev'], a)
            hit = (mt[0][:n_ev] < h) & (mt[1][:n_ev] >= h)
            fa = mt[0] >= h
            bs = np.array([[hit[i].mean(), fa[np.r_[i, j]].mean()] for i, j in bi])
            hits[(name, a)] = hit
            o['calibrated'][str(a)] = {'h': h, 'pd': float(hit.mean()), 'pfa': float(fa.mean()), 'pfa_cal': float((mc[0] >= h).mean()),
                                       'pd_ci': np.quantile(bs[:, 0], [0.025, 0.975]).tolist(),
                                       'pfa_ci': np.quantile(bs[:, 1], [0.025, 0.975]).tolist(),
                                       'groups': {g: [float(hit[m_].mean()) if m_.any() else None, int(m_.sum())] for g, m_ in groups.items()}}
        res['detectors'][name] = o
    for name in res['detectors']:
        for a in ALPHAS:
            h1, h2 = hits[(name, a)], hits[('Gradient boosting (lags)', a)]
            diff = np.array([h1[i].mean() - h2[i].mean() for i, _ in bi])
            res['detectors'][name]['calibrated'][str(a)]['diff_vs_gb'] = [float(h1.mean() - h2.mean())] + np.quantile(diff, [0.025, 0.975]).tolist()
    for name in ('DeepQCD', 'Gradient boosting (lags)', 'Logistic (this step only)', 'WL-CUSUM', *rules):
        c = res['detectors'][name]['calibrated']
        print(f'  {name:32s} ' + '  '.join(f'a={a}: PD {c[a]["pd"]:.3f} [{c[a]["pd_ci"][0]:.2f}, {c[a]["pd_ci"][1]:.2f}] PFA {c[a]["pfa"]:.3f}'
                                           for a in c) + f'  | oracle@0.05 {res["detectors"][name]["oracle"]["pd_at"]["0.05"]:.3f}', flush=True)
    # save the models for the replay
    torch.save({'state': [mdl.state_dict() for mdl in models], 'mu': mu_, 'sd': sd_, 'features': FEATURES, 'z_mu': mu,
                'gbm': bst.model_to_string(), 'w_lr': w_lr}, os.path.join(OUT, f'{TAG}_models.pt'))
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(OUT, f'{TAG}.json'), 'w'), default=float)
    print(f'done in {res["seconds"]} s', flush=True)


def maxima(d, s):
    d = np.where(s['valid'], d, -np.inf)
    t = np.arange(d.shape[1])[None, :]
    m_pre = np.where((t >= WARM) & (t < s['tau1'][:, None]), d, -np.inf).max(1)
    m_win = np.where((t >= s['tau1'][:, None]) & (t < s['tau2'][:, None]), d, -np.inf).max(1)
    return m_pre, m_win


def eq11(m_pre, m_win, n_ev, alpha, n_grid=1000):
    fin = np.r_[m_pre[np.isfinite(m_pre)], m_win[np.isfinite(m_win)]]
    h = np.unique(np.r_[np.quantile(fin, np.linspace(0, 1, n_grid)), np.inf])
    pfa = (m_pre[None, :] >= h[:, None]).mean(1)
    pd = ((m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])).mean(1)
    ok = pfa <= alpha
    return float(h[np.flatnonzero(ok)[np.argmax(pd[ok])]]) if ok.any() else float(np.inf)


if __name__ == '__main__':
    main()

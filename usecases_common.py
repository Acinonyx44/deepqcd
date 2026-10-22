"""
Shared pieces of the finance use cases (usecase_*.py): the original DeepQCD network and training, the window-limited
CUSUM on the Gaussian-LLR projection, and the paper's transient metrics on native test streams.

Test streams: X (n, L, P), the change on [tau1, tau2) for change streams (n_ev of them, first), tau1 = tau2 = L + 1
for no-change streams. PFA = P(alarm in [warm, tau1)) over all streams; PD = P(first alarm in [tau1, tau2)) over
change streams (the paper's Eq. 11 picks the threshold with the largest PD subject to PFA <= alpha).
"""
import numpy as np
import torch

from original_qcd import Model, fit, normalization, scores, window_dec_stat_streams
from transient_real import projection

ALPHAS = (0.01, 0.05, 0.1)


def deepqcd(xs_tr, ys_tr, X, seeds=(0, 1, 2), epochs=20, rng=None, chunk=1000):
    """Train the paper's network once per seed on (xs_tr, ys_tr) (already normalized) and score X. Returns the list of
    statistics (n, L) and the models."""
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(xs_tr))
    nv = len(perm) // 7
    va, tr = perm[:nv], perm[nv:]
    out, models = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = Model(xs_tr.shape[2])
        fit(model, xs_tr[tr], ys_tr[tr], xs_tr[va], ys_tr[va], epochs=epochs, seed=seed, log=lambda *a, **k: None)
        model.eval()
        out.append(scores(model, X, chunk=chunk))
        models.append(model)
    return out, models


def wl_cusum(xs_tr, ys_tr, X, K):
    """window_dec_stat over the last K steps on the Gaussian-LLR projection fitted on training steps (label 0 vs 1).
    The first K steps of each stream start the window (statistic -inf there)."""
    pre = [x[y == 0] for x, y in zip(xs_tr, ys_tr) if (y == 0).any()]
    post = [x[y == 1] for x, y in zip(xs_tr, ys_tr) if (y == 1).any()]
    z, mu = projection(pre, post)
    zz = z(X)
    return np.c_[np.full((len(zz), K), -np.inf), window_dec_stat_streams(zz[:, K:], zz[:, :K], mu)]


def logistic_step(xs_tr, ys_tr, X, iters=60, max_rows=2_000_000, seed=0):
    """A learned per-step scorer without memory: logistic regression of the label on the current step's features
    (IRLS on up to 2 million training steps). Inputs normalized. Returns scores (n, L)."""
    Z, y = xs_tr.reshape(-1, xs_tr.shape[2]).astype(np.float64), ys_tr.ravel().astype(np.float64)
    if len(Z) > max_rows:
        k = np.random.default_rng(seed).choice(len(Z), max_rows, replace=False)
        Z, y = Z[k], y[k]
    Z = np.c_[Z, np.ones(len(Z))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ (p - y))
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return (X.reshape(-1, X.shape[2]) @ w[:-1] + w[-1]).reshape(X.shape[:2])


def lagged(A, lags=5, wins=(10, 30)):
    """Per-step inputs with a fixed look-back: x[n], x[n-1] .. x[n-lags], and the mean and the max of each feature over
    the last w steps for each w in wins (zeros before the stream starts). A (n, L, P) -> (n, L, P (1 + lags + 2 len(wins)))."""
    n, L, P = A.shape
    parts = [A] + [np.concatenate([np.zeros((n, k, P), A.dtype), A[:, :L - k]], 1) for k in range(1, lags + 1)]
    c = np.cumsum(A, 1, dtype=np.float64)
    for w in wins:
        s = c.copy()
        s[:, w:] -= c[:, :-w]
        parts.append((s / np.minimum(np.arange(1, L + 1), w)[None, :, None]).astype(np.float32))
        m = A.copy()
        for k in range(1, min(w, L)):
            m[:, k:] = np.maximum(m[:, k:], A[:, :L - k])
        parts.append(m)
    return np.concatenate(parts, 2)


def gbm_lags(xs_tr, ys_tr, X, lags=5, wins=(10, 30), max_rows=2_000_000, seed=0, chunk=500):
    """A strong learner with a fixed look-back instead of memory: gradient-boosted trees (LightGBM) on the lagged inputs,
    trained per step on the training windows (up to max_rows steps; a seventh of the windows for early stopping, as for
    DeepQCD). Inputs normalized. Returns raw scores (n, L)."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(xs_tr))
    nv = len(perm) // 7
    keep = min(1.0, max_rows / ys_tr.size)

    def rows(idx):
        F, y = [], []
        for s in range(0, len(idx), chunk):
            f = lagged(xs_tr[idx[s:s + chunk]], lags, wins)
            f, t = f.reshape(-1, f.shape[2]), ys_tr[idx[s:s + chunk]].ravel()
            m = rng.random(len(t)) < keep
            F.append(f[m]); y.append(t[m])
        return np.concatenate(F), np.concatenate(y)
    (Ft, yt), (Fv, yv) = rows(perm[nv:]), rows(perm[:nv])
    params = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1)
    bst = lgb.train(params, lgb.Dataset(Ft, yt), 2000, valid_sets=[lgb.Dataset(Fv, yv)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f'  gradient boosting: {len(yt)} training steps, {bst.best_iteration} trees', flush=True)
    out = np.empty(X.shape[:2])
    for s in range(0, len(X), chunk):
        f = lagged(X[s:s + chunk], lags, wins)
        out[s:s + chunk] = bst.predict(f.reshape(-1, f.shape[2]), raw_score=True).reshape(f.shape[:2])
    return out


def add_detector(path, name, o):
    """Add one detector's results to a results file written by a full run (so a new baseline needs no rerun of the rest)."""
    import json
    report(name, o)
    res = json.load(open(path))
    res['detectors'][name] = o
    json.dump(res, open(path, 'w'))


def evaluate(d, tau1, tau2, n_ev, warm, valid=None, groups=None, n_grid=1000):
    """Transient metrics of one statistic d (n, L). Returns the PD-PFA curve, PD at each alpha, the threshold, the
    per-stream hits at each alpha (for value measures) and PD per group (dict name -> boolean mask over change
    streams)."""
    L = d.shape[1]
    if valid is not None:
        d = np.where(valid, d, -np.inf)
    t = np.arange(L)[None, :]
    m_pre = np.where((t >= warm) & (t < tau1[:, None]), d, -np.inf).max(1)
    m_win = np.where((t >= tau1[:, None]) & (t < tau2[:, None]), d, -np.inf).max(1)
    fin = np.r_[m_pre[np.isfinite(m_pre)], m_win[np.isfinite(m_win)]]
    h = np.unique(np.r_[np.quantile(fin, np.linspace(0, 1, n_grid)), np.inf])
    pfa = (m_pre[None, :] >= h[:, None]).mean(1)
    hit = (m_pre[None, :n_ev] < h[:, None]) & (m_win[None, :n_ev] >= h[:, None])
    pd = hit.mean(1)
    out = {'pd_at': {}, 'h_at': {}, 'groups': {}}
    hits = {}
    for a in ALPHAS:
        ok = pfa <= a
        j = int(np.flatnonzero(ok)[np.argmax(pd[ok])]) if ok.any() else len(h) - 1
        out['pd_at'][str(a)] = float(pd[j])
        out['h_at'][str(a)] = float(h[j])
        hits[str(a)] = hit[j]
        if groups:
            out['groups'][str(a)] = {g: float(hit[j, m].mean()) for g, m in groups.items() if m.any()}
    o = np.argsort(pfa)
    out['pfa'], out['pd'] = np.round(pfa[o], 4).tolist(), np.round(pd[o], 4).tolist()
    return out, hits


def first_alarm(d_row, h, start, stop):
    a = np.flatnonzero(d_row[start:stop] >= h)
    return int(a[0]) + start if len(a) else -1


def merge_seeds(runs):
    """Average PD over seeds (seed 0's curve and threshold kept), with each seed's PD at each alpha."""
    out = dict(runs[0])
    out['pd_at'] = {a: float(np.mean([r['pd_at'][a] for r in runs])) for a in runs[0]['pd_at']}
    out['pd_at_seeds'] = {a: [r['pd_at'][a] for r in runs] for a in runs[0]['pd_at']}
    if runs[0]['groups']:
        out['groups'] = {a: {g: float(np.mean([r['groups'][a][g] for r in runs])) for g in runs[0]['groups'][a]}
                         for a in runs[0]['groups']}
    out['curves_other_seeds'] = [{'pfa': r['pfa'], 'pd': r['pd']} for r in runs[1:]]
    return out


def report(name, out):
    s = f'{name:34s} PD at PFA ' + '  '.join(f'{a}: {v:.3f}' for a, v in out['pd_at'].items())
    if 'pd_at_seeds' in out:
        s += '  seeds at 0.05: ' + ', '.join(f'{v:.2f}' for v in out['pd_at_seeds']['0.05'])
    if out.get('groups'):
        s += '  | ' + ', '.join(f'{g} {v:.2f}' for g, v in out['groups']['0.05'].items())
    print(s, flush=True)

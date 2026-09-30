"""Checks for qcd.py against slow, obviously-correct reference implementations.
Run:  .venv/bin/python tests/test_qcd.py   (or pytest, if installed)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qcd import (DeepQCD, NetDetector, Recursive, WindowCUSUM, cusum, shiryaev, shiryaev_roberts,  # noqa: E402
                 stopping_times)

rng = np.random.default_rng(0)


def run_chunked(det, x, chunk):
    """Feed x (n, T, P) to a detector in chunks and concatenate the decision statistics."""
    det.reset(len(x))
    return np.concatenate([det(x[:, i:i + chunk]) for i in range(0, x.shape[1], chunk)], axis=1)


def gaussian_llr(x, x_prev):
    return x[..., 0] - 0.5  # N(1,1) vs N(0,1)


def ar_llr(x, x_prev, mu0=0.0, lam0=-0.3, mu1=1.0, lam1=0.2):
    x = x[..., 0]
    xp = np.concatenate([x_prev, x[:, :-1]], axis=1)
    return (x - 0.5 * (xp * (lam0 + lam1) + mu0 + mu1)) * (xp * (lam1 - lam0) + mu1 - mu0)


def test_ar_llr_is_log_density_ratio():
    """Eq. (10) must equal log N(x; mu1 + lam1 x_prev, 1) - log N(x; mu0 + lam0 x_prev, 1)."""
    x = rng.standard_normal((3, 50, 1)).astype(np.float32)
    x_prev = rng.standard_normal((3, 1))
    xp = np.concatenate([x_prev, x[:, :-1, 0]], axis=1)
    ref = -0.5 * (x[..., 0] - (1.0 + 0.2 * xp)) ** 2 + 0.5 * (x[..., 0] - (0.0 - 0.3 * xp)) ** 2
    assert np.allclose(ar_llr(x, x_prev), ref, atol=1e-5)


def test_recursive_chunking_is_invisible():
    """Chunk boundaries must not change any statistic, including for the AR LR that needs x_{t-1}."""
    x = rng.standard_normal((5, 100, 1)).astype(np.float32) + 0.3
    for update in (cusum, shiryaev_roberts, shiryaev(0.01)):
        for llr in (gaussian_llr, ar_llr):
            full = run_chunked(Recursive(update, llr, None), x, 100)
            parts = run_chunked(Recursive(update, llr, None), x, 7)
            assert np.allclose(full, parts, rtol=1e-9, atol=1e-12)


def test_recursive_matches_notebook_loops():
    """The vectorized recursions equal the notebooks' scalar loops on one stream."""
    x = rng.standard_normal((1, 200, 1)).astype(np.float32) + 0.2
    l = gaussian_llr(x.astype(np.float64), None)[0]
    g_c, g_sr, p, rho = 0.0, 0.0, 0.0, 0.01
    ref_c, ref_sr, ref_sh = [], [], []
    for lt in l:
        g_c = max(0, g_c + lt)
        g_sr = (1 + g_sr) * np.exp(lt)
        p_til = p + (1 - p) * rho
        p = p_til * np.exp(lt) / (p_til * np.exp(lt) + 1 - p_til)
        ref_c.append(g_c), ref_sr.append(g_sr), ref_sh.append(p)
    assert np.allclose(run_chunked(Recursive(cusum, gaussian_llr, None), x, 50)[0], ref_c)
    assert np.allclose(run_chunked(Recursive(shiryaev_roberts, gaussian_llr, None), x, 50)[0], ref_sr, rtol=1e-9)
    assert np.allclose(run_chunked(Recursive(shiryaev(rho), gaussian_llr, None), x, 50)[0], ref_sh, rtol=1e-9)


def test_window_cusum_matches_reference():
    """d_t = max over non-empty suffixes of the last K LLRs, computed the slow way; chunking invisible."""
    K = 6
    x = rng.standard_normal((4, 60, 1)).astype(np.float32)
    l = gaussian_llr(x.astype(np.float64), None)
    ext = np.concatenate([np.zeros((4, K - 1)), l], axis=1)  # zero LLRs before t = 1
    ref = np.empty_like(l)
    for t in range(l.shape[1]):
        w = ext[:, t:t + K]
        ref[:, t] = np.max([w[:, k:].sum(1) for k in range(K)], axis=0)
    assert np.allclose(run_chunked(WindowCUSUM(gaussian_llr, K, None), x, 60), ref)
    assert np.allclose(run_chunked(WindowCUSUM(gaussian_llr, K, None), x, 7), ref)
    # for h > 0 it agrees with the notebook's window_dec_stat, which also allows the empty suffix (0)
    ref0 = np.maximum(ref, 0)
    assert np.array_equal(run_chunked(WindowCUSUM(gaussian_llr, K, None), x, 13) >= 0.5, ref0 >= 0.5)


def test_stopping_times_matches_brute_force():
    """First crossing per threshold, across chunk boundaries, inf when never crossed by max_t."""
    n, T, P = 6, 95, 1
    x_all = (rng.standard_normal((n, T, P)) + 0.4).astype(np.float32)
    h = np.array([0.5, 2.0, 5.0, 40.0])
    det = Recursive(cusum, gaussian_llr, h)
    stop, = stopping_times([det], n, lambda t0, L: x_all[:, t0:t0 + L], chunk=17, max_t=T, warn=False)
    d = run_chunked(Recursive(cusum, gaussian_llr, None), x_all, T)
    for i in range(n):
        for j, hj in enumerate(h):
            hit = np.flatnonzero(d[i] >= hj)
            assert stop[i, j] == (hit[0] + 1 if len(hit) else np.inf)


def test_net_detector_chunking_is_invisible():
    torch.manual_seed(0)
    net = DeepQCD(3).eval()
    x = rng.standard_normal((2, 40, 3)).astype(np.float32)
    full = run_chunked(NetDetector(net, None), x, 40)
    parts = run_chunked(NetDetector(net, None), x, 1)
    assert np.allclose(full, parts, atol=1e-6)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')

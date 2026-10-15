"""Checks for original_qcd.py: every vectorized port against a line-for-line copy of the authors' loop.
Run:  .venv/bin/python tests/test_original_qcd.py   (or pytest, if installed)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from original_qcd import (Model, block, experiment1, scores, transient_streams, window_dec_stat,  # noqa: E402
                          window_dec_stat_streams)

rng = np.random.default_rng(0)


def test_window_dec_stat_streams_matches_cell_22():
    K, mu_post, T = 7, 1.3, 40
    init, x = rng.normal(size=(5, K)), rng.normal(size=(5, T)) + 0.4
    g = window_dec_stat_streams(x, init, mu_post)
    for n in range(5):
        obs_window = list(init[n])          # initial observation window (initial state)
        for t in range(T):
            obs_window.append(x[n, t])      # update the state (observation window)
            obs_window = obs_window[1:]
            assert abs(window_dec_stat(obs_window, mu_post) - g[n, t]) < 1e-12


def test_experiment1_matches_cell_19():
    n, tau1, K = 300, 30, 6
    tau2 = tau1 + K
    d = rng.uniform(size=(n, tau2 + 1))
    d[:, tau1 - 1:] += 0.3
    h_vec = np.linspace(0.5, 1.4, 25)
    fa, miss = experiment1(d, np.full(n, tau1), np.full(n, tau2), h_vec)
    num_false_alarm_events, num_missed_events = np.zeros(len(h_vec)), np.zeros(len(h_vec))
    for jj in range(n):
        t, alarm_flag_vec = 0, np.zeros(len(h_vec))
        while t <= tau2:
            t += 1
            dec_stat = d[jj, t - 1]
            if t < tau1:
                num_false_alarm_events += (alarm_flag_vec == 0) * (dec_stat >= h_vec)
                alarm_flag_vec += (alarm_flag_vec == 0) * (dec_stat >= h_vec)
            elif t < tau2:
                alarm_flag_vec += (alarm_flag_vec == 0) * (dec_stat >= h_vec)
            elif t == tau2:
                num_missed_events += (1 - alarm_flag_vec)
    assert np.array_equal(fa, num_false_alarm_events) and np.array_equal(miss, num_missed_events)


def test_transient_streams_labels():
    f0 = lambda L: np.zeros((L, 1))
    f1 = lambda L: np.ones((min(L, 13), 1))  # an "episode" that lasts 13 steps at most
    x, y, taus = transient_streams(200, 300, 0.01, 0.02, f0, f1, rng)
    assert np.array_equal(x[..., 0], y)  # label 1 exactly where the post-change data is
    for (t1, t2), yy in zip(taus, y):
        t = np.arange(300)
        assert np.array_equal(yy, ((t >= t1) & (t < t2)).astype(np.float32))
        assert t1 >= 300 or t2 - t1 <= 13


def test_block_contiguous():
    pool = [np.arange(100)[:, None], np.arange(1000, 1005)[:, None]]
    for _ in range(50):
        b = block(pool, 20, rng)[:, 0]
        assert np.all(np.diff(b) == 1)


def test_streaming_equals_whole_stream():
    m = Model(3).eval()
    x = rng.normal(size=(4, 50, 3)).astype(np.float32)
    whole = scores(m, x)
    state, steps = None, []
    with torch.no_grad():
        for t in range(50):  # model2: one observation at a time, state carried over
            d, state = m(torch.as_tensor(x[:, t:t + 1]), state)
            steps.append(d.numpy())
    assert np.abs(whole - np.concatenate(steps, 1)).max() < 1e-5


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok ', name)

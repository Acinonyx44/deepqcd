"""
Shared machinery for the DeepQCD reproductions (paper Secs. 3-5).

Everything is written in the paper's "generic QCD procedure" form (Fig. 2):

    s_t   = phi(x_t, s_{t-1})      state update
    d_t   = omega(s_t)             decision statistic
    Gamma = inf{t : d_t >= h}      stopping time

A detector object holds s_t for n streams at once and maps a chunk of observations x (n, L, P) to
decision statistics d (n, L), carrying its state across chunks. `stopping_times` feeds every detector
the same streams and records the first threshold crossing for a whole grid of thresholds h in one pass,
which is what makes evaluating thousands of streams take minutes instead of the notebooks' days.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- DeepQCD network (Fig. 4)

class DeepQCD(nn.Module):
    """Recurrent layer = state update phi, dense layers = decision mapping omega.
    No feature-transformation layers: Sec. 5 feeds the raw observations to the LSTM."""

    def __init__(self, input_dim, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())

    def forward(self, x, state=None):
        s, state = self.lstm(x, state)
        return self.head(s).squeeze(-1), state


def train(net, x, y, xv, yv, epochs=20, batch=32, patience=3, lr=1e-3):
    """Alg. 1: minimize the binary cross-entropy between d_t and the labels, early stopping on the
    validation BCE. x: (N, T, P) float32 observations, y: (N, T) float32 labels in {0, 1}."""
    x, y, xv, yv = map(torch.as_tensor, (x, y, xv, yv))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCELoss()
    best, bad, best_state = np.inf, 0, None
    for epoch in range(epochs):
        start = time.time()
        net.train()
        for idx in torch.randperm(len(x)).split(batch):
            loss = bce(net(x[idx])[0], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = bce(net(xv)[0], yv).item()
        print(f'  epoch {epoch + 1:2d}  train loss {loss.item():.4f}  val loss {val:.4f}  ({time.time() - start:.0f}s)')
        if val < best:
            best, bad, best_state = val, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return best


def check_causality(net, x):
    """Feeding a whole stream at once must equal feeding it one step at a time (Alg. 2, real-time
    detection). This is what makes the chunked evaluation below valid. x: (n, T, P) float32."""
    x = torch.as_tensor(x)
    with torch.no_grad():
        full, _ = net(x)
        state, steps = None, []
        for t in range(x.shape[1]):
            d, state = net(x[:, t:t + 1], state)
            steps.append(d)
    err = (full - torch.cat(steps, 1)).abs().max().item()
    print(f'  max |whole-stream - step-by-step| = {err:.1e}')
    assert err < 1e-5
    return err


# ---------------------------------------------------------------- detectors
# Interface: reset(n) starts n fresh streams; __call__(x) with x (n, L, P) float32 returns d (n, L).
# `h` is the ascending grid of thresholds the detector is evaluated at (None if only d_t is wanted).

class NetDetector:
    """DeepQCD: the LSTM state is s_t, the dense head is omega."""

    def __init__(self, net, h):
        self.net, self.h = net, h

    def reset(self, n):
        self.state = None

    @torch.no_grad()
    def __call__(self, x):
        d, self.state = self.net(torch.from_numpy(x), self.state)
        return d.numpy().astype(np.float64)


class Recursive:
    """Classical LR-based procedure (Appendix A): s_t = update(s_{t-1}, llr_t), d_t = s_t, where
    llr_t = log f1(x_t) / f0(x_t). `llr(x, x_prev)` gets the chunk (n, L, P) and the observation just
    before it (n, P) (zeros at t = 1), so conditional LRs of non-IID models (e.g. AR) can be computed.
    An `llr` that needs more state than x_{t-1} (e.g. GARCH variances) can be an object with a reset(n)."""

    def __init__(self, update, llr, h):
        self.update, self.llr, self.h = update, llr, h

    def reset(self, n):
        self.s = np.zeros(n)
        self.x_prev = None
        if hasattr(self.llr, 'reset'):
            self.llr.reset(n)

    def __call__(self, x):
        x = x.astype(np.float64)
        if self.x_prev is None:
            self.x_prev = np.zeros_like(x[:, 0])
        l = self.llr(x, self.x_prev)
        self.x_prev = x[:, -1]
        out = np.empty_like(l)
        with np.errstate(over='ignore'):  # SR statistic overflows to inf long after its alarm; harmless
            for t in range(l.shape[1]):
                self.s = self.update(self.s, l[:, t])
                out[:, t] = self.s
        return out


def shiryaev(rho):
    """Eq. (A.6): posterior P(tau <= t | x_1..x_t) for tau ~ geo(rho); optimal in the Bayesian setting."""
    def update(s, l):
        s_til = s + (1 - s) * rho
        with np.errstate(over='ignore'):
            return 1 / (1 + (1 - s_til) / s_til * np.exp(-l))
    return update


def cusum(s, l):
    """Eq. (A.2): optimal for Lorden's minimax problem (IID)."""
    return np.maximum(0, s + l)


def shiryaev_roberts(s, l):
    """Eq. (A.9): Shiryaev with rho -> 0; asymptotically optimal for Pollak's problem (IID)."""
    return (1 + s) * np.exp(l)


class WindowCUSUM:
    """Window-limited CUSUM of Guepie et al. [28] (Sec. 5.3), for a transient change of known duration K:
    d_t = max_{t-K+1 <= k <= t} sum_{i=k}^{t} llr_i, i.e. the largest suffix sum of the last K LLRs.
    The window starts filled with zero LLRs (the notebook fills it with K pre-change samples; for
    thresholds h > 0 the two only differ during the first K steps)."""

    def __init__(self, llr, K, h):
        self.llr, self.K, self.h = llr, K, h

    def reset(self, n):
        self.tail = np.zeros((n, self.K - 1))  # LLRs of the K-1 observations before the chunk
        self.x_prev = None
        if hasattr(self.llr, 'reset'):
            self.llr.reset(n)

    def __call__(self, x):
        x = x.astype(np.float64)
        if self.x_prev is None:
            self.x_prev = np.zeros_like(x[:, 0])
        l = self.llr(x, self.x_prev)
        self.x_prev = x[:, -1]
        n, L = l.shape
        ext = np.concatenate([self.tail, l], axis=1)
        S = np.concatenate([np.zeros((n, 1)), np.cumsum(ext, axis=1)], axis=1)  # S[j] = sum ext[:j]
        end = np.arange(self.K, self.K + L)  # ext index (exclusive) of each output time
        d = np.full((n, L), -np.inf)
        for m in range(1, self.K + 1):  # suffix of length m
            d = np.maximum(d, S[:, end] - S[:, end - m])
        self.tail = ext[:, ext.shape[1] - (self.K - 1):]
        return d


# ---------------------------------------------------------------- simulation

def stopping_times(detectors, n, sample, chunk=2000, max_t=10 ** 6, warn=True):
    """Run all detectors on the same n streams. `sample(t0, L)` must return the observations
    x_{t0+1}, ..., x_{t0+L} of the n streams, shape (n, L, P) float32; it is called with consecutive
    time ranges, so a stateful generator (e.g. AR) can carry x_{t-1} across calls.
    Returns, per detector, an (n, len(h)) array of stopping times Gamma = inf{t : d_t >= h} for each
    threshold h (inf if no alarm by max_t)."""
    for det in detectors:
        assert np.all(np.diff(det.h) > 0), 'thresholds must be ascending'
        det.reset(n)
    stops = [np.full((n, len(det.h)), np.inf) for det in detectors]
    run_max = [np.full(n, -np.inf) for _ in detectors]
    t0 = 0
    while t0 < max_t and any(np.isinf(s[:, -1]).any() for s in stops):  # h ascending: last one alarms last
        L = min(chunk, max_t - t0)
        x = sample(t0, L)
        for det, stop, rm in zip(detectors, stops, run_max):
            cm = np.maximum(np.maximum.accumulate(det(x), axis=1), rm[:, None])  # running max of d_t
            rm[:] = cm[:, -1]
            for i in np.flatnonzero(np.isinf(stop[:, -1])):
                idx = np.searchsorted(cm[i], det.h)  # first index where running max >= h
                new = np.isinf(stop[i]) & (idx < L)
                stop[i, new] = t0 + idx[new] + 1
        t0 += L
    if warn:
        for det, stop in zip(detectors, stops):
            if np.isinf(stop).any():
                print(f'  warning: {type(det).__name__}: {np.isinf(stop).any(1).sum()} streams never alarmed')
    return stops


def decision_statistics(det, x):
    """d_t of one detector on given streams x (n, T, P), for plotting."""
    det.reset(len(x))
    return det(x)


def bayes_metrics(stop, tau):
    """PFA = P(Gamma < tau) and ADD = E[(Gamma - tau)^+] over all streams (false alarms count as 0 delay,
    as in the notebooks), per threshold. stop: (n, H), tau: (n,)."""
    return (stop < tau[:, None]).mean(0), np.maximum(stop - tau[:, None], 0).mean(0)


def cadd(stop, tau):
    """Conditional ADD for a change at a fixed time tau: E[Gamma - tau | Gamma >= tau], per threshold (Pollak's
    criterion; streams that false-alarm before tau are excluded). Also returns the fraction that alarmed early."""
    ok = stop >= tau
    delay = np.where(ok, stop - tau, 0).sum(0) / np.maximum(ok.sum(0), 1)
    return np.where(ok.any(0), delay, np.nan), 1 - ok.mean(0)


def interp_at(err, val, level):
    """val (e.g. ADD or PD) interpolated at a given false-alarm level (PFA or FAP), in log scale of the level."""
    ok = np.isfinite(err) & (err > 0) & np.isfinite(val)
    order = np.argsort(err[ok])
    return np.interp(np.log(level), np.log(err[ok][order]), val[ok][order], left=np.nan, right=np.nan)

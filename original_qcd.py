"""
The authors' transient-QCD code, ported function by function, for Prof. Wang's directive:

    "Work on transient as the paper said, using real world data.
     Use functions and algorithms strictly from the original QCD code."

Each function below names the notebook cell or script line it comes from. The ports change the framework
(TensorFlow/Keras -> PyTorch, per-experiment Python loops -> NumPy over many experiments at once) and
nothing else; tests/test_original_qcd.py checks every vectorized version against a line-for-line copy of
the original loop.

    N    original/Sec. 5.3 -- TransientQCD/Justification_TransientQCD.ipynb      (cell numbers as in the file)
    IoT  original/Sec. 6.2 -- IoT Attack Detection/SignalDetect_IoT_Botnet.py    (line numbers)

What comes from where:
    window_dec_stat              N cell 6 (verbatim; mu_post was a notebook global)
    window_dec_stat_streams      N cell 22: the window starts as K pre-change samples, then slides
    transient_streams            N cell 7 (training: rho1 = 0.001, rho2 = 0.002) and cell 9 (validation)
    Model                        N cell 11: LSTM(16) -> Dense(10, relu) -> Dense(1, sigmoid), Keras initializers
    fit                          N cells 12-13: Adam (Keras defaults), binary cross-entropy, batch 32, 20 epochs
    experiment1                  N cells 19 and 22: false-alarm and missed-detection counts per threshold
    block                        IoT lines 497-499 / 541-543: a contiguous block at a random offset
    normalization                IoT lines 415-435: standardize by the mean and std of the normal training data
"""
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- N cell 6

def window_dec_stat(obs_window, mu_post):
    result = 0
    for i in range(len(obs_window)):
        result = max(0, result + obs_window[i] - mu_post / 2)
    return result


def window_dec_stat_streams(x, init, mu_post):
    """window_dec_stat at every time t = 1..T of n streams, with the window handling of N cell 22:
    obs_window starts as K pre-change samples (`init`); at each t the new observation is appended and the
    oldest dropped, then g = window_dec_stat(obs_window). x: (n, T), init: (n, K). Returns g: (n, T)."""
    n, T = x.shape
    K = init.shape[1]
    inc = np.concatenate([init, x], 1).astype(np.float64) - mu_post / 2
    g = np.empty((n, T))
    for t in range(T):  # the window at time t + 1 is columns t + 1 .. t + K of [init, x]
        r = np.zeros(n)
        for i in range(K):
            r = np.maximum(0, r + inc[:, t + 1 + i])
        g[:, t] = r
    return g


# ---------------------------------------------------------------- N cells 7 and 9

def transient_streams(n, timesteps, rho1, rho2, f0, f1, rng):
    """tau1 ~ geo(rho1), tau2 = tau1 + geo(rho2); x_t ~ f0 for t < tau1, f1 for tau1 <= t < tau2, f0 for t >= tau2;
    label 1 exactly on [tau1, tau2). Time runs t = 0 .. timesteps - 1 as in the notebook's loop.

    f0(L) returns L pre-change observations (L, d). f1(L) returns at most L post-change observations: for
    synthetic data exactly L, for a real episode fewer if the episode ends first, and then tau2 moves to where
    the episode really ends (the change cannot last longer than it did). The stretch after tau2 is a fresh f0
    call, like the second branch of the notebook's loop."""
    xs, ys, taus = None, np.zeros((n, timesteps), np.float32), np.zeros((n, 2), int)
    t = np.arange(timesteps)
    for i in range(n):
        tau1 = int(rng.geometric(rho1))
        tau2 = tau1 + int(rng.geometric(rho2))
        parts = [f0(min(tau1, timesteps))]
        if tau1 < timesteps:
            post = f1(min(tau2, timesteps) - tau1)
            tau2 = tau1 + len(post)
            parts.append(post)
            if tau2 < timesteps:
                parts.append(f0(timesteps - tau2))
        x = np.concatenate(parts, 0)
        if xs is None:
            xs = np.empty((n, timesteps, x.shape[1]), np.float32)
        xs[i] = x
        ys[i] = (t >= tau1) & (t < tau2)
        taus[i] = tau1, tau2
    return xs, ys, taus


# ---------------------------------------------------------------- N cell 11

class Model(nn.Module):
    """model = Sequential([LSTM(16, return_sequences=True), Dense(10, 'relu'), Dense(1, 'sigmoid')]).
    Feeding a stream one step at a time with the state carried over is model2 (stateful, batch 1) of N cell 15."""

    def __init__(self, input_dim, units=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, units, batch_first=True)
        self.dense1 = nn.Linear(units, 10)
        self.dense2 = nn.Linear(10, 1)
        keras_init(self)

    def forward(self, x, state=None):
        s, state = self.lstm(x, state)
        return torch.sigmoid(self.dense2(torch.relu(self.dense1(s)))).squeeze(-1), state


def keras_init(m):
    """Keras defaults: glorot_uniform kernels, orthogonal recurrent kernel, zero biases with the forget-gate
    bias at 1 (unit_forget_bias). Keras and PyTorch both order the LSTM gates i, f, c/g, o."""
    def glorot(w, fan_in, fan_out):
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(w, -lim, lim)
    H = m.lstm.hidden_size
    with torch.no_grad():
        glorot(m.lstm.weight_ih_l0, m.lstm.input_size, 4 * H)
        rk = torch.empty(H, 4 * H)
        nn.init.orthogonal_(rk)
        m.lstm.weight_hh_l0.copy_(rk.T)
        m.lstm.bias_ih_l0.zero_()
        m.lstm.bias_hh_l0.zero_()
        m.lstm.bias_ih_l0[H:2 * H] = 1.0
        for d in (m.dense1, m.dense2):
            glorot(d.weight, d.in_features, d.out_features)
            d.bias.zero_()


# ---------------------------------------------------------------- N cells 12-13

def keras_bce(d, y):
    eps = 1e-7  # Keras clips predictions to [eps, 1 - eps] before the log
    d = d.clamp(eps, 1 - eps)
    return -(y * torch.log(d) + (1 - y) * torch.log(1 - d)).mean()


def fit(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=20, seed=0, log=print):
    """model.compile(optimizer='adam', loss='binary_crossentropy'); model.fit(..., batch_size=32, epochs=20).
    No early stopping: the weights after the last epoch are the ones used (N cell 16)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-7)
    x, y = torch.as_tensor(x_train), torch.as_tensor(y_train)
    xv, yv = torch.as_tensor(x_val), torch.as_tensor(y_val)
    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        start, tot = time.time(), 0.0
        model.train()
        for idx in torch.randperm(len(x), generator=g).split(batch_size):
            loss = keras_bce(model(x[idx])[0], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = np.mean([keras_bce(model(xv[i:i + 250])[0], yv[i:i + 250]).item() for i in range(0, len(xv), 250)])
        history['loss'].append(tot / len(x))
        history['val_loss'].append(float(val))
        log(f'  epoch {epoch + 1:2d}/{epochs}  loss {tot / len(x):.4f}  val_loss {val:.4f}  ({time.time() - start:.0f}s)')
    return history


@torch.no_grad()
def scores(model, x, chunk=1000):
    """d_t for t = 1..T of each stream, the RNN state reset per stream (model2.reset_states(), N cell 19)."""
    out = [model(torch.as_tensor(x[i:i + chunk], dtype=torch.float32))[0].numpy() for i in range(0, len(x), chunk)]
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------- N cells 19 and 22

def experiment1(d, tau1, tau2, h_vec):
    """The counting of N cells 19/22. t = 1, 2, ...; the first t with d_t >= h raises the alarm flag; an alarm
    at t < tau1 is a false-alarm event; at t == tau2 an experiment with no alarm yet is a missed event (the
    statistic at t = tau2 itself is never compared). d: (n, T) with d[:, t-1] the statistic at time t;
    tau1, tau2: (n,) ints. Returns (num_false_alarm_events, num_missed_events), one entry per threshold."""
    n, T = d.shape
    t = np.arange(1, T + 1)[None, :]
    tau1, tau2 = np.asarray(tau1)[:, None], np.asarray(tau2)[:, None]
    m_pre = np.where(t < tau1, d, -np.inf).max(1)
    m_all = np.where(t < tau2, d, -np.inf).max(1)
    h = np.asarray(h_vec)[:, None]
    return (m_pre[None, :] >= h).sum(1), (m_all[None, :] < h).sum(1)


def pd_pfa(d, tau1, tau2, h_vec):
    """pfa_vec = false alarms / test_size, mdr_vec = misses / test_size, PD = 1 - mdr - pfa (N cells 19, 25)."""
    fa, miss = experiment1(d, tau1, tau2, h_vec)
    pfa, pmd = fa / len(d), miss / len(d)
    return pfa, 1 - pmd - pfa, pmd


# ---------------------------------------------------------------- IoT script

def block(pool, L, rng):
    """IoT 497-499: j = rd.randint(0, data0_len - left_len); data0[j:j + left_len]. `pool` is a list of contiguous
    stretches (one long benign file in the IoT code); the block is drawn uniformly over every position where it
    fits inside one stretch. If no stretch is long enough, consecutive blocks are chained."""
    lens = np.array([len(p) for p in pool])
    ok = np.flatnonzero(lens >= L)
    if len(ok) == 0:
        out, need = [], L
        while need > 0:
            p = pool[rng.integers(len(pool))]
            k = min(need, len(p))
            j = rng.integers(0, len(p) - k + 1)
            out.append(p[j:j + k])
            need -= k
        return np.concatenate(out, 0)
    w = (lens[ok] - L + 1).astype(float)
    i = ok[rng.choice(len(ok), p=w / w.sum())]
    j = rng.integers(0, lens[i] - L + 1)
    return pool[i][j:j + L]


def normalization(normal_train):
    """IoT 415-435: data_mean, data_std of the benign training data; every array is mapped to (x - mean) / std."""
    a = np.concatenate(normal_train, 0)
    mu, sd = a.mean(0), a.std(0, ddof=1)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return lambda x: ((x - mu) / sd).astype(np.float32)

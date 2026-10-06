"""
Real datasets for DeepQCD, all reduced to one shape: *episodes*.

An episode is a stretch of a recording that starts in the normal regime and switches once, at a known index:

    x   (T, P) float32     the observations
    tau int                index of the first post-change observation (x[:tau] normal, x[tau:] changed)

Change-free stretches (`calm`) are kept separately; the harness mixes them into training so the network
also learns to stay quiet, exactly like the streams whose geometric tau falls beyond the horizon in Sec. 5.

Where a recording has several changes, consecutive ones are turned into episodes in one of two ways:

    from_changepoints   segments between change points: episode k runs from change k-1 to change k+1, with
                        its change at change k (TCPD, bee dance, HASC)
    from_intervals      labelled anomaly intervals [s, e): episode = x[previous e : e] with tau = s, and the
                        normal stretches in between become `calm` (SKAB, SMD, HAI, NAB, Yahoo, fish kill)

Each loader also fixes the evaluation geometry: test windows take a pre-change stretch of random length in
[pmin, pmax] (so tau is random, see notes/datasets.md on SKAB) and at most H post-change steps.

All data is expected under data/ (git-ignored); notes/realdata.md says where each set comes from.
"""
import csv
import glob
import gzip
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@dataclass
class RealData:
    name: str
    label: str
    unit: str                      # what one time step is
    train: list                    # [(x (T, P) float32, tau)]
    test: list
    pmin: int                      # shortest pre-change stretch in a window
    pmax: int                      # longest
    H: int                         # post-change horizon
    calm: list = field(default_factory=list)  # change-free (T, P) stretches for training
    notes: str = ''
    synthetic: object = None       # optional: training_set(n, T, tau) -> (x, y) replacing real training data


def _path(*p):
    return os.path.join(ROOT, *p)


def _split(items, frac, seed):
    """Seeded split of a list into (train, test)."""
    idx = np.random.default_rng(seed).permutation(len(items))
    k = int(round(frac * len(items)))
    return [items[i] for i in sorted(idx[:k])], [items[i] for i in sorted(idx[k:])]


def _standardize_by(x, ref):
    mu, sd = ref.mean(0), ref.std(0)
    return ((x - mu) / np.where(sd > 1e-8, sd, 1)).astype(np.float32)


def from_changepoints(x, cps, min_seg=5):
    cps = [c for c in sorted(set(int(c) for c in cps)) if 0 < c < len(x)]
    edges = [0] + cps + [len(x)]
    eps = []
    for k in range(1, len(edges) - 1):
        a, c, b = edges[k - 1], edges[k], edges[k + 1]
        if c - a >= min_seg and b - c >= min_seg:
            eps.append((x[a:b].astype(np.float32), c - a))
    return eps


def from_intervals(x, lab, min_pre=10, min_post=1):
    """lab: 0/1 per step. Returns (episodes, calm stretches)."""
    lab = np.asarray(lab).astype(bool)
    d = np.diff(np.r_[0, lab.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    eps, calm, prev = [], [], 0
    for s, e in zip(starts, ends):
        if s - prev >= min_pre and e - s >= min_post:
            eps.append((x[prev:e].astype(np.float32), s - prev))
        if s > prev:
            calm.append(x[prev:s].astype(np.float32))
        prev = e
    if len(x) > prev:
        calm.append(x[prev:].astype(np.float32))
    return eps, calm


# ---------------------------------------------------------------- industrial / process

def skab():
    files = sorted(glob.glob(_path('SKAB', 'data', 'valve*', '*.csv')) + glob.glob(_path('SKAB', 'data', 'other', '*.csv')))
    runs = []
    for f in files:
        rows = list(csv.reader(open(f), delimiter=';'))
        a = np.array(rows[1:])
        x = a[:, 1:9].astype(np.float64)
        anom = a[:, 9].astype(float) > 0
        eps, _ = from_intervals(x, anom, min_pre=20)
        runs.append(eps[0])  # every SKAB run has exactly one fault segment
    free = np.array(list(csv.reader(open(_path('SKAB', 'data', 'anomaly-free', 'anomaly-free.csv')), delimiter=';'))[1:])
    calm = [free[:, 1:9].astype(np.float32)]
    tr, te = _split(runs, 0.6, 0)
    return RealData('skab', 'SKAB water pump', 's', tr, te, pmin=20, pmax=400, H=120, calm=calm,
                    notes='native runs; tau re-randomized by cropping (31/34 runs change at step 557-578)')


def tep():
    """Braatz TEP. Training: normal run d00 spliced onto each fault's training run (whose 480 samples are all
    post-fault). Testing: the native test runs, fault at sample 160 of 960."""
    load = lambda f: np.loadtxt(_path('tep', f + '.dat'))
    d00 = load('d00').T  # stored transposed (52, 500)
    tr, te = [], []
    for k in range(1, 22):
        tr.append((np.concatenate([d00, load(f'd{k:02d}')]).astype(np.float32), len(d00)))
        te.append((load(f'd{k:02d}_te').astype(np.float32), 160))
    return RealData('tep', 'Tennessee Eastman (21 faults)', '3 min', tr, te, pmin=20, pmax=150, H=200,
                    calm=[d00.astype(np.float32)],
                    notes='train = spliced normal+fault runs, test = native runs (fault at 160); one model for all faults')


def _close_gaps(lab, gap):
    """Fill 0-runs shorter than `gap` between 1-runs (the occupancy label flickers for a minute or two)."""
    lab = np.asarray(lab).astype(bool).copy()
    ones = np.flatnonzero(lab)
    for a, b in zip(ones[:-1], ones[1:]):
        if 1 < b - a <= gap:
            lab[a:b] = True
    return lab


def occupancy(light=True):
    def read(f):
        rows = list(csv.reader(open(_path('occupancy', f))))[1:]
        a = np.array([r[2:] for r in rows], dtype=np.float64)
        x, occ = a[:, :5], a[:, 5] > 0
        if not light:
            x = np.delete(x, 2, axis=1)
        return from_intervals(x, _close_gaps(occ, 15), min_pre=10, min_post=10)
    tr, calm = read('datatraining.txt')
    te = read('datatest.txt')[0] + read('datatest2.txt')[0]
    name = 'occupancy' if light else 'occupancy-nolight'
    label = 'UCI room occupancy' + ('' if light else ' (no light sensor)')
    return RealData(name, label, 'min', tr, te, pmin=10, pmax=300, H=60, calm=calm,
                    notes='native: each arrival (label gaps < 15 min closed, stays >= 10 min)')


def cmapss():
    """C-MAPSS FD001: 100 engines run to failure. There is no labelled fault onset; we use the field's
    piecewise-linear RUL convention (degradation starts 125 cycles before failure)."""
    a = np.loadtxt(_path('cmapss', 'train_FD001.txt'))
    x_all = a[:, 5:]
    keep = x_all.std(0) > 1e-6
    eps = []
    for u in np.unique(a[:, 0]):
        x = x_all[a[:, 0] == u][:, keep]
        tau = len(x) - 125
        if tau >= 15:
            eps.append((x.astype(np.float32), tau))
    tr, te = _split(eps, 0.7, 0)
    return RealData('cmapss', 'C-MAPSS FD001 turbofans', 'cycle', tr, te, pmin=5, pmax=150, H=125,
                    notes='onset by convention: 125 cycles before failure')


# ---------------------------------------------------------------- IT / security

def smd():
    """Server Machine Dataset: 28 machines; each machine is standardized by its own (unlabelled) train file.
    Machines are split between training and testing, so the test asks for cross-machine generalization."""
    names = sorted(os.path.basename(f) for f in glob.glob(_path('OmniAnomaly', 'ServerMachineDataset', 'test', '*.txt')))
    per = {}
    for n in names:
        ref = np.loadtxt(_path('OmniAnomaly', 'ServerMachineDataset', 'train', n), delimiter=',')
        x = _standardize_by(np.loadtxt(_path('OmniAnomaly', 'ServerMachineDataset', 'test', n), delimiter=','), ref)
        lab = np.loadtxt(_path('OmniAnomaly', 'ServerMachineDataset', 'test_label', n), delimiter=',')
        per[n] = from_intervals(np.clip(x, -10, 10), lab, min_pre=30)
    tr_m, te_m = _split(names, 0.5, 0)
    tr = [e for n in tr_m for e in per[n][0]]
    te = [e for n in te_m for e in per[n][0]]
    calm = [c for n in tr_m for c in per[n][1]]
    return RealData('smd', 'SMD server machines', 'min', tr, te, pmin=30, pmax=1000, H=100, calm=calm,
                    notes='14 machines train / 14 test')


def hai():
    """HAI 21.03 (ICS testbed, 1 Hz). Standardized by train1; attacks split at random between train and test."""
    def read(f):
        with gzip.open(_path('hai', f + '.csv.gz'), 'rt') as fh:
            r = csv.reader(fh)
            head = next(r)
            a = np.array([row[1:] for row in r], dtype=np.float64)
        return head[1:], a
    head, ref = read('train1')
    cols = [i for i, h in enumerate(head) if not h.startswith('attack')]
    ref = ref[:, cols]
    keep = ref.std(0) > 1e-6
    eps, calm = [], [_standardize_by(ref, ref)[:, keep]]
    for f in ['test1', 'test2', 'test3', 'test4', 'test5']:
        h, a = read(f)
        x = np.clip(_standardize_by(a[:, cols], ref)[:, keep], -20, 20)
        e, c = from_intervals(x, a[:, h.index('attack')], min_pre=120)
        eps += e
        calm += c
    tr, te = _split(eps, 0.6, 0)
    return RealData('hai', 'HAI 21.03 ICS attacks', 's', tr, te, pmin=60, pmax=600, H=120, calm=calm[:4],
                    notes=f'{keep.sum()} sensors; attacks split at random')


# ---------------------------------------------------------------- benchmarks of annotated series

def _causal_norm(x, frac=0.15):
    """Per-series standardization on the first 15 % (NAB's 'probationary period'), so no labels are used."""
    k = max(10, int(frac * len(x)))
    return _standardize_by(x, x[:k])


def nab():
    labels = json.load(open(_path('NAB', 'labels', 'combined_labels.json')))
    series = []
    for f, ts in sorted(labels.items()):
        if not ts:
            continue
        rows = list(csv.reader(open(_path('NAB', 'data', f))))[1:]
        times = [r[0] for r in rows]
        x = np.clip(_causal_norm(np.array([[float(r[1])] for r in rows])), -20, 20)
        lab = np.zeros(len(x))
        for t in ts:
            i = times.index(t.split('.')[0]) if t.split('.')[0] in times else None
            if i is not None:
                lab[i] = 1
        cps = np.flatnonzero(lab)
        cps = cps[cps > 0.15 * len(x)]  # labels inside the normalization stretch are dropped
        series.append(from_changepoints(x, cps, min_seg=20))
    tr, te = _split(series, 0.6, 0)
    return RealData('nab', 'NAB (58 series)', 'step', [e for s in tr for e in s], [e for s in te for e in s],
                    pmin=50, pmax=1000, H=100, notes='onset = labelled anomaly time; series split 60/40')


def tcpd():
    ann = json.load(open(_path('TCPD', 'annotations.json')))
    series = []
    for f in sorted(glob.glob(_path('TCPD', 'datasets', '*', '*.json'))):
        d = json.load(open(f))
        if d['n_dim'] != 1 or d['name'] not in ann:
            continue
        x = np.array(d['series'][0]['raw'], dtype=float)[:, None]
        x = np.nan_to_num(_causal_norm(x)).clip(-20, 20)
        # consensus: a change point counts if at least 3 of the 5 annotators put one within +-5 steps
        pts = sorted(p for a in ann[d['name']].values() for p in a)
        cps = []
        for p in pts:
            if sum(any(abs(q - p) <= 5 for q in a) for a in ann[d['name']].values()) >= 3 and \
                    (not cps or p - cps[-1] > 5):
                cps.append(p)
        series.append(from_changepoints(x, cps, min_seg=5))
    tr, te = _split(series, 0.6, 0)
    return RealData('tcpd', 'TCPD (univariate, consensus CPs)', 'step', [e for s in tr for e in s],
                    [e for s in te for e in s], pmin=5, pmax=100, H=30,
                    notes='heterogeneous changes (level, trend, variance) across unrelated series')


def _mat(*p):
    import scipy.io
    m = scipy.io.loadmat(_path('klcpd_code', 'data', *p))
    return m['Y'].astype(float), m['L'].ravel() > 0


def beedance():
    seqs = []
    for i in range(1, 7):
        y, l = _mat('beedance', f'beedance-{i}.mat')
        seqs.append(from_changepoints(_standardize_by(y, y), np.flatnonzero(l), min_seg=5))
    return RealData('beedance', 'Bee waggle dance (6 seqs)', 'frame', [e for s in seqs[:4] for e in s],
                    [e for s in seqs[4:] for e in s], pmin=5, pmax=60, H=30, notes='seqs 1-4 train, 5-6 test')


def hasc():
    y, l = _mat('hasc', 'hasc-1.mat')
    half = len(y) // 2
    y = _standardize_by(y, y[:half])
    cps = np.flatnonzero(l)
    return RealData('hasc', 'HASC accelerometer activities', 'sample',
                    from_changepoints(y[:half], cps[cps < half], 20),
                    from_changepoints(y[half:], cps[cps >= half] - half, 20), pmin=50, pmax=500, H=100,
                    notes='one recording, first half train / second half test')


def fishkiller():
    y, l = _mat('fishkiller', 'fishkiller.mat')
    half = len(y) // 2
    y = _standardize_by(y, y[:half])
    tr, calm = from_intervals(y[:half], l[:half], min_pre=50)
    te, _ = from_intervals(y[half:], l[half:], min_pre=50)
    return RealData('fishkiller', 'Dam water level (fish kills)', 'step', tr, te, pmin=50, pmax=1000, H=50,
                    calm=calm, notes='one recording, halves; changes are short oscillation bursts')


def yahoo():
    seqs = []
    for f in sorted(glob.glob(_path('klcpd_code', 'data', 'yahoo', '*.mat'))):
        import scipy.io
        m = scipy.io.loadmat(f)
        y = np.clip(_causal_norm(m['Y'].astype(float)), -20, 20)
        seqs.append(from_intervals(y, m['L'].ravel() > 0, min_pre=20))
    tr, te = _split(seqs, 0.6, 0)
    return RealData('yahoo', 'Yahoo S5 subset (15 series)', 'hour', [e for s in tr for e in s[0]],
                    [e for s in te for e in s[0]], pmin=20, pmax=500, H=20,
                    calm=[c for s in tr for c in s[1]], notes='series split 60/40')


# ---------------------------------------------------------------- power, finance

def pmubage():
    """pmuBAGE synthetic PMU events (20 s at 30 Hz). The event sits at the middle of every window, so we take
    tau = 300 (checked: the mean |dx| peaks at samples 300-304 in every file). Features: P, Q, V, f at the
    first 10 PMUs (40 channels)."""
    eps = []
    for f in sorted(glob.glob(_path('pmubage', '*.npy'))):
        a = np.load(f)[:, :, :10, :]  # events, PQVF, PMU, time
        for ev in a:
            x = ev.reshape(40, -1).T
            eps.append((_standardize_by(x, x[:250]).clip(-50, 50), 300))
    tr, te = _split(eps, 0.6, 0)
    return RealData('pmubage', 'pmuBAGE grid events', '1/30 s', tr, te, pmin=20, pmax=280, H=100,
                    notes='84 frequency + 100 voltage events; tau = 300 by construction')


SP500_EVENTS = ['1962-05-21', '1987-10-14', '1989-10-13', '1997-10-27', '1998-08-04', '2000-04-03',
                '2001-09-17', '2002-07-01', '2007-02-27', '2007-07-24', '2008-09-15', '2010-05-06',
                '2011-08-01', '2015-08-20', '2018-02-02', '2018-10-10']


def sp500():
    """Daily S&P 500 (1950-2019). Returns are divided by their trailing one-year std (lagged a day, so it
    is causal), which makes the calm regime ~ unit variance like the GARCH source. The network is trained
    on the simulated GARCH regime change of deepqcd_vol.py ([r, r^2] features) and tested on dated stress
    episodes; the dates are our choice, not a published label set."""
    import sources
    rows = list(csv.reader(open(_path('finance', 'SP500.csv'))))[1:]
    dates = np.array([r[0] for r in rows])
    p = np.array([float(r[4]) for r in rows])
    r = np.diff(np.log(p))
    dates = dates[1:]
    sd = np.sqrt(np.convolve(r ** 2, np.ones(250) / 250, mode='full')[:len(r)])
    z = r[251:] / sd[250:-1]
    dates = dates[251:]
    x = np.stack([z, z ** 2], 1).astype(np.float32)
    eps = []
    for d in SP500_EVENTS:
        i = int(np.searchsorted(dates, d))
        eps.append((x[i - 300:i + 60], 300))
    src = sources.build('garch-sq')
    return RealData('sp500', 'S&P 500 stress episodes', 'day', [], eps, pmin=20, pmax=250, H=60,
                    synthetic=src, notes='trained on simulated GARCH, tested on 16 dated real episodes')


LOADERS = {
    'skab': skab, 'tep': tep, 'occupancy': occupancy, 'occupancy-nolight': lambda: occupancy(light=False),
    'cmapss': cmapss, 'smd': smd, 'hai': hai, 'nab': nab, 'tcpd': tcpd, 'beedance': beedance, 'hasc': hasc,
    'fishkiller': fishkiller, 'yahoo': yahoo, 'pmubage': pmubage, 'sp500': sp500,
}

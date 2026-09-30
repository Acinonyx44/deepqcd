"""
Run one quickest-change-detection experiment end to end and report what the detector actually did:
pick an observation model, put a change at a chosen time, train (or reuse) the DeepQCD network, calibrate
a threshold to a false-alarm budget, then measure where the alarms land relative to the change.

    .venv/bin/python detect.py --source garch --tau 500 --fap 1000

This is the paper's two-phase procedure with one addition: the threshold is *calibrated* rather than
chosen by hand. The paper sweeps h and plots a curve; in use you have a false-alarm budget instead ("at
most one false alarm per 1000 steps on average"), so the honest way to compare detectors is to give each
one the h that meets the same budget and then look at the delays. That is what happens here, and the
comparison is not meaningful any other way: the statistics have different units, so equal h means nothing.

Phases, and where each one lives:

  1. offline training (Alg. 1)  labelled streams from the source, BCE, early stopping.  Cached in
                                runs/models/<source>.pt; --retrain forces a fresh fit.
  2. calibration                streams with no change at all (tau = inf), to measure the average false
                                alarm period FAP(h) for every h on each detector's grid.
  3. real-time detection (Alg. 2)  fresh streams with the change at --tau; each detector runs one
                                observation at a time and stops the first time d_t >= h.
  4. report                     at the calibrated h: how often it cried wolf before the change, how
                                often and how late it caught the change, and the delay distribution.

Outputs a table and figures/detect_<source>_tau<tau>.png.

Sources: iid, ar, garch, garch-sq (see sources.py). The transient change of paper Sec. 5.3 has two
change-points and is scored by detection probability rather than delay, so it keeps its own script.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import sources
from qcd import DeepQCD, NetDetector, decision_statistics, load_or_train, stopping_times


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='iid', choices=list(sources.SOURCES), help='observation model')
    p.add_argument('--tau', type=int, default=500, help='time of the change in the test streams')
    p.add_argument('--fap', type=float, default=1000, help='false-alarm budget to calibrate each threshold to')
    p.add_argument('--trials', type=int, default=2000, help='test streams')
    p.add_argument('--calib', type=int, default=400, help='no-change streams used for calibration')
    p.add_argument('--max-delay', type=int, default=2000, help='give up on a test stream this long after tau')
    p.add_argument('--max-t', type=int, default=100_000, help='give up on a calibration stream after this')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--train-streams', type=int, default=3200)
    p.add_argument('--retrain', action='store_true', help='ignore any cached weights')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def calibrate(fap, grid, target):
    """The grid point whose measured FAP is closest to the target, in log scale. Returns its index, the
    threshold, the FAP actually achieved, and whether the target was inside the measured range."""
    ok = np.isfinite(fap) & (fap > 0)
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[np.argmin(np.abs(np.log(fap[ok]) - np.log(target)))]
    inside = fap[ok].min() <= target <= fap[ok].max()
    return i, grid[i], fap[i], inside


def main():
    a = parse_args()
    src = sources.build(a.source)
    unit = src.unit
    print(f'Source: {src.label}  ({src.input_dim}-dimensional observations)')

    # ---- phase 1: offline training (Alg. 1) --------------------------------------------------------
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    net = DeepQCD(src.input_dim)
    path = os.path.join('runs', 'models', f'{src.name}.pt')

    def make_data():
        T = src.T_train
        x, y = src.training_set(a.train_streams, T, np.random.geometric(src.rho, a.train_streams))
        xv, yv = src.training_set(a.train_streams // 6, T, np.full(a.train_streams // 6, T // 2))
        return x, y, xv, yv

    print(f'Training phase (tau ~ geo({src.rho}) in the training streams)')
    load_or_train(path, net, make_data, retrain=a.retrain, epochs=a.epochs)

    # Re-seed so the evaluation is the same whether or not training just ran.
    np.random.seed(a.seed + 1)

    names = ['DeepQCD'] + [b[0] for b in src.benchmarks()]
    grids = [sources.prob_grid()] + [b[2] for b in src.benchmarks()]
    factories = [lambda h: NetDetector(net, h)] + [b[1] for b in src.benchmarks()]

    # ---- phase 2: calibration ----------------------------------------------------------------------
    print(f'Calibration phase: {a.calib} streams with no change, to measure FAP(h)')
    stops = stopping_times([f(g) for f, g in zip(factories, grids)], a.calib,
                           src.sampler(a.calib, np.full(a.calib, np.inf)), max_t=a.max_t, warn=False)
    fap = {k: s.mean(0) for k, s in zip(names, stops)}  # inf for thresholds that never alarmed

    # ---- phase 3: real-time detection (Alg. 2) -----------------------------------------------------
    print(f'Detection phase: {a.trials} streams with the change at t = {a.tau}')
    horizon = a.tau + a.max_delay
    stops = stopping_times([f(g) for f, g in zip(factories, grids)], a.trials,
                           src.sampler(a.trials, np.full(a.trials, float(a.tau))), max_t=horizon, warn=False)
    stops = dict(zip(names, stops))

    # ---- phase 4: report ---------------------------------------------------------------------------
    print(f'\nCalibrated to one false alarm per ~{a.fap:g} {unit}; change at t = {a.tau}, '
          f'{a.trials} streams, giving up {a.max_delay} {unit} after the change')
    print(f'{"detector":<24}{"h":>12}{"FAP":>10}{"early":>8}{"caught":>8}{"ADD":>8}{"med":>7}{"p90":>7}')
    op, delays = {}, {}
    for k in names:
        c = calibrate(fap[k], grids[names.index(k)], a.fap)
        if c is None:
            print(f'{k:<24}{"never alarms on its grid":>52}')
            continue
        i, h, achieved, inside = c
        s = stops[k][:, i]
        early = s < a.tau                      # false alarm: stopped before the change
        caught = (s >= a.tau) & np.isfinite(s)  # alarm at or after the change, within the horizon
        d = s[caught] - a.tau
        op[k] = (h, achieved, early.mean(), caught.mean())
        delays[k] = d
        flag = '' if inside else '  (target outside the measured range)'
        print(f'{k:<24}{h:>12.4g}{achieved:>10.0f}{early.mean():>8.1%}{caught.mean():>8.1%}'
              f'{d.mean() if len(d) else np.nan:>8.1f}{np.median(d) if len(d) else np.nan:>7.0f}'
              f'{np.percentile(d, 90) if len(d) else np.nan:>7.0f}{flag}')
    print('  early  = alarmed before the change (false alarm)      ADD = mean delay among the caught')
    print('  caught = alarmed within the horizon, at or after tau   med/p90 = median and 90th percentile delay')

    # ---- one example stream ------------------------------------------------------------------------
    t = np.arange(1, min(horizon, a.tau + 150) + 1)
    x1 = src.sampler(1, np.array([float(a.tau)]))(0, len(t))
    d1 = decision_statistics(NetDetector(net, None), x1)[0]
    h_net = op['DeepQCD'][0] if 'DeepQCD' in op else 0.5
    hit = np.flatnonzero(d1 >= h_net)
    first = t[hit[0]] if len(hit) else None                      # the alarm the detector would actually raise
    after = t[hit[hit >= a.tau - 1][0]] if (hit >= a.tau - 1).any() else None  # first crossing at/after the change

    # ---- figures -----------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a0 = ax[0, 0]
    a0.plot(t, x1[0, :, 0], color='lightgray', lw=0.7, label='observation $x_t$')
    a0.set(xlabel=f't ({unit})', ylabel='$x_t$', title='One stream: the statistic crossing its threshold')
    a1 = a0.twinx()
    a1.plot(t, d1, color='C0', label='DeepQCD $d_t$')
    a1.axhline(h_net, color='C3', ls=':', lw=1.2, label=f'threshold h = {h_net:.4g}')
    a1.set_ylabel('$d_t$')
    lo, hi = a0.get_ylim()
    a0.axvline(a.tau, color='gray', lw=1)
    a0.annotate('change', (a.tau, hi), xytext=(3, -12), textcoords='offset points', fontsize=8)
    if first is not None and first < a.tau:
        # At a realistic false-alarm budget a good fraction of streams cry wolf before the change; when this
        # one does, say so rather than reporting a negative delay, and mark the later true detection too.
        a0.axvline(first, color='C3', lw=1)
        a0.annotate(f'false alarm, {a.tau - first} before the change', (first, lo), xytext=(-5, 12),
                    textcoords='offset points', fontsize=8, color='C3', ha='right')
    if after is not None:
        a0.axvline(after, color='C2', lw=1)
        a0.annotate(f'detection, delay {after - a.tau}', (after, lo), xytext=(4, 12),
                    textcoords='offset points', fontsize=8, color='C2')
    a1.legend(loc='upper left', fontsize=8)

    a0 = ax[0, 1]
    top = max((np.percentile(d, 98) for d in delays.values() if len(d)), default=10)
    bins = np.linspace(0, max(top, 5), 40)
    for k in names:
        if k in delays and len(delays[k]):
            a0.hist(delays[k], bins=bins, histtype='step', lw=1.5, label=k)
    a0.set(xlabel=f'detection delay ({unit})', ylabel='streams',
           title=f'Delay distribution at FAP ~ {a.fap:g}')
    a0.legend(fontsize=8)

    a0 = ax[1, 0]
    for k in names:
        f, s = fap[k], stops[k]
        caught = (s >= a.tau) & np.isfinite(s)
        add = np.where(caught.any(0), np.where(caught, s - a.tau, 0).sum(0) / np.maximum(caught.sum(0), 1), np.nan)
        keep = np.isfinite(f) & (f > 0)
        a0.plot(f[keep], add[keep], label=k)
        if k in op:
            a0.plot(op[k][1], np.interp(op[k][1], f[keep], add[keep]), 'o', ms=6, color=a0.lines[-1].get_color())
    a0.axvline(a.fap, color='gray', ls=':', lw=1)
    a0.set(xscale='log', xlabel=f'FAP ({unit} between false alarms)', ylabel=f'conditional ADD ({unit})',
           title='Speed / false-alarm tradeoff, operating points marked')
    a0.legend(fontsize=8)

    a0 = ax[1, 1]
    grid = np.arange(0, int(max(top, 5)) + 1)
    for k in names:
        if k in delays and len(delays[k]):
            a0.plot(grid, [(delays[k] <= g).mean() * op[k][3] for g in grid], label=k)
    a0.set(xlabel=f'{unit} after the change', ylabel='fraction of streams already alarmed',
           title=f'How fast the change is caught (FAP ~ {a.fap:g})', ylim=(0, 1))
    a0.legend(fontsize=8)

    for row in ax:
        for a0 in row:
            a0.grid(alpha=0.3)
    fig.suptitle(f'{src.label} — change at t = {a.tau}', y=0.995)
    fig.tight_layout()
    out = os.path.join('figures', f'detect_{src.name}_tau{a.tau}.png')
    os.makedirs('figures', exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

# DeepQCD on real data

Every dataset from `notes/datasets.md` that a cloud session can download, run through one protocol
(`deepqcd_real.py`, loaders in `realdata.py`, data from `fetch_data.sh`). Figure: `figures/real.png`. Raw
curves and per-seed numbers: `runs/real/` (git-ignored; `deepqcd_real.py --report` rebuilds the table and figure).

## Protocol in brief

- **Episodes → windows.** Each dataset is reduced to episodes (normal, then changed at a known index). Train
  and test windows take a pre-change stretch of random length in `[pmin, pmax]` and at most `H` post-change
  steps. That makes tau random even where all recordings change at the same index (SKAB, TEP, pmuBAGE).
  A quarter of the training windows are change-free where the dataset has such data.
- **DeepQCD** is the paper's network trained from scratch on each dataset's training episodes: an LSTM
  (16 units, or 32 above 10 inputs) feeding a dense layer and a sigmoid, BCE loss, early stopping on
  held-out episodes. Three seeds are run.
- **Rivals**, fitted on the same training windows:
  - a **CUSUM** on two Gaussians, f0 fitted to pre-change samples and f1 to post-change samples;
  - an **MEWMA chart**, which needs only normal data.
- **Scoring.** On the same test windows, the threshold is swept to trace the trade-off curve:
  - **PFA**: fraction of windows that alarm before the change;
  - **ADD**: mean delay over the windows without a false alarm, with a window that never alarms counting
    its full horizon;
  - **DR**: fraction of those windows that alarm within the horizon.

  Detectors are compared at the operating point for a false-alarm budget PFA ≤ 0.05 / 0.1 / 0.25, never at
  equal thresholds.

## Scorecard (PFA ≤ 0.1)

DR = detection rate within the horizon (higher is better), ADD = delay (lower is better). DeepQCD is the
median of 3 seeds. **Bold** marks the best of the three detectors when the gap is clear.

| dataset | episodes train / test | DeepQCD DR · ADD | CUSUM (fitted) DR · ADD | MEWMA DR · ADD | verdict |
|---|---|---|---|---|---|
| Bee waggle dance | 74 / 43 | **0.80 · 15.1** | 0.24 · 22.7 | 0.30 · 22.2 | **DeepQCD**, by a lot |
| SMD server machines | 155 / 171 | **0.18 · 22.6** | 0.02 · 29.7 | 0.03 · 27.9 | **DeepQCD** (all weak) |
| C-MAPSS turbofans † | 67 / 29 | 1.00 · **45.1** | 1.00 · 49.8 | 1.00 · 51.0 | DeepQCD, modestly |
| TCPD (univariate) | 35 / 24 | 0.27 · 17.8 | 0.27 · 17.8 | 0.27 · 17.5 | tie |
| Yahoo S5 | 32 / 20 | 0.17 · 2.5 | 0.17 · 2.6 | 0.06 · 2.8 | tie (nobody detects) |
| Fish kill | 25 / 15 | 0.00 · 15.8 | 0.01 · 15.6 | 0.00 · 16.3 | nobody detects |
| S&P 500 stress ‡ | GARCH sim / 16 | 0.86 · 15.6 | **0.93 · 8.7** | 0.86 · 14.0 | classical, slightly |
| HAI 21.03 ICS attacks | 30 / 20 | 0.84 · 30.3 | 0.90 · 29.9 | **0.94 · 29.7** | classical, slightly |
| NAB | 63 / 46 | 0.16 · 84.0 | **0.29 · 72.5** | 0.14 · 85.9 | classical |
| HASC activities | 17 / 48 | 0.00 · 89.9 | **0.13 · 83.4** | 0.02 · 88.2 | classical (all weak) |
| Occupancy | 7 / 12 | 0.82 · 5.4 | **1.00 · 1.1** | 1.00 · 14.3 | classical |
| Occupancy, no light sensor | 7 / 12 | 0.09 · 46.0 | **0.40 · 40.5** | 0.23 · 50.3 | classical |
| pmuBAGE grid events | 110 / 74 | 0.73 · 39.9 | 0.88 · 21.2 | **0.91 · 17.4** | classical |
| SKAB water pump | 20 / 14 | 0.09 · 116.7 | 0.21 · 99.7 | **0.28 · 93.1** | classical |
| Tennessee Eastman | 42 / 21 | 0.66 · 103.3 | 0.76 · 72.6 | **0.87 · 50.5** | classical, clearly |

† The fault onset is a convention: 125 cycles before failure. ‡ The network is trained on the simulated GARCH
change, and the 16 event dates are our choice. Full numbers at all three PFA levels are in the run log
(`deepqcd_real.py --report`).

**Overall: 3 wins, 3 ties (two of them "nobody detects"), 9 losses.** Trained from scratch on a real
dataset, DeepQCD rarely beats a well-fitted classical chart. The MEWMA chart in particular is a hard
baseline, and it needs no abnormal data at all.

## What separates the wins from the losses

- **Wins come from changes that are not a level shift.**
  - Bee dance: the dance phases differ in the *pattern* of motion (turning direction, waggle oscillation).
  - SMD: incidents show up as co-movements across 38 server metrics.
  - C-MAPSS: slow multivariate drift.

  A Gaussian mean/covariance model sees little of any of these, while the LSTM can learn them. In each case
  the training set also has many episodes of the same kind of change (67-155).
- **Losses are mostly step changes**: pmuBAGE events, most Tennessee Eastman faults, SKAB valve faults, the
  occupancy CO₂/light jump. Those are exactly what MEWMA and CUSUM are built for. DeepQCD has to learn the
  same thing from 7-110 examples and does it less sharply.
- **Little or one-sided normal data hurts DeepQCD most.**
  - Occupancy has 7 training arrivals.
  - SKAB has 20 runs and one anomaly-free file.
  - Tennessee Eastman originally had a single 500-sample normal run. The network memorized it and read the
    test runs' normal stretch as abnormal: its median output there was 0.15 against 0.009 on the training
    normal data, and detection at PFA ≤ 0.1 was 5 %. Adding the dataset's second normal run (`d00_te`, in
    no test episode) brought it to 66 %. All numbers above use both runs, for every detector.

  A model that only needs normal data (MEWMA) is far less exposed to this.
- **Transient labels do not fit the persistent-change setting.** Yahoo's labels are mostly single-point
  outliers and fish kills last ~10-20 steps. None of the three detectors gets above ~17 % detection there.
  These sets measure outlier detection, not quickest change detection.
- **Real volatility (S&P 500).** A network trained only on simulated GARCH is in the same range as the
  classical detectors on 16 real stress episodes. At PFA ≤ 0.05 it detects 79 %, against 63 % for the
  fitted CUSUM and 81 % for MEWMA. At PFA ≤ 0.1 the CUSUM is faster. The tie mirrors the synthetic finding
  in `notes/results.md`: once false alarms are matched, the detectors land close together.
- **Seed spread is large on small datasets.** ADD at PFA ≤ 0.1 ranges over 37-50 cycles (C-MAPSS) and
  15-22 days (S&P 500) across the three seeds. With 12-48 test episodes, differences of a few points in DR
  are within noise. The clear verdicts are the large gaps: bee dance, SMD, Tennessee Eastman, pmuBAGE and
  SKAB.

## Caveats

- One train/test split per dataset (seeded, by recording, entity or episode); only DeepQCD's training seed
  varies.
- The PFA-matched operating point is chosen on the test windows themselves. That is an oracle calibration,
  but it is the same for all detectors.
- There was no per-dataset tuning of DeepQCD (window length, hidden size, features), and no feature
  engineering. The paper's own real-data experiments use pretrained I3D video features and engineered
  network-flow statistics. Raw sensor channels are a harder test.
- The CUSUM is fitted to post-change data from the training episodes, as DeepQCD is, so both rely on the
  test changes resembling the training changes. MEWMA does not.

## Not run, and why

- **Hosts unreachable from a cloud session** (UCI, PhysioNet, Kaggle, Harvard Dataverse, Stanford, S3,
  thedatum.org): N-BaIoT (the paper's own IoT set), CICIoT2023, CIC-IDS2017, STEAD, CHB-MIT, MIT-BIH AF,
  Tennessee Eastman (Rieth, 500 runs per fault), SMAP/MSL, UCR anomaly archive, TSB-AD.
- **Access by request:** SWaT and WADI.
- **Need a laptop download:** REDD/UK-DALE, GOES flares, FEMTO/CWRU bearings, FI-2010/LOBSTER.
- **Reachable but not run:**
  - JHU COVID-19 (no onset labels to score against);
  - VIX (redundant with the S&P 500 returns);
  - GutenTAG (synthetic);
  - PSML.

The highest-value next run is the Rieth Tennessee Eastman set on a laptop. With 500 independent runs per
fault instead of 1, it tests directly whether DeepQCD's loss there is a data-size problem.

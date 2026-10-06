# DeepQCD on real data: where it works, where it doesn't

There are 21 real-data problems: 15 benchmark datasets from `notes/datasets.md` and 6 new applications,
each run through one protocol (`deepqcd_real.py`, loaders in `realdata.py`, data from `fetch_data.sh`).
Figure: `figures/real.png`. Raw curves and per-seed numbers: `runs/real/` (git-ignored). Rebuild everything
with `deepqcd_real.py --report`.

## Protocol in brief

- **Episodes → windows.** Each dataset is reduced to episodes: normal, then changed at a known index.
  Train and test windows take a pre-change stretch of random length in `[pmin, pmax]` and at most `H`
  post-change steps, so tau is random even where all recordings change at the same index. A quarter of
  the training windows are change-free where the dataset has such data.
- **DeepQCD** is the paper's network trained from scratch on each dataset: an LSTM (16 units, 32 above
  10 inputs) feeding a dense layer and a sigmoid, BCE loss, early stopping on held-out episodes. Three
  seeds are run.
- **DeepQCD-hybrid** (ours, `--hybrid`) is the same network with extra inputs: the observations
  re-referenced to the stream's own first 20 steps, the fitted log-LR and the MEWMA statistic. The
  network starts from what the classical charts know.
- **Classical rivals**, fitted on the same training windows:
  - a **CUSUM** on two fitted Gaussians;
  - a **MEWMA** chart (λ = 0.1);
  - a **Shewhart** T² chart (λ = 1, for abrupt changes);
  - where a field has one, its own standard rule: **STA/LTA** for seismic, the **freeze index** for gait, a
    **self-calibrating chart** for keystrokes.

  "Best classical" below is whichever of these does best on that dataset, so the bar is high.
- **Scoring.** On the same test windows the threshold is swept, and each detector is read at the operating
  point for a false-alarm budget PFA ≤ 0.1:
  - **DR**: fraction of the other windows that alarm within the horizon;
  - **ADD**: mean delay, with a miss counted as the full horizon.
- **Verdict rule** (applied mechanically): WIN if DR is at least 5 points higher, or DR is within 5 points
  and ADD at least 10 % lower; *loss* is the mirror image; *none* if every detector catches under 10 %.

## Scorecard at PFA ≤ 0.1 (DR · ADD; DeepQCD = median of 3 seeds)

### Benchmark datasets

| dataset | train eps | DeepQCD | | DeepQCD-hybrid | | best classical |
|---|---|---|---|---|---|---|
| Bee waggle dance | 74 | 0.80 · 15.1 | **WIN** | 0.78 · 14.6 | **WIN** | MEWMA 0.30 · 22.2 |
| SMD server machines | 155 | 0.18 · 22.6 | **WIN** | 0.17 · 22.5 | **WIN** | Shewhart 0.10 · 26.4 |
| C-MAPSS turbofans † | 67 | 1.00 · 45.1 | tie | 1.00 · 34.6 | **WIN** | CUSUM 1.00 · 49.8 |
| Room occupancy | 7 | 0.82 · 5.4 | loss | 1.00 · 0.5 | **WIN** | CUSUM 1.00 · 1.1 |
| Occupancy, no light sensor | 7 | 0.09 · 46.0 | loss | 0.44 · 37.1 | tie | CUSUM 0.40 · 40.5 |
| HAI ICS attacks | 30 | 0.84 · 30.3 | loss | 0.94 · 28.9 | tie | MEWMA 0.94 · 29.7 |
| TCPD | 35 | 0.27 · 17.8 | tie | 0.27 · 17.7 | tie | MEWMA 0.27 · 17.5 |
| Yahoo S5 | 32 | 0.17 · 2.5 | tie | 0.15 · 2.5 | tie | Shewhart 0.17 · 2.4 |
| Fish kill | 25 | 0.00 · 15.8 | none | 0.00 · 16.1 | none | CUSUM 0.01 · 15.6 |
| Tennessee Eastman | 42 | 0.66 · 103.3 | loss | 0.88 · 64.6 | loss | Shewhart 0.90 · 37.2 |
| pmuBAGE grid events | 110 | 0.73 · 39.9 | loss | 0.85 · 29.3 | loss | MEWMA 0.91 · 17.4 |
| SKAB water pump | 20 | 0.09 · 116.7 | loss | 0.20 · 101.1 | loss | MEWMA 0.28 · 93.1 |
| NAB | 63 | 0.16 · 84.0 | loss | 0.11 · 89.7 | loss | CUSUM 0.29 · 72.5 |
| HASC activities | 17 | 0.00 · 89.9 | loss | 0.00 · 89.8 | loss | CUSUM 0.13 · 83.4 |
| S&P 500 stress ‡ | GARCH sim | 0.86 · 15.6 | loss | 0.73 · 22.9 | loss | CUSUM 0.93 · 8.7 |

### New applications

| application | train eps | DeepQCD | | DeepQCD-hybrid | | best classical |
|---|---|---|---|---|---|---|
| **Crypto pump-and-dump** (5 s chunks) | 190 | 0.91 · 1.3 | **WIN** | 0.94 · 0.9 | **WIN** | CUSUM 0.53 · 5.7 |
| IoT Mirai botnet, paper protocol (packets) | 300 | 1.00 · 0.2 | **WIN** | 1.00 · 0.2 | **WIN** | CUSUM 1.00 · 1.1 |
| IoT Mirai botnet, temporal blocks (packets) | 300 | 1.00 · 1.4 | **WIN** | 1.00 · 1.3 | **WIN** | CUSUM 1.00 · 1.7 |
| Account takeover, keystrokes (entries) | 600 | 0.43 · 29.2 | loss | 0.91 · 8.0 | loss | self-calibrating 0.97 · 5.7 |
| Freezing of gait (1/32 s) | 151 | 0.18 · 101.5 | loss | 0.26 · 95.2 | tie | CUSUM 0.28 · 98.1 |
| Earthquake P-wave onset (10 ms) | 92 | 0.27 · 235.1 | loss | 0.46 · 191.1 | loss | Shewhart 0.93 · 46.8 |

† The fault onset is a convention: 125 cycles before failure. ‡ The network is trained on simulated GARCH
only, and the 16 event dates are our choice.

**Tally over 21 problems:**

| | WIN | tie | none | loss |
|---|---|---|---|---|
| DeepQCD | 5 | 3 | 1 | 12 |
| DeepQCD-hybrid | 7 | 5 | 1 | 8 |

The hybrid is never much worse than plain DeepQCD, with one exception: S&P 500, where its classical inputs
come from synthetic training data. It turns occupancy and C-MAPSS into wins and Tennessee Eastman,
keystrokes and freezing of gait into near-ties.

## Where DeepQCD works

1. **The change is a *pattern*, not a level.**
   - Bee-dance phases differ in motion pattern: 0.80 vs 0.30 detected.
   - SMD incidents are co-movements across 38 server metrics.
   - Pump-and-dumps have a characteristic joint signature of rush orders, trades and volume.

   A Gaussian mean/covariance chart sees little of this; an LSTM can learn it.
2. **Normal data is heavy-tailed or bursty, so Gaussian charts drown in false alarms.** Pump-and-dump is
   the clearest case. Ordinary trading has spikes, and every Gaussian chart (CUSUM, MEWMA, Shewhart at any
   λ we tried) needs a high threshold to stay under PFA 0.1, so it catches only ~50 %. DeepQCD learned
   what a *pump* spike looks like versus an ordinary one and catches 91-94 % within ~1 chunk (≈5-7 s
   against ≈28 s).
3. **There are many training episodes of the same kind of change** (74-300 in every win).

## Where it does not

1. **Abrupt step changes.** The Shewhart chart is a near-optimal detector for those, and DeepQCD has to
   learn the same thing from a few dozen examples, less sharply:
   - seismic P onset: Shewhart 0.93, STA/LTA 0.81, DeepQCD 0.27;
   - Tennessee Eastman: 0.90 vs 0.66;
   - pmuBAGE events;
   - SKAB valves.
2. **Little training data**: occupancy (7 arrivals), SKAB (20), HASC (17), plus one-sided normal data.
   Tennessee Eastman with its single 500-sample normal run collapsed to 5 % detection until a second normal
   run was added.
3. **Each stream has its own normal** (keystrokes: every typist differs; test typists are unseen). A
   chart that simply calibrates on the first 15 entries of the session catches 97 % within ~6 entries.
   Plain DeepQCD manages 43 %; the hybrid, given the self-referenced input, reaches 91 %.
4. **Labels that are not lasting changes** (fish kill, Yahoo: 1-20-step bursts). Nobody detects these;
   they are outlier-detection data, not quickest change detection.

## Versus the paper's own applications

The paper reports two real applications (Sec. 6):
- **Video:** UCF-Crime road accidents, I3D features, against a frame-prediction detector.
- **IoT:** N-BaIoT BASHLITE spam on a thermostat, 115 features. Normal and attack samples are drawn
  *independently* (an IID splice) with the change at t = 1, against PCA-CUSUM, QuantTree and others.

Neither UCF-Crime nor N-BaIoT is reachable from a cloud session. The Kitsune Mirai capture has the same
115-feature N-BaIoT representation on a real infection, so we ran it both with the paper's IID-splice
protocol and with contiguous time blocks:

- **The paper's IoT result reproduces in direction, but the problem is easy.** At PFA ≤ 0.1, DeepQCD alarms
  0.2 packets after the attack starts against 1.1 for the best chart (IID splice), and 1.4 vs 1.7 with real
  temporal blocks. Every detector catches 100 %: the attack shifts the features by 8-16 standard deviations.
  DeepQCD's edge is about one packet, and most of it vanishes once the stream keeps its real time
  structure. The large gaps in the paper's Fig. 15 come from weaker baselines and the t = 1 protocol, which
  our synthetic experiments showed flatters recurrent detectors (`notes/results.md`).
- **The applications where DeepQCD beats the classical state of the art by more than in the paper's IoT
  case:**
  - **Pump-and-dump:** +38 points detection and 4× faster.
  - **Bee dance:** +50 points.
  - **SMD:** +8 points and ~15 % faster, at low absolute detection.
  - **C-MAPSS** (hybrid): 31 % faster.
  - **Occupancy** (hybrid): 2× faster.

  These are the cases with a learnable non-Gaussian signature and enough episodes, exactly the profile
  above.

## Caveats

- One train/test split per dataset; only the networks' seeds vary. Seed spread is large on small sets: at
  PFA ≤ 0.1, ADD ranges over 37-50 cycles on C-MAPSS and 27-30 entries on keystrokes. With 12-48 test
  episodes, differences of a few points are noise; the clear verdicts are the large gaps.
- The Mirai test windows are cut from one capture, the second half of its normal and attack stretches. The
  150 test episodes are therefore not independent infections.
- The PFA-matched operating point is chosen on the test windows (an oracle calibration), identically for
  every detector.
- No per-dataset tuning of the networks, and raw channels as inputs. The paper's real-data experiments
  used engineered or pretrained features.
- C-MAPSS onsets and S&P 500 dates are conventions, not ground truth.

## Not run, and why

- **Hosts unreachable from a cloud session** (UCI, PhysioNet, Kaggle, Harvard Dataverse, Stanford, S3,
  Hugging Face, thedatum.org):
  - N-BaIoT and UCF-Crime (the paper's own data);
  - CICIoT2023 and CIC-IDS2017;
  - STEAD (we used PhaseNet's 154 traces instead) and CHB-MIT;
  - MIT-BIH AF;
  - Tennessee Eastman (Rieth, 500 runs per fault);
  - SMAP/MSL, UCR, TSB-AD;
  - the original Daphnet files (we used a GitHub-hosted .mat of the same data).
- **Access by request:** SWaT and WADI.
- **Need a laptop download:** REDD/UK-DALE, GOES flares, FEMTO/CWRU bearings, FI-2010/LOBSTER.
- **Reachable but not used:**
  - the comsyssec "mirai" set (synthetic localhost traffic, interleaved packets);
  - JHU COVID-19 (no onset labels);
  - VIX (redundant with the S&P 500 returns);
  - GutenTAG (synthetic).

The most informative next runs need a laptop:
1. **N-BaIoT itself**, under both protocols, to settle the paper's IoT claim on its own data.
2. **The 500-run Tennessee Eastman set**, to test whether more data closes the gap on step-like faults.
3. **STEAD**, to give the seismic network ~10⁵ training traces instead of 92.

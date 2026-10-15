# Transient QCD on real data with the authors' code

Prof. Wang's directive (October 2026):

> Work on transient as the paper said, using real world data. Use functions and algorithms strictly from the
> original QCD code.

This note records how the directive was turned into experiments, what was taken from the authors' code, the one
thing that had to be added, and the results. The interactive report shows the same results with toggles.

## What comes from the original code

`original_qcd.py` holds the authors' functions, ported from TensorFlow/Keras to PyTorch/NumPy without changing
what they compute. `tests/test_original_qcd.py` runs each vectorized port and a line-for-line copy of the
original loop on the same inputs and checks that the outputs are equal.

| Ingredient | Original code | Port |
|---|---|---|
| transient model, labels 1 on [tau1, tau2) | Sec. 5.3 notebook, cells 7 and 9 | `transient_streams` |
| network LSTM(16) -> Dense(10, relu) -> Dense(1, sigmoid) | cell 11 | `Model` (Keras initializers) |
| training: Adam 1e-3, BCE, batch 32, 20 epochs, no early stopping | cells 12-13 | `fit` |
| online detection, state reset per stream | cells 15-16 | `scores` |
| window-limited CUSUM | cells 6 and 22 | `window_dec_stat` (verbatim), `window_dec_stat_streams` |
| PFA, PMD, PD per threshold | cells 19, 22, 25 | `experiment1`, `pd_pfa` |
| real recordings -> streams (contiguous blocks at random offsets) | IoT script, `random_sample`, `random_sample_eva` (lines 467-548) | `block` |
| standardize by the normal training data | IoT script, `do_normalization` (lines 415-435) | `normalization` |

## The one addition

`window_dec_stat` accumulates `x - mu_post/2`, which is the log-likelihood ratio of N(mu_post, 1) against N(0, 1)
divided by mu_post. It is defined for a scalar observation with unit variance. The real datasets are
multivariate, so each observation is mapped to

    z = w'(x - m0) / sqrt(w' C0 w),   w = C0^{-1} (m1 - m0),

where m0 and C0 are fitted on normal training data and m1 on training events. Under Gaussian models with a
common covariance, the log-likelihood ratio of x is a linear function of z, and z is N(0, 1) before the change.
`mu_post` is the mean of z over the training events, and `window_dec_stat` is then called unchanged. Only the
benchmark uses this mapping. As a control, a second DeepQCD is trained on z alone, so the two detectors can also
be compared on identical inputs.

## Which datasets qualify

The transient model f0 -> f1 -> f0 and the paper's test (tau1 = 1000, K = 25) require:

- **C1** labels that mark intervals which start and end, with normal data after them;
- **C2** most events lasting at least K = 25 steps;
- **C3** at least five test events preceded by tau1 - 1 + K = 1024 or more normal steps (so tau1 = 1000 can be
  tested without splicing);
- **C4** at least 5 training and 5 test events lasting at least K.

`python transient_real.py --survey` checks every labelled dataset in `realdata.py` and writes
`runs/transient_real/survey.json`. The datasets used are:

- HAI and Daphnet freezing of gait, which meet all four conditions;
- SMD, which fails C2 (most anomalies last under 25 minutes) and is tested on its 60 test events that last at
  least K;
- Occupancy, with and without the light sensor (the authors ship a loader for it), and SKAB, which fail C3.

Change-point, persistent-change and single-instant datasets fail C1. Fish kill and Yahoo fail C2.

## Protocol

**Training:** notebook cell 7 unchanged.
- 3200 streams of 3000 steps, with tau1 ~ geo(0.001) and tau2 - tau1 ~ geo(0.002).
- t < tau1 is a contiguous block of normal training data, [tau1, tau2) a real training event from its labelled
  onset, and t >= tau2 another normal block.
- tau2 is moved to the event's real end when the geometric draw runs past it.
- Validation follows cell 9: 500 streams with the rates swapped.

**Testing:** cells 19 and 22, with tau1 = 1000 and K = 25 (paper), plus tau1 = 200 and K = 10 / 50. There are
two constructions:
- *blocks*, the authors' IoT evaluation construction: 10,000 experiments, each a normal test block followed by
  the first K steps of a real test event;
- *native*, with no splicing: each test event preceded by its own recorded normal stretch.

**Seeds:** DeepQCD on all sensors is trained with three seeds and DeepQCD on z with one.

Reproduce with `runs/run_jobs.sh` and `runs/run_jobs2.sh` (job lists in `runs/jobs.txt` and `runs/jobs2.txt`).

## Results

Interactive version: `python report/build_report.py` builds `report/index.html` from the run outputs. It has
toggles for every setting below and the paper's Figs. 5-9 at several parameters (Part I, `paper_sweep.py`).

### Part I in one paragraph

The authors' saved notebook outputs fall on our curves.
- **Bayesian setting:** DeepQCD's ADD at PFA = 0.01 is within 6% of the optimal Shiryaev procedure (or its modified
  AR version) in six of seven settings, and 16% above it for the weak AR drift mu1 = 0.5.
- **Minimax setting, change at tau = 1 (the paper's protocol):** DeepQCD is far faster than CUSUM (ADD 0.04 vs 1.17
  at FAP 1000, IID paper setting).
- **Minimax setting, change at tau = 200:** DeepQCD is never more than 1% faster than CUSUM. It is 2-15% slower at
  FAP 1000, about tied at FAP 10,000, and far slower at FAP 100.
- **Transient, Fig. 9:** reproduces at the paper's setting (PD 0.87 vs 0.86 at PFA 0.05). The window-limited CUSUM
  leads for mu = 0.5 (0.17 vs 0.12) and for K = 10.
- **Notebook bug:** the minimax notebook as released copies one noise draw to all seven coordinates of the benchmark
  data (cells 21, 23, 28, 30). The saved outputs match the correct generator, so the published figure is unaffected.

### Part II tables

**blocks, tau1 = 1000, K = 25** (max PD with PFA <= alpha; DeepQCD all sensors, mean of seeds / DeepQCD on z / WL-CUSUM)

| dataset | n | alpha = 0.01 | alpha = 0.05 | alpha = 0.10 |
|---|---|---|---|---|
| hai | 10000 | 0.06 / 0.30 / 0.26 | 0.43 / 0.35 / 0.51 | 0.57 / 0.36 / 0.51 |
| fog | 10000 | 0.04 / 0.00 / 0.00 | 0.06 / 0.02 / 0.00 | 0.06 / 0.04 / 0.00 |
| smd | 10000 | 0.00 / 0.15 / 0.15 | 0.03 / 0.21 / 0.26 | 0.05 / 0.26 / 0.30 |
| occupancy | 10000 | 0.93 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 |
| occupancy-nolight | 10000 | 0.47 / 0.41 / 0.20 | 0.47 / 0.41 / 0.20 | 0.47 / 0.41 / 0.20 |
| skab | 10000 | 0.09 / 0.09 / 0.15 | 0.15 / 0.19 / 0.15 | 0.17 / 0.26 / 0.15 |

**native, tau1 = 1000, K = 25** (max PD with PFA <= alpha; DeepQCD all sensors, mean of seeds / DeepQCD on z / WL-CUSUM)

| dataset | n | alpha = 0.01 | alpha = 0.05 | alpha = 0.10 |
|---|---|---|---|---|
| hai | 19 | 0.25 / 0.21 / 0.26 | 0.25 / 0.21 / 0.26 | 0.33 / 0.26 / 0.26 |
| fog | 22 | 0.06 / 0.00 / 0.00 | 0.06 / 0.00 / 0.00 | 0.08 / 0.00 / 0.00 |
| smd | 38 | 0.03 / 0.08 / 0.13 | 0.03 / 0.13 / 0.18 | 0.05 / 0.16 / 0.18 |
| occupancy | 1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| occupancy-nolight | 1 | 0.67 / 0.00 / 1.00 | 0.67 / 0.00 / 1.00 | 0.67 / 0.00 / 1.00 |
| skab | 0 | - | - | - |

**blocks, tau1 = 200, K = 25** (max PD with PFA <= alpha; DeepQCD all sensors, mean of seeds / DeepQCD on z / WL-CUSUM)

| dataset | n | alpha = 0.01 | alpha = 0.05 | alpha = 0.10 |
|---|---|---|---|---|
| hai | 10000 | 0.16 / 0.32 / 0.52 | 0.59 / 0.36 / 0.52 | 0.59 / 0.38 / 0.52 |
| fog | 10000 | 0.04 / 0.01 / 0.00 | 0.08 / 0.11 / 0.00 | 0.11 / 0.14 / 0.04 |
| smd | 10000 | 0.02 / 0.20 / 0.16 | 0.11 / 0.27 / 0.35 | 0.24 / 0.34 / 0.42 |
| occupancy | 10000 | 0.95 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 |
| occupancy-nolight | 10000 | 0.09 / 0.09 / 0.20 | 0.49 / 0.39 / 0.20 | 0.49 / 0.41 / 0.37 |
| skab | 10000 | 0.25 / 0.31 / 0.14 | 0.28 / 0.33 / 0.14 | 0.28 / 0.34 / 0.20 |

**native, tau1 = 200, K = 25** (max PD with PFA <= alpha; DeepQCD all sensors, mean of seeds / DeepQCD on z / WL-CUSUM)

| dataset | n | alpha = 0.01 | alpha = 0.05 | alpha = 0.10 |
|---|---|---|---|---|
| hai | 19 | 0.25 / 0.21 / 0.26 | 0.25 / 0.21 / 0.26 | 0.33 / 0.26 / 0.26 |
| fog | 57 | 0.05 / 0.00 / 0.00 | 0.05 / 0.00 / 0.04 | 0.08 / 0.00 / 0.05 |
| smd | 52 | 0.04 / 0.06 / 0.10 | 0.05 / 0.12 / 0.13 | 0.08 / 0.23 / 0.25 |
| occupancy | 6 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| occupancy-nolight | 6 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| skab | 14 | 0.17 / 0.21 / 0.14 | 0.17 / 0.21 / 0.14 | 0.17 / 0.21 / 0.14 |

### Splicing

The step at the change point is the distance between the mean of the 10 steps before tau1 and the first 5 after it,
in standardized units, as a median over test events (`python transient_real.py --splice`). It is measured with the
real preceding stretch (native) and with a random normal test block in front (blocks). The baseline is two random
normal blocks spliced together, with no change.

| dataset | native | spliced | ratio | normal -> normal | native events |
|---|---|---|---|---|---|
| hai | 3.36 | 12.11 | 3.6x | 9.77 | 19 |
| fog | 0.81 | 1.45 | 1.8x | 1.41 | 70 |
| smd | 3.96 | 9.46 | 2.4x | 5.49 | 60 |
| occupancy | 5.07 | 6.06 | 1.2x | 2.12 | 8 |
| occupancy-nolight | 0.12 | 2.79 | 22.5x | 2.05 | 8 |
| skab | 0.66 | 2.11 | 3.2x | 1.62 | 14 |

### Findings

- **No real dataset reproduces the paper's simulated edge under its own protocol** (blocks, tau1 = 1000, K = 25,
  PD at PFA 0.05, DeepQCD mean of three seeds against WL-CUSUM):
  - HAI: 0.43 (seed range 0.34-0.54) vs 0.51;
  - SMD: 0.03 vs 0.26. The network does not transfer to unseen machines; restricted to z it reaches 0.21;
  - SKAB: 0.15 vs 0.15;
  - Occupancy with light: 0.95 vs 1.00;
  - Daphnet: 0.06 vs 0.00, so neither works. A freeze changes the tremor frequency, not the mean, and 25 samples
    are 0.8 s.
- **DeepQCD leads only on Occupancy without the light sensor** (0.47 vs 0.20), and only in spliced streams. Both
  miss all six real arrivals when the stream is not spliced: a real arrival moves the sensors by 0.12 standardized
  units, a splice by 2.79.
- **Seed-to-seed spread** in PD reaches 0.2. A lower validation loss does not mean a higher PD, because training
  scores long geometric transients while the test asks for the first 25 steps.

### Open questions for Prof. Wang

1. Is the scalar Gaussian-LLR direction z an acceptable input to `window_dec_stat`, or should another fitted model
   be used?
2. Should the real-data runs use the Sec. 5 LSTM with BCE (used here) or the Sec. 6 RegRNN (GRU 2x64, batch norm,
   MSE)?
3. Should blocks (the authors' IoT construction) or native events be primary for real data?
4. Should tau1 = 1000 and K = 25 be kept for every dataset, or set per dataset from its time scale?

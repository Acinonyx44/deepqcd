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
- **C3** contiguous normal stretches of at least tau1 - 1 + K = 1024 steps in the test data;
- **C4** at least 5 training and 5 test events lasting at least K.

`python transient_real.py --survey` checks every labelled dataset in `realdata.py` and writes
`runs/transient_real/survey.json`. The datasets used are:

- HAI, Daphnet freezing of gait and SMD, which meet all four conditions;
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

Runs in progress; the results table is added when they finish (`python transient_real.py --table`).

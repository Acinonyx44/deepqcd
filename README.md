# deepqcd

Working copy for studying and reproducing **DeepQCD: An end-to-end deep learning approach to quickest change detection**
(Kurt, Zheng, Yilmaz, Wang — *Journal of the Franklin Institute*, 2024). The pristine download stays in
`~/Downloads/DeepQCD/`; everything here is ours to edit.

## Layout

```
paper/                DeepQCD.pdf and its extracted text (grep-able)
notes/                our reading notes / audits (notebook-review.md = cell-by-cell audit of the Sec. 5 notebooks)
original/             the authors' code, untouched, one folder per paper section
  Sec. 5.1 -- IID/                     Keras notebooks: IID Gaussian, Bayesian (Fig. 5) and minimax (Fig. 6)
  Sec. 5.2 -- AR/                      Keras notebook:  AR(1) drift + correlation change (Figs. 7-8)
  Sec. 5.3 -- TransientQCD/            Keras notebook:  transient change vs window-limited CUSUM (Fig. 9)
  Sec. 6.1 -- Video Anomaly Detection/ PyTorch: I3D feature extractor + GRU detector on UCF-Crime (Fig. 13)
  Sec. 6.2 -- IoT Attack Detection/    PyTorch: GRU detector on N-BaIoT botnet traffic (Fig. 15)

qcd.py                shared machinery: DeepQCD net, training loop (Alg. 1), classical detectors written as the
                      generic QCD procedure (CUSUM, Shiryaev, SR, window-limited CUSUM), vectorized stopping-time
                      simulation over many streams x many thresholds
sources.py            the observation models (IID, AR(1), GARCH), each bundled with the model-based detectors
                      that know it; shared by the experiment scripts and detect.py
detect.py             run one experiment end to end: pick a source and a change time, train or reload the
                      detector, calibrate every threshold to the same false-alarm budget, report the delays
deepqcd_iid.py        Sec. 5.1  IID Gaussian:  DeepQCD vs Shiryaev (Bayesian) and vs CUSUM / SR (minimax)
deepqcd_ar.py         Sec. 5.2  AR(1):         DeepQCD vs the "modified" Shiryaev / CUSUM / SR of Eq. (10)
deepqcd_transient.py  Sec. 5.3  transient:     DeepQCD vs window-limited CUSUM, PD vs PFA
deepqcd_vol.py        ours:     volatility-regime change in GARCH(1,1) returns: DeepQCD vs GARCH-aware and
                      misspecified-IID CUSUM / Shiryaev and a rolling-variance rule (--sq feeds [r, r^2])
notes/results.md      the numbers from all full runs, what reproduces, what does not, and why
notes/datasets.md     real datasets (mostly on GitHub) that fit the recipe, how to turn each into QCD streams, pitfalls
notes/workflow.md     the exact train/test workflow from the paper (Algs. 1 and 2), mapped to this code
tests/test_qcd.py     checks of qcd.py against slow reference implementations (chunking, LRs, first crossings)
regen.sh              reruns every experiment, logging to runs/
figures/              output figures (tracked); runs/ holds logs, cached weights, --quick figures (git-ignored)
```

## Setup

```bash
uv sync                      # creates .venv with torch / numpy / matplotlib (Python 3.12)
uv sync --group notebooks    # + JupyterLab, if you want to open original/Sec. 5.x notebooks
```

## Run

```bash
.venv/bin/python tests/test_qcd.py            # ~10 s
.venv/bin/python deepqcd_iid.py               # ~5 min on a laptop CPU -> figures/iid.png
.venv/bin/python deepqcd_ar.py                #                        -> figures/ar.png
.venv/bin/python deepqcd_transient.py         #                        -> figures/transient.png
.venv/bin/python deepqcd_vol.py [--sq]        # ~15 min                -> figures/vol[_sq].png
```

Those four reproduce the paper's figures: they sweep the threshold and plot a tradeoff curve, as the paper does.
To instead run a single detection experiment and see what the detector actually did — one source, one change
time, one calibrated threshold, and the resulting delays — use the driver:

```bash
.venv/bin/python detect.py --source garch --tau 500 --fap 1000
.venv/bin/python detect.py --source ar --tau 500 --trials 5000     # sources: iid, ar, garch, garch-sq
```

It trains once and caches the weights in `runs/models/`, keyed by source, `--train-streams`, `--epochs` and `--seed`,
so later runs with the same settings start at the calibration step.
`notes/workflow.md` walks through what each phase does and where it lives in the code.

```bash
./regen.sh                                    # regenerate every number in notes/results.md, ~45 min
```

Every experiment script takes `--quick` (tiny dataset, 2 epochs, 10x fewer test streams) for a ~10 s smoke test;
the numbers it prints are then meaningless, only the plumbing is exercised, and its figure goes to `runs/quick/`
instead of overwriting the tracked one in `figures/`.

## What is and isn't runnable

- **Sec. 5 (synthetic)** is self-contained. The authors' notebooks pin TensorFlow 2.0-alpha (2019) and call
  `predict` once per time step, so they are slow and fragile on current stacks; the three `deepqcd_*.py` scripts
  are the PyTorch replacements. Differences from the notebooks are listed in each script's docstring (the
  important one: the minimax notebook's CUSUM/SR baselines draw one scalar noise for all 7 dimensions).
- **Sec. 6 (real data)** needs external assets that are not in this repo: the UCF-Crime videos, the I3D
  weights (`rgb_imagenet.pt`, see the `.txt` in 6.1), the N-BaIoT CSVs, and the hand-marked accident
  frames (`MARK_road_*.txt`, whose contents are in the 6.1 xlsx).

## Beyond the paper

- `deepqcd_iid.py` / `deepqcd_ar.py` also measure the minimax ADD for a change at tau = 200 (conditional on no
  earlier false alarm). The paper's tau = 1 protocol rewards start-up sensitivity that a recurrent net learns and
  the time-invariant CUSUM / SR cannot; at tau = 200 DeepQCD lands on (IID) or slightly above (AR) their curves.
- `deepqcd_vol.py` transfers the recipe to finance: a jump in long-run volatility under GARCH(1,1) clustering.

## The idea in one paragraph

Every quickest-change-detection procedure is `s_t = phi(x_t, s_{t-1})`, `d_t = omega(s_t)`, alarm at the first
`t` with `d_t >= h`. Classical methods (CUSUM, Shiryaev, Shiryaev-Roberts) hard-code `phi` and `omega` from
known pre-/post-change densities. DeepQCD makes `phi` a recurrent layer and `omega` a dense head with a
sigmoid, trains end-to-end with binary cross-entropy against labels `0` before the change-point and `1`
after, so `d_t` learns to approximate `P(change already happened | x_1..x_t)`; the threshold `h` then sweeps
the delay / false-alarm tradeoff curve.

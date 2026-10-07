# deepqcd

A reproduction and stress test of **DeepQCD: An end-to-end deep learning approach to quickest change detection**
(Kurt, Zheng, Yilmaz, Wang — *Journal of the Franklin Institute*, 2024), in PyTorch. It covers the paper's
synthetic experiments, a finance extension, and 21 real-data problems, with DeepQCD compared against classical
detectors at matched false-alarm rates.

## Bottom line

DeepQCD is a sound method that the paper oversells. It is near-optimal when the data model is known. On
real data it is a specialist, not a general replacement for classical change detectors.

- **Reproduced.** It is near-optimal in the Bayesian setting (Fig. 5), and within a few percent of the
  model-based detectors on AR(1) data (Figs. 7-8).
- **Overstated.** The claim that it "beats CUSUM / SR" (Figs. 6, 8) holds only because the change is placed at
  t = 1, which rewards a recurrent net's start-up jumpiness. With the change at t = 200 the edge disappears.
  The transient result (Fig. 9) is a tie, not a win.
- **Hidden cost.** Its false alarms come early: 10-21 % more before t = 500 than its average false-alarm period
  implies.
- **On real data**, at PFA ≤ 0.1, against the best of a fitted-Gaussian CUSUM, MEWMA, Shewhart and each field's
  standard rule:

  | | wins | ties | nobody detects | losses |
  |---|---|---|---|---|
  | DeepQCD | 5 | 3 | 1 | 12 |
  | DeepQCD-hybrid (ours) | 7 | 5 | 1 | 8 |

  - It wins when the change is a *pattern* rather than a level shift, normal data is spiky or heavy-tailed,
    and there are dozens to hundreds of labelled episodes. The standout is crypto pump-and-dump: 91-94 %
    caught within ~5 s, against 53 % within ~28 s for the best classical chart. Bee dance and server
    incidents (SMD) are also wins.
  - It loses on abrupt jumps (earthquake onset, most plant faults, grid events: Shewhart or STA/LTA win by a
    lot), with little data, and when each stream has its own normal.
- **The paper's IoT application is easy.** On a real Mirai capture in the same 115 N-BaIoT features, every
  detector catches the attack within 1-2 packets; DeepQCD's edge is about one packet.
- **Practical rule.**
  - Use DeepQCD for pattern changes with plenty of labelled episodes.
  - Use a Shewhart / MEWMA chart for abrupt shifts or small data.
  - Use the hybrid when unsure: it is rarely worse than plain DeepQCD, with one exception (S&P 500).

Details: `notes/results.md` (synthetic), `notes/realdata.md` (real data), `notes/jev.md` (Jev trial).

## Layout

```
paper/                DeepQCD.pdf and its extracted text (grep-able)
original/             the authors' code, untouched, one folder per paper section
  Sec. 5.1 -- IID/                     Keras notebooks: IID Gaussian, Bayesian (Fig. 5) and minimax (Fig. 6)
  Sec. 5.2 -- AR/                      Keras notebook:  AR(1) drift + correlation change (Figs. 7-8)
  Sec. 5.3 -- TransientQCD/            Keras notebook:  transient change vs window-limited CUSUM (Fig. 9)
  Sec. 6.1 -- Video Anomaly Detection/ PyTorch: I3D feature extractor + GRU detector on UCF-Crime (Fig. 13)
  Sec. 6.2 -- IoT Attack Detection/    PyTorch: GRU detector on N-BaIoT botnet traffic (Fig. 15)

core
  qcd.py              DeepQCD net, training loop (Alg. 1), classical detectors as the generic QCD procedure
                      (CUSUM, Shiryaev, SR, window-limited CUSUM), vectorized stopping-time simulation
  sources.py          observation models (IID, AR(1), GARCH), each bundled with the detectors that know it
  tests/test_qcd.py   checks of qcd.py against slow reference implementations

synthetic (paper Sec. 5 and a finance extension)
  deepqcd_iid.py        Sec. 5.1  IID Gaussian: DeepQCD vs Shiryaev (Bayesian), CUSUM / SR (minimax)
  deepqcd_ar.py         Sec. 5.2  AR(1): DeepQCD vs the "modified" Shiryaev / CUSUM / SR of Eq. (10)
  deepqcd_transient.py  Sec. 5.3  transient change: DeepQCD vs window-limited CUSUM, PD vs PFA
  deepqcd_vol.py        ours: volatility-regime change in GARCH(1,1) returns (--sq feeds [r, r^2])
  detect.py             one experiment end to end: calibrate every detector to a false-alarm budget, report delays

real data
  fetch_data.sh       downloads all real datasets into data/ (all GitHub-hosted, ~1.1 GB, git-ignored)
  realdata.py         loaders: 21 problems (15 benchmarks, 6 new applications) reduced to change episodes
  deepqcd_real.py     DeepQCD and DeepQCD-hybrid vs CUSUM, MEWMA, Shewhart and field-standard rules
  jev_qcd.py          trial: TypeSafe AI's Jev decision model as a zero-shot d_t (needs API access)

notes
  notes/results.md          synthetic results: what reproduces, what does not, and why
  notes/realdata.md         real-data scorecard, where DeepQCD works and where not, vs the paper's applications
  notes/datasets.md         survey of ~40 real datasets for QCD, how to turn each into streams, pitfalls
  notes/jev.md              the Jev trial: idea, protocol, cost estimate, how to run it
  notes/workflow.md         the paper's train/test workflow (Algs. 1-2) mapped to this code
  notes/notebook-review.md  cell-by-cell audit of the authors' Sec. 5 notebooks

regen.sh              reruns every experiment, logging to runs/
figures/              output figures (tracked); runs/ holds logs, cached weights, curves, --quick output (git-ignored)
```

## Setup

```bash
uv sync                      # .venv with torch / numpy / matplotlib (Python 3.12)
uv sync --group realdata     # + scipy, for the real datasets' .mat files
uv sync --group notebooks    # + JupyterLab, to open the original Sec. 5 notebooks
```

## Run

Synthetic experiments (the paper's figures, plus GARCH):

```bash
.venv/bin/python tests/test_qcd.py            # ~10 s
.venv/bin/python deepqcd_iid.py               # ~5 min on a laptop CPU -> figures/iid.png
.venv/bin/python deepqcd_ar.py                #                        -> figures/ar.png
.venv/bin/python deepqcd_transient.py         #                        -> figures/transient.png
.venv/bin/python deepqcd_vol.py [--sq]        # ~15 min                -> figures/vol[_sq].png
```

These sweep the threshold and plot a trade-off curve, as the paper does. To see what a detector actually does
at one calibrated threshold (one source, one change time, the resulting delays):

```bash
.venv/bin/python detect.py --source garch --tau 500 --fap 1000     # sources: iid, ar, garch, garch-sq
```

Trained weights are cached in `runs/models/`, keyed by source, `--train-streams`, `--epochs` and `--seed`.

Real data:

```bash
./fetch_data.sh
.venv/bin/python deepqcd_real.py              # all 21 problems, ~40 min on 4 CPUs -> figures/real.png, runs/real/
.venv/bin/python deepqcd_real.py --hybrid     # the hybrid variant, saved alongside
.venv/bin/python deepqcd_real.py skab tep     # a subset
.venv/bin/python deepqcd_real.py --report     # rebuild the table and figures/real.png from runs/real/
.venv/bin/python deepqcd_real.py --rivals     # recompute only the classical rivals into the saved runs
```

Jev trial (zero-shot; needs `api.typesafe.ai` reachable and `TYPESAFE_API_KEY` set):

```bash
.venv/bin/python jev_qcd.py --estimate        # call count and cost (~$1.60 for all five datasets), no API
.venv/bin/python jev_qcd.py --mock            # plumbing check with a labelled local stand-in, no API
.venv/bin/python jev_qcd.py occupancy pumpdump
```

Everything:

```bash
./regen.sh                                    # every number in notes/, ~1.5 h (real data if fetched)
```

Every experiment script takes `--quick` (tiny data, 2 epochs, fewer test streams) for a ~10 s smoke test. Its
numbers are meaningless, and its figure goes to `runs/quick/` instead of overwriting `figures/`.

## What is and isn't runnable

- **Paper Sec. 5 (synthetic)** is self-contained. The authors' notebooks pin TensorFlow 2.0-alpha and call
  `predict` once per time step, so the `deepqcd_*.py` scripts are the PyTorch replacements. Differences are
  listed in each script's docstring.
- **Paper Sec. 6 (real data)** needs assets not in this repo and not reachable from a cloud session: the
  UCF-Crime videos, the I3D weights (`rgb_imagenet.pt`), the N-BaIoT CSVs, and the hand-marked accident frames.
  The Mirai problems in `deepqcd_real.py` are our stand-in for the IoT application: same 115 features, a real
  infection.
- **Our real-data suite** runs anywhere with GitHub access. Datasets on other hosts are listed in
  `notes/datasets.md`; the most informative next runs (N-BaIoT itself, the 500-run Tennessee Eastman set,
  STEAD) need a laptop download.

## The idea in one paragraph

Every quickest-change-detection procedure is `s_t = phi(x_t, s_{t-1})`, `d_t = omega(s_t)`, alarm at the first
`t` with `d_t >= h`. Classical methods (CUSUM, Shiryaev, Shiryaev-Roberts) hard-code `phi` and `omega` from
known pre-/post-change densities. DeepQCD makes `phi` a recurrent layer and `omega` a dense head with a
sigmoid. It trains end-to-end with binary cross-entropy against labels `0` before the change-point and `1`
after, so `d_t` learns to approximate `P(change already happened | x_1..x_t)`. The threshold `h` then sweeps
the delay / false-alarm trade-off. Our hybrid also feeds the classical statistics into the network; the Jev
trial asks a decision model for the same probability zero-shot.

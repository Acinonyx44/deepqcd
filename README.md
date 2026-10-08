# deepqcd

A reproduction and stress test of **DeepQCD: An end-to-end deep learning approach to quickest change detection**
(Kurt, Zheng, Yilmaz, Wang — *Journal of the Franklin Institute*, 2024), in PyTorch. It covers the paper's
synthetic experiments, a finance extension, and 21 real-data problems, with DeepQCD compared against classical
detectors at matched false-alarm rates.

## Takeaways

DeepQCD is a sound method that the paper oversells. It is near-optimal when the data model is known. On real
data it is a specialist, not a general replacement for classical change detectors.

**The paper's own claims:**
- **Reproduced.** It is near-optimal in the Bayesian setting (Fig. 5), and within a few percent of the
  model-based detectors on AR(1) data (Figs. 7-8).
- **Overstated.** The claim that it "beats CUSUM / SR" (Figs. 6, 8) holds only because the change is placed at
  t = 1, which rewards a recurrent net's start-up jumpiness. With the change at t = 200 the edge disappears.
  The transient result (Fig. 9) is a tie, not a win.
- **Hidden cost.** Its false alarms come early: 10-21 % more before t = 500 than its average false-alarm period
  implies.
- **The paper's IoT application is easy.** On a real Mirai capture in the same 115 N-BaIoT features, every
  detector catches the attack within 1-2 packets; DeepQCD's edge is about one packet.

**On 21 real problems** (PFA ≤ 0.1, against the best classical detector on each):

| | wins | ties | nobody detects | losses |
|---|---|---|---|---|
| DeepQCD | 5 | 3 | 1 | 12 |
| DeepQCD-hybrid (ours) | 7 | 5 | 1 | 8 |

**The protocol decides most of this.** Rerunning the same 21 datasets the paper's way (rows shuffled into IID
streams, delay at τ = 1, false alarms at τ = ∞; `notes/realdata-tau1.md`, `figures/real_tau1.png`) gives smooth,
paper-like curves, and DeepQCD then wins 13 and loses 2 (at FAP 100). Same data, same network, same rivals.

- **DeepQCD wins** when the change is a *pattern* rather than a level shift, normal data is spiky or heavy-tailed,
  and there are dozens to hundreds of labelled episodes:
  - crypto pump-and-dump: 91-94 % caught within ~5 s, against 53 % within ~28 s for the best chart;
  - bee dance: 0.80 vs 0.30;
  - server incidents (SMD).
- **It loses on:**
  - abrupt jumps: earthquake onset, most plant faults, grid events, where Shewhart or STA/LTA win by a lot;
  - small data: 7-20 training episodes;
  - streams with their own normal: keystroke takeover, where a self-calibrating chart catches 97 %.
- **The hybrid** (classical statistics fed into the network) is rarely worse than plain DeepQCD, with one
  exception (S&P 500). It turns occupancy and C-MAPSS into wins.

**Without labels (unsupervised):**
- **Unsupervised detectors win on about two thirds of the problems**, often beating the supervised ones:
  - Shewhart / MEWMA charts on normal data: Tennessee Eastman 0.90 vs DeepQCD 0.66, seismic 0.93 vs 0.27,
    grid events 0.91 vs 0.73;
  - a self-calibrating chart on keystrokes;
  - zero-shot Chronos on SKAB, occupancy without light, and SMD.
- **Labels pay off clearly only on distinctive change signatures:** pump-and-dump (0.91 vs 0.50), bee dance
  (0.80 vs 0.47), NAB.
- **Zero-shot Chronos** (no training at all) beats DeepQCD on seismic (0.84 vs 0.25) and grid events, but not
  the Shewhart chart. It fails on slow level drifts: C-MAPSS, HAI, S&P 500, pump-and-dump.

**Which detector to use:**

| situation | use |
|---|---|
| abrupt jump in level or variance | Shewhart chart, or STA/LTA for seismic |
| small persistent shift, only normal data | MEWMA chart |
| each user / stream has its own normal | self-calibrating chart (baseline from the stream's own start) |
| distinctive pattern, many labelled episodes | DeepQCD, or the hybrid |
| unsure, labels available | DeepQCD-hybrid |
| no data at all, subtle change in dynamics | Chronos zero-shot forecast surprise |

Details: `notes/results.md` (synthetic), `notes/realdata.md` (real data), `notes/zeroshot.md` (zero-shot),
`notes/datasets.md` (dataset survey). A plain-language walkthrough of every chart, with likely questions and
answers, is in `notes/defense-guide.md`.

## Comparison tables

### Synthetic experiments (paper Sec. 5, and GARCH)

Model-based rivals know the true densities. Delays are in time steps; lower is better. Full runs are in `notes/results.md`.

| experiment | setting | DeepQCD | best model-based | verdict |
|---|---|---|---|---|
| IID Gaussian, p = 7 | Bayesian ADD at PFA 0.01 | 2.49 | Shiryaev 2.41 | near-optimal (reproduced) |
| | minimax ADD at FAP 1000, change at t = 1 | 0.06 | CUSUM 1.17 | "beats CUSUM" (paper's protocol) |
| | minimax ADD at FAP 1000, change at t = 200 | 1.21 | CUSUM 1.11 | tie: the t = 1 win is a start-up artifact |
| AR(1) | Bayesian ADD at PFA 0.01 | 7.68 | mod. Shiryaev 7.16 | within ~7 % |
| | minimax ADD at FAP 1000, change at t = 200 | 4.51 | mod. SR 4.29 | ~5 % slower |
| Transient (K = 25) | detection probability at PFA 0.1 | 0.833 | window-limited CUSUM 0.839 | tie (paper claims a win) |
| GARCH volatility × 2 | Bayesian ADD at PFA 0.01, [r, r²] input | 71.2 days | GARCH Shiryaev 66.8 | within ~7 % |
| | conditional ADD at FAP 1000, change at t = 200 | 26.6 days | IID CUSUM 26.0 | tie |

### All 21 real-data problems

Each cell is **detection rate · mean delay** (delay in the dataset's own step, misses counted as the full horizon)
at the operating point with PFA ≤ 0.1, on the same test windows. DeepQCD and the hybrid are the median of 3 seeds.
"Field rule" is the domain's standard detector where one exists. "Best" lists every detector within 0.03 in
detection rate and 10 % in delay of the top one.

| dataset | step | DeepQCD | DeepQCD-hybrid | CUSUM (fitted) | MEWMA | Shewhart | field rule | Chronos ⁱ | best |
|---|---|---|---|---|---|---|---|---|---|
| SKAB water pump | s | 0.09 · 116.7 | 0.20 · 101.1 | 0.21 · 99.7 | 0.28 · 93.1 | 0.23 · 98.9 | — | 0.34 · 88.4 | Chronos |
| Tennessee Eastman (21 faults) | 3 min | 0.66 · 103.3 | 0.88 · 64.6 | 0.76 · 72.6 | 0.87 · 50.5 | 0.90 · 37.2 | — | 0.61 · 102.3 | Shewhart |
| UCI room occupancy | min | 0.82 · 5.4 | 1.00 · 0.5 | 1.00 · 1.1 | 1.00 · 14.3 | 0.55 · 34.0 | — | 0.84 · 14.8 | hybrid |
| UCI room occupancy (no light sensor) | min | 0.09 · 46.0 | 0.44 · 37.1 | 0.40 · 40.5 | 0.23 · 50.3 | 0.18 · 49.9 | — | 0.54 · 30.7 | Chronos |
| C-MAPSS FD001 turbofans | cycle | 1.00 · 45.1 | 1.00 · 34.6 | 1.00 · 49.8 | 1.00 · 51.0 | 1.00 · 87.4 | — | 0.31 · 111.5 | hybrid |
| SMD server machines | min | 0.18 · 22.6 | 0.17 · 22.5 | 0.02 · 29.7 | 0.03 · 27.9 | 0.10 · 26.4 | — | 0.17 · 23.2 | DeepQCD ≈ hybrid ≈ Chronos |
| HAI 21.03 ICS attacks | s | 0.84 · 30.3 | 0.94 · 28.9 | 0.90 · 29.9 | 0.94 · 29.7 | 0.80 · 42.5 | — | 0.31 · 79.1 | hybrid ≈ MEWMA |
| NAB (58 series) | step | 0.16 · 84.0 | 0.11 · 89.7 | 0.29 · 72.5 | 0.14 · 85.9 | 0.14 · 85.8 | — | 0.09 · 92.6 | CUSUM |
| TCPD (univariate, consensus CPs) | step | 0.27 · 17.8 | 0.27 · 17.7 | 0.27 · 17.8 | 0.27 · 17.5 | — | — | 0.09 · 21.4 | MEWMA ≈ hybrid ≈ CUSUM … |
| Bee waggle dance (6 seqs) | frame | 0.80 · 15.1 | 0.78 · 14.6 | 0.24 · 22.7 | 0.30 · 22.2 | 0.13 · 24.5 | — | 0.47 · 19.6 | DeepQCD ≈ hybrid |
| HASC accelerometer activities | sample | 0.00 · 89.9 | 0.00 · 89.8 | 0.13 · 83.4 | 0.02 · 88.2 | 0.07 · 86.5 | — | 0.15 · 82.7 | Chronos ≈ CUSUM |
| Dam water level (fish kills) | step | 0.00 · 15.8 | 0.00 · 16.1 | 0.01 · 15.6 | 0.00 · 16.3 | 0.00 · 16.3 | — | 0.06 · 15.4 | none detect |
| Yahoo S5 subset (15 series) | hour | 0.17 · 2.5 | 0.15 · 2.5 | 0.17 · 2.6 | 0.06 · 2.8 | 0.17 · 2.4 | — | 0.14 · 2.0 | Shewhart ≈ DeepQCD ≈ CUSUM … |
| pmuBAGE grid events | 1/30 s | 0.73 · 39.9 | 0.85 · 29.3 | 0.88 · 21.2 | 0.91 · 17.4 | 0.66 · 45.0 | — | 0.97 · 10.5 | Chronos |
| S&P 500 stress episodes | day | 0.86 · 15.6 | 0.73 · 22.9 | 0.93 · 8.7 | 0.86 · 14.0 | 0.86 · 12.4 | — | 0.47 · 37.1 | CUSUM |
| Seismic P-wave onset (PhaseNet) | 10 ms | 0.27 · 235.1 | 0.46 · 191.1 | 0.20 · 250.1 | 0.75 · 122.0 | 0.93 · 46.8 | STA/LTA 0.81 · 88.3 | 0.84 · 75.0 | Shewhart |
| Freezing of gait (Daphnet) | 1/32 s | 0.18 · 101.5 | 0.26 · 95.2 | 0.28 · 98.1 | 0.00 · 115.3 | 0.09 · 113.7 | Freeze 0.20 · 103.3 | 0.02 · 114.4 | CUSUM ≈ hybrid |
| Crypto pump-and-dump | 5 s chunk | 0.91 · 1.3 | 0.94 · 0.9 | 0.53 · 5.7 | 0.42 · 7.0 | 0.50 · 5.9 | — | 0.04 · 11.5 | hybrid ≈ DeepQCD |
| Account takeover (keystrokes) | entry | 0.43 · 29.2 | 0.91 · 8.0 | 0.15 · 37.3 | 0.10 · 37.8 | 0.08 · 38.3 | Self-calibrating 0.97 · 5.7 | 0.71 · 13.4 | Self-calibrating |
| IoT Mirai botnet (temporal blocks) | packet | 1.00 · 1.4 | 1.00 · 1.3 | 1.00 · 1.7 | 1.00 · 3.3 | 1.00 · 2.2 | — | 0.60 · 139.1 | hybrid ≈ DeepQCD ≈ CUSUM |
| IoT Mirai botnet (paper protocol: IID splice) | packet | 1.00 · 0.2 | 1.00 · 0.2 | 1.00 · 1.1 | 1.00 · 1.2 | 1.00 · 1.3 | — | 0.86 · 59.6 | hybrid ≈ DeepQCD |

ⁱ Chronos was run separately (`notes/zeroshot.md`): on the first 200 of the same test windows (100 for the IID
IoT set), and allowed to alarm only every few steps, which costs it a little delay compared with the full-rate
detectors.

What needs what:
- **Labelled changes:** DeepQCD, the hybrid, and the fitted CUSUM.
- **Normal data only:** MEWMA and Shewhart.
- **No training data:** the field rules and Chronos.

## What we did

1. **Reproduced paper Sec. 5 in PyTorch.**
   - Wrote the IID, AR(1) and transient experiments with the authors' architecture and protocol, and a
     vectorized simulator that evaluates thousands of streams in minutes.
   - The authors' notebooks call `predict` once per step; the cell-by-cell audit is in `notes/notebook-review.md`.
2. **Probed the protocol.**
   - Added the change at t = 200 (conditional ADD), which exposed the t = 1 start-up artifact.
   - Wrote `detect.py`, which calibrates every detector to one false-alarm budget before measuring delays. It
     revealed DeepQCD's front-loaded false alarms.
3. **Finance extension.** A GARCH(1,1) volatility-regime change (`deepqcd_vol.py`): DeepQCD matches the
   model-based detectors at matched false-alarm rates; feeding [r, r²] helps.
4. **Surveyed ~40 real datasets** (`notes/datasets.md`): what a dataset needs for QCD, three ways to build
   streams (native runs, splicing, annotated series), and pitfalls such as SKAB's fixed change index.
5. **Real-data harness** (`realdata.py`, `deepqcd_real.py`): 21 problems reduced to one protocol, with random
   change times, matched-PFA scoring, 3 seeds, and classical and domain rivals.
   - Fixed along the way: Tennessee Eastman needed a second normal run; STA/LTA needed a robust noise level;
     a Shewhart rival was added after it proved the strongest on abrupt changes.
6. **Hybrid DeepQCD.** The LSTM also sees the fitted log-LR, the MEWMA statistic and the input re-referenced to
   the stream's start: 7 wins instead of 5.
7. **Zero-shot trial** (`zeroshot_qcd.py`): asked pretrained models for d_t with no training. Four backends:
   - **Chronos:** run on all 21 datasets.
   - **Qwen** (open LLM, P(yes) from logits): needs a GPU; a CPU spot check was poor.
   - **TypeSafe Jev** and **OpenAI-compatible** endpoints: built and verified offline, not yet run (needs a
     session with the API host reachable and a key).

## Datasets

| group | dataset | what changes | train / test episodes | source |
|---|---|---|---|---|
| synthetic | IID Gaussian, p = 7 | mean 0 → 1 | simulated | paper Sec. 5.1 |
| | AR(1) | drift and correlation | simulated | paper Sec. 5.2 |
| | transient | mean shift for ~25 steps | simulated | paper Sec. 5.3 |
| | GARCH(1,1) | long-run volatility × 2 | simulated | ours |
| industrial | Tennessee Eastman (Braatz) | 21 process faults | 42 / 21 | GitHub mirror |
| | SKAB water pump | valve / pump faults | 20 / 14 | waico/SKAB |
| | C-MAPSS FD001 | engine degradation (onset by convention) | 67 / 29 | NASA, GitHub mirror |
| | UCI occupancy (± light sensor) | room becomes occupied | 7 / 12 | GitHub mirror |
| security / IT | HAI 21.03 | ICS cyber-attacks | 30 / 20 | icsdataset/hai |
| | SMD | server incidents, 38 metrics | 155 / 171 | NetManAIOps/OmniAnomaly |
| | IoT Mirai (temporal / paper's IID splice) | botnet infection, N-BaIoT features | 300 / 150 | ymirsky/KitNET-py |
| | account takeover | different typist (CMU keystrokes) | 600 / 300 | GitHub mirror |
| finance | S&P 500 | 16 dated stress episodes | GARCH sim / 16 | GitHub mirror |
| | crypto pump-and-dump | 317 Telegram pumps | 190 / 127 | SystemsLab-Sapienza |
| science / health | seismic (PhaseNet) | earthquake P-wave arrival | 92 / 62 | AI4EPS/PhaseNet |
| | freezing of gait (Daphnet) | Parkinson's freeze onset | 151 / 72 | GitHub mirror |
| | pmuBAGE | power-grid events | 110 / 74 | NanpengYu/pmuBAGE |
| benchmarks | NAB, TCPD, Yahoo S5 | labelled anomalies / change points | 63/46, 35/24, 32/20 | numenta, alan-turing-institute, KL-CPD |
| | bee dance, HASC, fish kill | behaviour / activity / water-level changes | 74/43, 17/48, 25/15 | KL-CPD |

All real data is GitHub-hosted and fetched by `fetch_data.sh`. Datasets on other hosts (N-BaIoT itself,
UCF-Crime, the 500-run Tennessee Eastman, STEAD, CHB-MIT, ...) are listed in `notes/datasets.md`.

## Methods

Every detector is the generic QCD procedure: a state `s_t = phi(x_t, s_{t-1})`, a statistic `d_t = omega(s_t)`,
and an alarm at the first `t` with `d_t >= h`. They differ in what `phi` and `omega` are, and in what they need.

| method | architecture / rule | needs |
|---|---|---|
| **DeepQCD** | LSTM (16 units; 32 above 10 inputs) → Dense(10, ReLU) → Dense(1, sigmoid); BCE against 0/1 labels; Adam, early stopping | labelled change episodes |
| **DeepQCD-hybrid** (ours) | same network; inputs = observations, observations minus the stream's own start, fitted log-LR, log(1 + MEWMA T²) | labelled change episodes |
| Shiryaev / CUSUM / Shiryaev-Roberts | exact likelihood-ratio recursions (synthetic experiments, where the model is known) | the true pre/post densities |
| CUSUM (fitted Gaussians) | CUSUM on the log-LR of two Gaussians fitted to pre- and post-change training data | labelled changes |
| MEWMA chart | multivariate EWMA (λ = 0.1) of whitened observations, Hotelling T² | normal data only |
| Shewhart chart | Hotelling T² of each observation (λ = 1) | normal data only |
| self-calibrating chart | diagonal MEWMA against the stream's own first 15 observations | nothing |
| STA/LTA | short / long-term energy ratio (seismology standard) | nothing |
| freeze index | 3-8 Hz / 0.5-3 Hz power ratio (gait standard) | nothing |
| window-limited CUSUM | max suffix sum of LLRs over the last K steps (transient changes) | the true densities |
| rolling variance | 20-day variance of returns (finance) | nothing |
| **Chronos-Bolt** (zero-shot) | Amazon's pretrained forecaster; CUSUM of squared quantile-normalized forecast errors | nothing |
| **LLM decision** (zero-shot) | open LLM (Qwen) or OpenAI-compatible endpoint; P(Yes) from next-token logprobs on a JSON summary of the stream | nothing (API key for hosted) |
| **TypeSafe Jev** (zero-shot) | "System One" decision model; typed yes/no question, probability returned | API key |

All detectors on a dataset see the same test windows. They are compared at the threshold that meets the same
false-alarm budget (PFA ≤ 0.05 / 0.1 / 0.25), never at equal thresholds.

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
  deepqcd_tau1.py     the same datasets under the paper's Sec. 6.2 protocol (IID resampling, tau = 1 / inf)
  zeroshot_qcd.py     trial: pretrained models as a zero-shot d_t (Jev, open LLM logits, OpenAI-compatible, Chronos)

notes
  notes/results.md          synthetic results: what reproduces, what does not, and why
  notes/realdata.md         real-data scorecard, where DeepQCD works and where not, vs the paper's applications
  notes/realdata-tau1.md    the same datasets under the paper's protocol, and why the verdict flips
  notes/datasets.md         survey of ~40 real datasets for QCD, how to turn each into streams, pitfalls
  notes/zeroshot.md         the zero-shot trial: backends, protocol, costs, how to run it
  notes/workflow.md         the paper's train/test workflow (Algs. 1-2) mapped to this code
  notes/notebook-review.md  cell-by-cell audit of the authors' Sec. 5 notebooks

regen.sh              reruns every experiment, logging to runs/
figures/              output figures (tracked); runs/ holds logs, cached weights, curves, --quick output (git-ignored)
```

## Setup

```bash
uv sync                      # .venv with torch / numpy / matplotlib (Python 3.12)
uv sync --group realdata     # + scipy, for the real datasets' .mat files
uv sync --group zeroshot     # + transformers, chronos-forecasting, for the zero-shot trial
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
.venv/bin/python deepqcd_tau1.py              # paper protocol, ~1.5 h -> figures/real_tau1.png, runs/tau1/
```

Zero-shot trial (pretrained models, no training; needs the model hosts reachable, see `notes/zeroshot.md`):

```bash
uv sync --group zeroshot                                        # transformers, chronos-forecasting
.venv/bin/python zeroshot_qcd.py --backend chronos              # Amazon Chronos-Bolt, all 21 datasets, no key
.venv/bin/python zeroshot_qcd.py --backend hf                   # open LLM, P(yes) from logits, no key
.venv/bin/python zeroshot_qcd.py --backend jev --estimate       # TypeSafe Jev (TYPESAFE_API_KEY), ~$1.60
.venv/bin/python zeroshot_qcd.py --backend openai occupancy     # any OpenAI-compatible endpoint (OPENAI_API_KEY)
.venv/bin/python zeroshot_qcd.py --backend hf --mock            # plumbing check, labelled stand-in, no network
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
the delay / false-alarm trade-off. Our hybrid also feeds the classical statistics into the network; the zero-shot
trial asks pretrained models (Jev, open LLMs, Chronos) for the same probability with no training.

## License

Our code and notes are under the MIT License (`LICENSE`). Third-party material is not covered by it:

- `paper/` is the published article, © 2024 The Franklin Institute / Elsevier, all rights reserved. It is
  included for reference only.
- `original/` is the paper authors' code, unmodified and unlicensed.
- Datasets fetched into `data/` keep their sources' terms.

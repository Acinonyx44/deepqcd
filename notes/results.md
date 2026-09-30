# Reproduction results (PyTorch ports, laptop CPU)

All numbers from the `deepqcd_*.py` scripts with their default settings (seed 0); logs in `runs/`, figures in
`figures/`. "ADD at PFA/FAP = x" is read off the tradeoff curve by log-interpolation, so compare detectors at
matched false-alarm level, never at the same numeric threshold.

## Sec. 5.1 IID Gaussian, p = 7, mean 0 -> 1 (`deepqcd_iid.py`, figures/iid.png)

Training: 20 epochs, val BCE 0.0042 (the notebook's saved run: 0.0041).

| setting | level | DeepQCD | Shiryaev | CUSUM | SR |
|---|---|---|---|---|---|
| Bayesian ADD | PFA 0.1 | 1.68 | 1.60 | | |
| Bayesian ADD | PFA 0.01 | 2.49 | 2.41 | | |
| minimax ADD, tau = 1 | FAP 100 | 0.00 | | 0.54 | 0.67 |
| minimax ADD, tau = 1 | FAP 1000 | 0.06 | | 1.17 | 1.23 |
| minimax ADD, tau = 1 | FAP 10000 | 0.46 | | 1.86 | 1.88 |

Paper's claims reproduced: near-optimal in the Bayesian setting (Fig. 5); "beats" CUSUM / SR at tau = 1 (Fig. 6).
The single-stream panel shows d_t sitting on top of the exact Shiryaev posterior. See the tau = 200 caveat below.

## Sec. 5.2 AR(1), drift 0 -> 1 and coefficient -0.3 -> 0.2 (`deepqcd_ar.py`, figures/ar.png)

Training: 20 epochs, val BCE 0.0141.

| setting | level | DeepQCD | mod. Shiryaev | mod. CUSUM | mod. SR |
|---|---|---|---|---|---|
| Bayesian ADD | PFA 0.1 | 5.38 | 5.11 | | |
| Bayesian ADD | PFA 0.01 | 7.68 | 7.16 | | |
| minimax ADD, tau = 1 | FAP 100 | 0.35 | | 2.90 | 3.20 |
| minimax ADD, tau = 1 | FAP 1000 | 2.34 | | 4.53 | 4.82 |
| minimax ADD, tau = 1 | FAP 10000 | 5.04 | | 6.20 | 6.52 |

Paper's claims reproduced: comparable to the modified Shiryaev (Fig. 7, DeepQCD ~5-7 % slower), well ahead of
the modified CUSUM / SR at tau = 1 (Fig. 8).

## Sec. 5.3 transient change, tau1 = 1000, K = 25 (`deepqcd_transient.py`, figures/transient.png)

Training stopped early at epoch 15 (val BCE 0.0348; validation uses swapped geometric rates, as in the notebook).

| level | DeepQCD PD | window-limited CUSUM PD |
|---|---|---|
| PFA 0.1 | 0.833 | 0.839 |
| PFA 0.03 | 0.844 | 0.852 |
| PFA 0.01 | 0.797 | 0.818 |

**Not reproduced as stated:** the paper says the window-limited CUSUM "is still slightly outperformed by DeepQCD"
(Fig. 9); we get the two curves on top of each other with the benchmark ahead by 0.5-2 % PD. The difference is at
the level of training / Monte-Carlo noise, so the fair summary is "ties a benchmark that knows f0, f1 and K",
not "beats it". The single-stream panel shows why it is hard: trained on transients of mean length 500, the
network's d_t ramps up over most of a 25-step transient, while the window statistic reacts within a few steps.

## Caveat on the tau = 1 minimax comparisons (Figs. 6 and 8)

The paper measures the minimax ADD only for a change at the very first observation. A recurrent network can learn
to be extra sensitive during the first few steps after start-up at almost no cost in FAP (a steady-state
quantity), while the CUSUM / SR recursions are time-invariant. The IID and AR scripts therefore also measure the
conditional ADD for a change at tau = 200 (streams that false-alarm before it excluded).

IID (`deepqcd_iid.py`, dashed curves in figures/iid.png):

| level | DeepQCD, tau = 1 | DeepQCD, tau = 200 | CUSUM, tau = 1 | CUSUM, tau = 200 | SR, tau = 200 |
|---|---|---|---|---|---|
| FAP 100 | 0.00 | 0.88 | 0.54 | 0.53 | (grid starts higher) |
| FAP 1000 | 0.06 | 1.21 | 1.17 | 1.11 | (grid starts higher) |
| FAP 10000 | 0.46 | 1.83 | 1.86 | 1.82 | 1.80 |

Confirmed: once the change happens after the detectors have settled, DeepQCD's conditional ADD lands on the
CUSUM / SR curves (slightly above them at low FAP), while CUSUM's own numbers barely move between tau = 1 and
tau = 200 because its recursion is time-invariant. The "better than minimax-optimal" gap in Fig. 6 is a start-up
effect of the tau = 1 protocol, not a steady-state advantage. (SR's linear threshold grid from the notebook,
1, 409, 817, ..., has no points with FAP between ~3 and ~2000, hence the blanks.)

AR(1) (`deepqcd_ar.py`, dashed curves in figures/ar.png):

| level | DeepQCD, tau = 1 | DeepQCD, tau = 200 | mod. CUSUM, tau = 1 | mod. CUSUM, tau = 200 | mod. SR, tau = 200 |
|---|---|---|---|---|---|
| FAP 100 | 0.35 | 3.48 | 2.90 | 2.66 | (grid starts higher) |
| FAP 1000 | 2.34 | 4.51 | 4.53 | 4.32 | 4.29 |
| FAP 10000 | 5.04 | 6.11 | 6.20 | 5.95 | 5.94 |

Same picture, slightly less flattering: after settling, DeepQCD is 3-30 % *slower* than the modified CUSUM / SR,
which are asymptotically minimax-optimal for this AR(1) model and know its parameters. So the fair reading of
Figs. 6 and 8 is "a model-free detector that gets within a few percent of the model-based optimum", which is
still the paper's real point; the "outperforms" wording rests on the tau = 1 protocol.

## Finance: volatility-regime change under GARCH(1,1) (`deepqcd_vol.py`, figures/vol.png, figures/vol_sq.png)

Our experiment, not in the paper. Standardized daily returns, GARCH(1,1) with alpha = 0.05, beta = 0.90; at tau the
long-run vol jumps from 1 to 2 (clustering dynamics unchanged). Benchmarks know all parameters; DeepQCD sees raw
r_t (`vol`) or [r_t, r_t^2] (`vol_sq`, the paper's "feature transformation"). ADD in trading days.

Training: raw r_t early-stopped at epoch 14 (val BCE 0.0802); [r, r^2] trained to epoch 19 (val BCE 0.0756).

Bayesian, tau ~ geo(0.001):

| level | DeepQCD raw | DeepQCD [r, r^2] | GARCH Shiryaev | IID Shiryaev |
|---|---|---|---|---|
| PFA 0.1 | 42.5 | 40.2 | 39.0 | n/a |
| PFA 0.01 | 78.0 | 71.2 | 66.8 | n/a |

The misspecified IID Shiryaev posterior saturates on ordinary vol clusters: with the paper's threshold grid
(h <= 0.995) it never gets below PFA ~ 0.25, so it has no entry at these levels (a finer grid near 1 would extend
its curve, but the point stands: a posterior built on the wrong dynamics is not a usable dial).

Minimax, tau = 1 (ADD) and tau = 200 (conditional ADD; the meaningful one here, since at tau = 1 the variance starts
at its calm level and ramps, and the recurrent net gets the start-up bonus):

| level | DeepQCD raw | DeepQCD [r, r^2] | GARCH CUSUM | IID CUSUM | rolling var (20d) |
|---|---|---|---|---|---|
| FAP 250, tau = 1 | 5.0 | 10.0 | 21.2 | 17.2 | 20.0 |
| FAP 1000, tau = 1 | 16.8 | 23.9 | 31.3 | 26.7 | 28.3 |
| FAP 5000, tau = 1 | 37.4 | 39.7 | 46.0 | 41.0 | 43.2 |
| FAP 250, tau = 200 | 26.3 | 21.4 | 17.7 | 17.0 | 18.0 |
| FAP 1000, tau = 200 | 28.7 | 26.6 | 26.8 | 26.0 | 27.0 |
| FAP 5000, tau = 200 | 42.1 | 39.7 | 41.0 | 40.1 | 42.1 |

Reading:
- Fairly evaluated (tau = 200), a model-free DeepQCD trained on simulated regimes **matches** the model-based
  detectors at FAP >= 1000 (~1-2 months of delay for a vol doubling at one false alarm per 4-20 years) and is
  20-50 % slower at FAP 250. It does not beat them; the tau = 1 rows that say otherwise are the start-up artifact.
- The feature transformation matters: [r, r^2] beats raw r_t everywhere that counts (Bayesian 5-9 %, tau = 200 up
  to 19 %). The LSTM can learn to square, but not for free.
- All three model-based rules end up within a few percent of each other once thresholds are matched by FAP; the
  20-day rolling variance is a perfectly competitive detector for this change, and the GARCH-aware CUSUM buys
  nothing over the IID one here (its parallel-recursion LR is itself an approximation). The GARCH-aware CUSUM does
  have a much better FAP *per threshold* (h = 6 -> ~5k days, h = 8 -> ~36k) because it discounts clusters, which
  is why its grid had to be cut at 8.
- Where the mismatch bites is the *Bayesian* posterior: the misspecified IID Shiryaev is unusable at low PFA,
  whereas the model-free network gets within 3-7 % ([r, r^2]) of the GARCH-aware posterior.

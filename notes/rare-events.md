# Rare changes: train where changes are common, deploy where they are rare

**Question.** Credit default, money laundering and delinquency are rare: a bank may see one case in ten thousand
account-days. Labelled training data for DeepQCD is easiest to collect where changes are common (enriched samples,
case-control data). Does a network trained at a high change rate still work at a low one, and how should its output
be read there?

## Files

- `rare_prior.py`: the experiment. Run with `.venv/bin/python rare_prior.py`. It takes about 7 minutes on 4 cores and
  writes `runs/rare/results.json` and the two networks `runs/rare/net_A.pt` and `runs/rare/net_B.pt`.

## Theory

In the Bayesian setting the network's output approximates the Shiryaev posterior P(tau <= t | x_1..x_t), and that
posterior depends on the change rate rho. For small rho, the posterior odds are close to rho times the
Shiryaev-Roberts statistic, so moving from the training rate rho_tr to the deployment rate rho_dep shifts the
log-odds by a constant:

    logit p_dep = logit p_train + log(rho_dep / rho_tr)

- **The shift is monotone, so it cannot change the PFA-ADD curve.** It changes which threshold gives which
  false-alarm rate.
- **The Bayesian rule needs a calibrated posterior.** The rule "stop when the posterior reaches 1 - alpha" gives
  PFA <= alpha only if the posterior is calibrated. So the shift decides whether that rule can be used.
- **The shift is exact for the exact posterior.** Shiryaev(1e-3), shifted by log 0.1, stops with PFA 0.059 / 0.029 /
  0.0057 at alpha = 0.1 / 0.05 / 0.01. Shiryaev(1e-4) stops with 0.058 / 0.028 / 0.0057.

## Setup

The setting is paper Sec. 5.1 with p = 1, f0 = N(0, 1) and f1 = N(1, 1). Both networks are trained exactly as in the
notebooks, on 3,200 streams of 2,000 steps, with 20 epochs.

- **A, enriched.** Trained with tau ~ geo(1e-3), the paper's rate. 85% of its training streams contain a change.
- **B, true rate.** Trained with tau ~ geo(1e-4) on the same budget. Only 18% of its streams contain a change.

The deployment test uses tau ~ geo(1e-4), truncated at 60,000 steps (99.75% of the mass), over 3,000 streams.

## Results

**Training at the enriched rate costs nothing.** ADD at a given PFA:

| Detector | PFA 0.1 | 0.05 | 0.01 | 0.005 |
|---|---|---|---|---|
| DeepQCD A, enriched | 16.2 | 18.5 | 22.5 | 23.8 |
| DeepQCD B, true rate | 17.1 | 19.4 | 23.2 | 23.9 |
| Shiryaev(1e-4), optimal | 16.1 | 18.4 | 22.3 | 23.5 |
| Shiryaev(1e-3), wrong prior | 16.1 | 18.2 | 22.4 | 23.9 |

**The rule "stop at posterior 1 - alpha" (realized PFA, ADD):**

| Posterior | alpha 0.1 | alpha 0.05 | alpha 0.01 |
|---|---|---|---|
| Shiryaev(1e-4), optimal | 0.062, 17.5 | 0.031, 19.6 | 0.005, 23.5 |
| Shiryaev(1e-3), wrong prior | 0.384, 8.7 | 0.234, 12.0 | 0.057, 17.8 |
| A, raw | **0.137**, 14.8 | 0.046, 18.5 | 0.003, 24.5 |
| A, shifted only | 0.004, 24.2 | 0.002, 26.4 | 0, 31.3 |
| A, calibrated at training rate | 0.385, 8.8 | 0.198, 13.0 | 0.024, 20.4 |
| A, calibrated, then shifted | 0.028, 20.0 | 0.010, 22.5 | 0.002, 27.1 |
| B, raw | 0.004, 24.2 | 0.002, 26.5 | 0, 32.1 |

"Calibrated" is a logistic fit of logit P(change by t) = a * logit(d_t) + b. It is fitted on 500 labelled streams at
the training rate, where labels are plentiful. The fit is a = 1.10, b = 0.87.

## Findings

1. **Where the network was trained hardly matters for detection.** The PFA-ADD curves of A, B and the optimal
   Shiryaev agree within one step. So the network can be trained where changes are common, which is where labelled
   data exists.
2. **The raw output is not the deployment posterior.** Read as a probability, A's output over-alarms at alpha = 0.1
   (PFA 0.137). It behaves like the wrong-prior Shiryaev, but less badly.
3. **The theoretical shift alone over-corrects.** It gives PFA 0.004 for alpha = 0.1, 25 times below target, and ADD
   24 instead of 17.5. The network was trained to be a posterior at rho = 1e-3, but its output is not calibrated
   there either: logit(d) is too low by about 0.9. The shift is exact only for an exact posterior.
4. **Calibrating at the training rate, then shifting, works.** Calibrated, A behaves like Shiryaev(1e-3) (0.385 /
   0.198 / 0.024 against 0.384 / 0.234 / 0.057). Shifted, it gives a valid rule: PFA 0.028 / 0.010 / 0.002, below
   alpha, and in the low and middle range its reliability at the deployment rate matches the empirical frequency.
   Neither step needs labelled data at the rare rate. It stays about twice as conservative as the exact posterior,
   because it is under-confident above about 0.8, where a two-parameter fit cannot follow the network's curve.
5. **Training at the true rate does not fix calibration.** B gives the same over-conservative thresholds as "shifted
   only", because with few changes in its training data its output is under-confident.

**For a bank:** train on enriched data, choose the threshold by the target false-alarm rate on held-out data when that
is possible, and otherwise calibrate on the enriched data and shift by log(rho_dep / rho_tr), knowing the rule will be
conservative. The deployment rate rho_dep has to be known or estimated; an error of a factor c in it moves the
threshold by log c.

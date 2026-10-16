# Market-maker stress detection on a synthetic S&P 500 futures market

**Question.** When a market maker has to offload a position, or pulls its quotes, the order flow switches regime for a
while and then switches back: a transient change. Can a firm that sees only the anonymous public book and trades detect
it quickly, and tell it apart from informed trading after news? And is a detection worth anything?

## Files

- `mmsim.py`: the market simulator.
- `mm_qcd.py`: the experiment.
- `report/mm_template.html`, `report/build_mm.py`, `report/mm_text.json`: the results page. `report/mm.html` is
  built from these and is not tracked.
- `data/mm/SP500.csv` and `data/mm/VIXCLS.csv` come from FRED; `fetch_data.sh` downloads them.

Run with `.venv/bin/python mm_qcd.py` and `.venv/bin/python mm_qcd.py --kappa 0.25`. Each run takes about 6 minutes on 4
cores and writes `runs/mm/results.json` and `runs/mm_kappa0.25/results.json` respectively. Then build the page with
`.venv/bin/python report/build_mm.py`.

## The market

- **Instrument and clock.** One-second steps over a 6.5-hour session, on a 0.25-point tick (E-mini-like).
- **Calibration.** The start is the S&P 500 close of 7744.06 on 2026-10-15. The public value is a random walk
  with the 60-day daily volatility (0.69%), an intraday U-shape and slow volatility clustering.
- **News.** A private value jumps at news times, about 10 a day with a standard deviation of 6 points. Informed
  traders then trade toward the new value; these episodes carry label 0.
- **Prices and impact.** Market makers quote around F = P + G, where G moves with signed aggressive volume (lambda =
  0.008 points per contract) and decays toward the private value at eta = 1/300 per second. So flow-driven
  deviations revert, and informed moves do not.
- **Quoting.** Six market makers. MM1-3 quote tight, always a 1-tick market, so the inside spread is one tick. MM4-6
  quote wide, 4 ticks. Each skews its quotes by kappa per contract of inventory.
- **Stress.** Each maker has its own policy: instant, TWAP, passive skew, short gamma, lean, withdrawal. Stress events
  arrive about 15 a day. Label 1 marks the seconds while any maker is under stress.
- **Walk-forward.** 40 days, each opening at the previous close. Days 1-30 train and days 31-40 are held out.

## Detection

**Protocol.** The paper's transient protocol:
- one stress event per test stream at tau1 = 1200 s;
- 300 streams per policy, plus 300 news-only streams;
- detection deadline K = 120 s;
- one pooled threshold per detector for a false-alarm probability alpha;
- the first 120 s of each stream are a warm-up.

**Detectors.**
- **DeepQCD:** the paper's network and training (`original_qcd.fit`) on 3,200 windows of 3,000 s from the training days.
- **WL-CUSUM:** `window_dec_stat` over 60 s, one per policy on that policy's Gaussian LLR direction, each scaled to its
  95% no-stress quantile, combined by taking the maximum.
- **Flow z-score:** |signed volume over 30 s| divided by its training standard deviation.

PD at alpha = 0.05, with market makers skewing their quotes openly (kappa as in mmsim.MM). The news row is the alarm
rate within K after a news jump:

| policy | DeepQCD | WL-CUSUM (per policy) | flow z-score |
|---|---|---|---|
| instant | 0.54 | 0.94 | 0.20 |
| twap | 0.85 | 0.75 | 0.01 |
| skew | 0.92 | 0.06 | 0.00 |
| gamma | 0.95 | 0.51 | 0.05 |
| lean | 0.83 | 0.17 | 0.01 |
| withdraw | 0.11 | 0.01 | 0.00 |
| news | 0.05 | 0.06 | 0.13 |

The same, with discreet skewing (kappa / 4):

| policy | DeepQCD | WL-CUSUM (per policy) | flow z-score |
|---|---|---|---|
| instant | 0.25 | 0.91 | 0.23 |
| twap | 0.89 | 0.35 | 0.02 |
| skew | 0.90 | 0.43 | 0.00 |
| gamma | 0.95 | 0.54 | 0.04 |
| lean | 0.95 | 0.95 | 0.02 |
| withdraw | 0.02 | 0.01 | 0.00 |
| news | 0.07 | 0.00 | 0.11 |

## Edge (alpha = 0.05; ticks, mean ± standard error)

"Price from value at alarm" is the distance from value in the stressed flow's direction, known only to the simulator.
"Fade" is the profit of trading against the stress at the alarm and closing 1 or 5 minutes later, minus 1 tick.

Open skewing:

| detector | stream | n | price from value at alarm | fade 1 min | fade 5 min |
|---|---|---|---|---|---|
| DeepQCD | instant | 162 | +3.2 ± 0.4 | +2.2 ± 0.8 | +3.7 ± 1.9 |
| DeepQCD | twap | 254 | +0.3 ± 0.3 | -1.8 ± 0.6 | +3.2 ± 1.5 |
| DeepQCD | skew | 276 | +0.1 ± 0.2 | -1.3 ± 0.6 | -2.1 ± 1.5 |
| DeepQCD | lean | 248 | +0.0 ± 0.2 | -0.4 ± 0.7 | +1.1 ± 1.6 |
| DeepQCD | news | 14 | -20.2 ± 3.5 | -11.9 ± 2.5 | -11.9 ± 9.0 |
| WL-CUSUM | instant | 283 | +2.2 ± 0.3 | +3.4 ± 0.6 | +4.3 ± 1.4 |
| WL-CUSUM | twap | 226 | +0.5 ± 0.4 | -1.3 ± 0.7 | +3.9 ± 1.6 |
| WL-CUSUM | skew | 18 | +1.1 ± 2.5 | -2.7 ± 3.2 | -3.6 ± 6.1 |
| WL-CUSUM | lean | 52 | +0.2 ± 0.8 | -1.3 ± 1.9 | +0.7 ± 3.6 |
| WL-CUSUM | news | 19 | -12.6 ± 2.0 | -12.4 ± 3.2 | -10.2 ± 5.6 |
| flow z-score | instant | 59 | +6.9 ± 1.3 | +6.0 ± 1.5 | +5.4 ± 3.3 |
| flow z-score | twap | 3 | -11.9 ± 2.0 | -15.3 ± 4.6 | -24.7 ± 7.6 |
| flow z-score | skew | 0 | - | - | - |
| flow z-score | lean | 3 | +14.1 ± 6.5 | +2.7 ± 9.5 | +20.7 ± 25.4 |
| flow z-score | news | 39 | -13.5 ± 1.5 | -9.6 ± 1.7 | -9.2 ± 3.6 |

Discreet skewing:

| detector | stream | n | price from value at alarm | fade 1 min | fade 5 min |
|---|---|---|---|---|---|
| DeepQCD | instant | 74 | +2.9 ± 0.5 | +1.4 ± 1.1 | -4.7 ± 2.8 |
| DeepQCD | twap | 268 | +0.1 ± 0.2 | -3.3 ± 0.7 | -1.9 ± 1.5 |
| DeepQCD | skew | 271 | -0.1 ± 0.2 | -1.8 ± 0.7 | -4.9 ± 1.6 |
| DeepQCD | lean | 286 | +0.4 ± 0.3 | -0.6 ± 0.6 | +0.1 ± 1.6 |
| DeepQCD | news | 22 | -13.8 ± 2.1 | -9.0 ± 2.6 | -7.9 ± 5.5 |
| WL-CUSUM | instant | 273 | +3.9 ± 0.3 | +1.5 ± 0.6 | -1.2 ± 1.5 |
| WL-CUSUM | twap | 106 | -0.0 ± 0.5 | +0.3 ± 1.2 | +0.9 ± 2.3 |
| WL-CUSUM | skew | 128 | -1.3 ± 0.7 | -2.2 ± 1.0 | -4.7 ± 2.2 |
| WL-CUSUM | lean | 284 | +0.3 ± 0.3 | -0.0 ± 0.6 | -0.1 ± 1.5 |
| WL-CUSUM | news | 0 | - | - | - |
| flow z-score | instant | 68 | +6.7 ± 1.1 | +4.6 ± 1.4 | -2.4 ± 3.3 |
| flow z-score | twap | 6 | -8.6 ± 4.9 | +1.0 ± 4.6 | +8.7 ± 11.5 |
| flow z-score | skew | 1 | -65.9 | -29.0 | -51.0 |
| flow z-score | lean | 5 | +8.7 ± 5.2 | +7.8 ± 6.7 | +41.8 ± 10.9 |
| flow z-score | news | 32 | -13.3 ± 1.6 | -10.2 ± 1.8 | -5.8 ± 3.9 |

## Findings

- **Detection.** DeepQCD detects the slow and passive stress policies (TWAP, skew, lean) and the short-gamma dependence
  change best. A CUSUM given each policy's Gaussian model detects instant unwinds best. No detector sees a quote
  withdrawal.
- **News.** DeepQCD alarms after 5-7% of news jumps, against 13% for the flow rule.
- **Detection is not profit.**
  - Passive unwinds leave the price at value.
  - Instant unwinds leave it 2-4 ticks away, and fading earns about 4 ticks over 5 minutes with open skewing. With
    discreet skewing that edge is not distinguishable from zero.
  - Fading after news loses about 10-12 ticks.

## Development notes (what changed and why)

1. **Spread tell removed.** The first version let the tight makers quote 1-2 ticks wide. A stressed maker's clamped
   quotes then pinned the spread at one tick, which gave stress away. Tight makers now always quote a one-tick
   market, as in E-mini futures.
2. **Fairer CUSUM.** One pooled Gaussian model for all policies gave the CUSUM no chance. One model per policy, combined
   by an unscaled maximum, let the instant-unwind statistic set every threshold. Each is now scaled to its own
   no-stress quantile.
3. **Cleaner edge measure.** The edge was first measured as the raw price move after the alarm, which is dominated by
   ±24 ticks of 5-minute noise. It is now the distance from value at the alarm, plus the net result of fading.

## Limits

- **The edge is partly built in.** The impact model (lambda, eta) is assumed, and the reversion follows from it.
- **Only the level and volatility are calibrated.** Order rates, sizes, the market makers and their policies are
  chosen, not fitted.
- **Next step: real labelled forced unwinds.** Sources include crypto exchange liquidation streams (e.g. Binance
  `forceOrder`) with public order books, and LOBSTER or CME market-by-order data. Train here, test there.

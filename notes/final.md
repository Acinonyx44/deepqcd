# Final studies: bank alarms at scale, and prediction markets

The two use cases that worked (laundering and card fraud) finalised and scaled, and the prediction-market work done in
full. Page: `report/build_final.py` builds `report/final.html` from `runs/bank` and `runs/pm`.

## Files

- `bank_data.py` loads five data sets into cached per-entity streams (`data/cache/<name>.npz`).
  - AMLworld HI-Small, LI-Small and HI-Medium come from Hugging Face mirrors (`OsamaMIT/IBM-AML-HI-Small`,
    `LordNR/AMLGraphX-IBM-AML`).
  - TabFormer and Sparkov (`santosh3110/credit_card_fraud_transactions`). The Sparkov mirror stops at Excel's 1,048,575
    rows and rounds card numbers, so a card is identified by its holder.
  - AMLworld is read with pyarrow and numpy (32-bit indices, early deletes), so HI-Medium (61 million account steps)
    builds in 4 minutes with 11 GB at peak. Its HI-Small features match `usecase_amlworld.py`.
- `bank_final.py <data> [--curve]` is the final protocol below. It writes `runs/bank/<data>.json`, plus
  `<data>_stream.npz` (streaming scores) and `<data>_curve.json`.
- `polymarket_fetch.py --full 2250` extended the trade data from 750 to 3,000 markets: 24.0 million trades by 1.54 million
  wallets, two-outcome markets with $1M-$30M volume closed January 2025 - October 2026.
- `pm_final.py --target metaorder|churn|informed|jump [--wallets]` writes `runs/pm/<target>[_wallets].json` and the models.
- `pm_replay.py` writes `runs/pm/replay.json`.
- `runs/run_final.sh` (resumable) and `runs/run_final2.sh` (the second bank pass, with fine streaming curves).

## Protocol (what changed against the earlier studies)

1. **Three periods.** Training before p0, calibration [p0, p1), test [p1, p2). Every threshold comes from the calibration
   period (Eq. 11 there) and is applied unchanged to the test period, which reports the realized PFA. Eq. 11 on the test
   streams is kept as the oracle.
   - AMLworld: train to 5 Sep, calibrate 6-7 Sep, test 8-10 Sep. HI-Medium: train to 9 Sep, calibrate 10-12, test 13-16.
     The days after these hold only the tails of laundering schemes and are left out.
   - TabFormer: train before 2014, calibrate 2014-15, test 2016-20. Sparkov: train to Aug 2012, calibrate Sep-Oct,
     test Nov 2012 - Mar 2013.
   - Polymarket: markets by closing date, 60% train, 15% calibrate, 25% test.
2. **Five DeepQCD seeds and their ensemble** (the mean output).
3. **Bootstrap intervals** (1,000 resamples) for PD, PFA and the paired PD difference with gradient boosting.
4. **Gradient boosting** with lagged inputs (LightGBM, up to 4,000 trees).
5. **Streaming (banks).** Every account or card is scored at every transaction of the calibration and test periods from
   its last PRE + 1 transactions.
   - Thresholds are set in the calibration period for 0.5-50 false alert-days per 1,000 active entity-days.
   - The test period reports the realized rate, the share of episodes alerted, and the share of their money at or after
     the first alert.
   - The curve places thresholds at false-alert rates from 0.1 to 1,000 per 1,000. The first pass used quantiles of all
     entity-days, which left the low rates unresolved; it is kept in `runs/bank/v1`.
6. **Learning curves** (`--curve`): DeepQCD (two seeds) and gradient boosting on 10%, 30% and 100% of the training events.

## Results: banks (window level, calibrated thresholds)

PD on test, the DeepQCD ensemble against gradient boosting; realized PFA in brackets.

| Data | test events | α = 0.01 | α = 0.05 | DeepQCD - GB at 0.01 |
|---|---|---|---|---|
| AMLworld HI-Small | 967 | 0.55 (0.017) vs 0.48 | 0.59 vs 0.55 | +0.08 [+0.05, +0.10] |
| AMLworld LI-Small | 828 | 0.34 (0.015) vs 0.30 | 0.40 vs 0.40 | +0.04 [+0.02, +0.06] |
| AMLworld HI-Medium | 6,372 | 0.38 (0.011) vs 0.26 | 0.54 vs 0.53 | +0.13 [+0.12, +0.14] |
| TabFormer | 1,677 | 0.69 (0.009) vs 0.73 | 0.74 vs 0.78 | -0.03 [-0.05, -0.02] |
| Sparkov | 163 | 0.99 vs 0.99 | 0.99 vs 0.98 | 0.00 |

- The logistic model, the CUSUM and the rules are far behind on every bank (rules combined at most 0.06 at α = 0.01, 0.12 on
  Sparkov).
- **Seeds.** Single seeds spread by up to 0.13 at α = 0.01; the ensemble is at or above the average seed.
- **Laundering patterns.** Laundering in the listed patterns (fan-in, cycle, ...) is caught at 0.54-0.97 at α = 0.01.
  Laundering in no listed attempt ("unlisted", half the accounts) is caught at 0.03-0.30. Non-ACH laundering is never
  caught.

## Results: banks (streaming)

Share of episodes alerted, read off the test curve at the same realized number of false alert-days per 1,000 active
entity-days (DeepQCD / gradient boosting).

| Data | 0.5 | 1 | 2 | 5 | 20 | 50 |
|---|---|---|---|---|---|---|
| AMLworld HI-Small | 0.13 / 0.05 | 0.17 / 0.10 | 0.22 / 0.18 | 0.32 / 0.31 | 0.69 / 0.62 | 0.74 / 0.75 |
| AMLworld LI-Small | 0.05 / 0.04 | 0.08 / 0.07 | 0.11 / 0.11 | 0.20 / 0.20 | 0.50 / 0.49 | 0.58 / 0.63 |
| AMLworld HI-Medium | 0.30 / 0.28 | 0.35 / 0.36 | 0.40 / 0.43 | 0.48 / 0.51 | 0.69 / 0.68 | 0.78 / 0.76 |
| TabFormer cards | 0.68 / 0.66 | 0.75 / 0.73 | 0.80 / 0.77 | 0.85 / 0.82 | 0.92 / 0.91 | 0.95 / 0.95 |
| Sparkov cards | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |

- **HI-Small.** DeepQCD catches more at the low alert rates a bank can staff, up to 2.4 times as much at 0.5 per
  1,000.
- **LI-Small and HI-Medium.** The two are level within 0.05. The HI-Medium window-test lead does not carry over to
  whole histories.
- **TabFormer.** DeepQCD is 0.00-0.03 ahead at every rate.
- **Drift.** With thresholds set for 5 per 1,000, DeepQCD realized 7.8 (HI-Small) and 7.2 (LI-Small) on test, against
  4.9 and 5.5 for gradient boosting.
- **Timing.** The median alert comes at the first laundering transfer; on TabFormer, after one fraudulent transaction.

## Results: learning curves

| Data | share of events | DeepQCD α = 0.01 | GB α = 0.01 | DeepQCD α = 0.05 | GB α = 0.05 |
|---|---|---|---|---|---|
| HI-Medium | 10% | 0.35 | 0.32 | 0.53 | 0.52 |
| HI-Medium | 30% | 0.35 | 0.31 | 0.54 | 0.54 |
| HI-Medium | 100% | 0.34 | 0.26 | 0.52 | 0.53 |
| TabFormer | 10% | 0.49 | 0.51 | 0.69 | 0.67 |
| TabFormer | 30% | 0.69 | 0.66 | 0.75 | 0.77 |
| TabFormer | 100% | 0.72 | 0.73 | 0.76 | 0.78 |

On HI-Medium the network's strict-level PD is flat from 10% of the events, while gradient boosting falls as the data
grow. The lead is not the network gaining from more labels.

## Results: Polymarket (calibrated thresholds, 750 test markets)

| Target | test events | DeepQCD α = 0.01 | GB α = 0.01 | DeepQCD α = 0.05 | GB α = 0.05 | CUSUM α = 0.05 |
|---|---|---|---|---|---|---|
| Metaorder | 1,891 | 0.26 | 0.23 | 0.43 | 0.38 | 0.32 |
| Churn | 563 | 0.44 | 0.37 | 0.46 | 0.46 | 0.31 |
| Informed flow | 1,487 | 0.06 | 0.08 | 0.17 | 0.19 | 0.15 |
| Informed flow, wallet features | 1,487 | 0.05 | 0.06 | 0.17 | 0.20 | 0.16 |
| Hour before a price jump | 826 | 0.13 | 0.04 | 0.14 | 0.15 | 0.13 |

- **Metaorders.** +0.05 [+0.03, +0.07] at α = 0.05. The earlier 750-market study was a tie.
- **Informed flow.** 22.6% of $2,000+ wallet episodes are followed by a 10-cent move their way within 24 hours, against
  19.0% the other way: mostly luck.
  - Only large episodes stand out ($20k+: 0.48 vs 0.31).
  - Wallet features add nothing beyond the rule "proven-wallet flow, 30 minutes" (0.19).
- **Jumps.** Sports markets 0.31, others 0.01 (DeepQCD, α = 0.05). News jumps give no warning.

## Results: market-making replay (748 test markets, 5-minute markouts)

- **Always quote.** +4.95 cents per $100 [-3.3, +13.8]. The earlier 298-market sample's -12.7 does not hold up.
- **Alarms.** Every alarm strategy's paired difference with always quoting includes zero. Pulling quotes on jump alarms at
  a 10% alarm rate costs money.
- **Oracles.** Pulling the hit side during true metaorders adds +6 to +25 cents per $100; during true informed episodes,
  +40 to +54.

## Limits

- **Synthetic banks.** AMLworld, TabFormer and Sparkov are synthetic. Polymarket is real.
- **Streaming simplifications.** Streaming counts an alert-day as false whenever no flagged span covers it. Alerts on an
  account before its first laundering transfer count as false.
- **Replay simplifications.** The replay ignores queue position and liquidity rewards, and its P&L is heavy-tailed near
  resolution.

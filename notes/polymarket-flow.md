# Polymarket order flow labelled by wallet: metaorders and churn

**Idea.** Polymarket settles on chain, so every trade carries its wallet. Episodes a market maker cares about can be
labelled exactly from public data, while the detectors see only anonymous flow (side, size, time, price). This is the
labelled lab that anonymous venues (Kalshi, futures) do not offer.

## Files

- `polymarket_fetch.py --full N`: every trade of N random two-outcome markets with $1M-$30M volume that closed in 2025
  or later. Each call adds markets. We used 250 + 500 = 750, of which 745 have at least 500 trades: 5.8 million trades
  in `data/polymarket/full/` (not tracked). It takes about 10 minutes per 500 markets.
- `polymarket_meta.py [--target metaorder|churn] [--clock minute|order]`: the experiment. It takes 1-2 minutes and
  writes `runs/polymarket_meta/{minute,churn_minute,order}/results.json`.

## Labels (per wallet)

Consecutive trades of one wallet at most 10 minutes apart, lasting at least 1 minute.

- **Metaorder.** At least 5 trades and $2,000, all in one direction. There are 8,580.
  - Over its life the price moves 2.6 cents in its direction (SE 0.2), 1.0 cent of that in the second half.
  - It doesn't reliably revert: -0.3 cents in the next 30 minutes (SE 0.3).
- **Churn.** At least 6 trades, $2,000 gross, 4 changes of side, and net at most 20% of gross. There are 1,172, with
  a median of 24 trades and 30 minutes.

## Protocol

- **Features.** Nine one-minute features: trades, signed flow, gross, largest trade, one-sidedness over 5 minutes,
  share of similar same-side sizes over 10 minutes, gap regularity, price change, and trades over 10 minutes.
- **Change streams.** 180 quiet minutes (no episode of that kind), then the episode (capped at 60 minutes).
- **No-change streams.** 3,576 random quiet 4-hour windows.
- **Split.** The 447 markets that closed first train; the other 298 test.
- **Detectors.**
  - DeepQCD: the original network and training, three seeds.
  - WL-CUSUM: `window_dec_stat` over 10 minutes on the Gaussian-LLR projection.
  - Two rules: absolute signed flow over 5 minutes, and trades over 10 minutes.

## Results

### Churn: DeepQCD wins

206 test sessions. The first three columns are PD at the given PFA.

| Detector | PFA 0.01 | 0.05 | 0.1 | PD at 0.05 for sessions of $20k+ | Share of all churn time caught |
|---|---|---|---|---|---|
| DeepQCD (3 seeds) | **0.35** | **0.42** (0.40, 0.45, 0.42) | 0.42 | ~0.61 | 0.26-0.32 |
| WL-CUSUM | 0.21 | 0.30 | 0.30 | 0.31 | 0.22 |
| Trades, 10 min | 0.19 | 0.26 | 0.26 | 0.26 | 0.19 |
| Signed flow, 5 min | 0.03 | 0.19 | 0.23 | 0.26 | 0.13 |

### Metaorders: tie, and the CUSUM is worth more

865 test metaorders. The first three columns are PD at the given PFA.

| Detector | PFA 0.01 | 0.05 | 0.1 | Delay | Cents saved per metaorder |
|---|---|---|---|---|---|
| DeepQCD (3 seeds) | 0.15 | 0.38 (0.40, 0.34, 0.39) | 0.45 | 3-4 min | about 0.04 |
| WL-CUSUM | 0.14 | 0.33 | 0.37 | 2 min | **0.17** |
| Trades, 10 min | 0.07 | 0.22 | 0.29 | 4 min | 0.13 |
| Signed flow, 5 min | 0.05 | 0.11 | 0.19 | 2 min | 0.08 |

**Data volume.** With 250 markets (150 training), DeepQCD and the CUSUM were level at 0.31 at PFA 0.05. Tripling the
data gave DeepQCD its small lead.

**The order clock fails.** With one step per taker order instead of per minute, every detector has PD at most 0.09 at
PFA 0.05. A metaorder's few child orders (for example 7) are spread among dozens of other orders. Their sizes look
like everyone else's, so per-order steps add noise rather than signal.

## Reading

- **DeepQCD wins on churn.** It is a signature of several cues: busy, but balanced, alternating and repetitive. The
  CUSUM and the trade count see only "busy".
- **It ties on metaorders.** These are mostly one statistic, signed flow, and the CUSUM catches the price-moving ones
  first.
- **The market maker's use is to run both.** Step back when a metaorder starts (CUSUM); keep quoting through churn
  (DeepQCD), since the churner pays the spread.
- **Labels can be leaned on for training.** The labels are heuristics on one wallet. Wash trading across several
  wallets, or a metaorder split across wallets, is not labelled, so some "false alarms" are real episodes.

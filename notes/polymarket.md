# Polymarket: does money move before the news?

**Question.** In deadline markets ("X by <date>?"), a YES resolution happens when the event happens, at a random time,
and the price then jumps to about 1. If some traders learn of the event first, their buying should show in the hours
before the jump. So each market is a stream with or without a transient change that ends at the public move. Can a
detector alarm in the hours before the public move without alarming elsewhere? And are its alarms worth trading?

## Files

- `polymarket_fetch.py`: downloads hourly YES prices and every taker trade of at least $1,000, with its wallet. It
  writes one file per market to `data/polymarket/m/` (not tracked).
  - First run `.venv/bin/python polymarket_fetch.py --list`. It lists closed markets from the gamma API with keyset
    pagination (`/markets/keyset`, cursor `after_cursor`) and writes `data/polymarket/deadline_markets.json`: resolved
    Yes/No markets with "by" or "before" in the question, open at least 3 days, with $100,000 volume or more.
  - Then run `.venv/bin/python polymarket_fetch.py 100000`, which takes about 7 minutes.
- `polymarket_qcd.py`: the experiment. Run `.venv/bin/python polymarket_qcd.py`, which takes about 2.5 minutes on 4
  cores. It writes `runs/polymarket/results.json`.
- `report/followup_template.html`, `report/build_followup.py`: the results page (with the rare-event experiment).
  Build it with `.venv/bin/python report/build_followup.py`.

## APIs

All are public and need no key.

- **gamma `/markets`.** Limit 100. Offset stops working past about 2,500, so use keyset pagination.
- **CLOB `/prices-history`.** Takes `market=<YES token id>&startTs&endTs&fidelity=60`. Windows must be 14 days or
  shorter.
- **data API `/trades`.** Takes `market=<conditionId>&filterType=CASH&filterAmount=1000&limit=1000&end=<ts>`. Results
  come newest first, and `offset` stops at 10,000, so page backwards in time with `end`. The side is the taker's;
  `outcomeIndex` 0 is YES.

## Universe and labels

- **Markets.** 2,083 closed markets with at least $100,000 volume, from February 2023 to October 2026. The question
  must contain "by" or "before" plus a date-like word; ranges ("between") are excluded. Of these, 582 resolved YES.
  The data come to about 3.1 million market-hours.
- **tau_pub.** The first hour after which the YES price stays at or above 0.8 until the market closes.
- **Change stream.** The 168 hours before tau_pub. The change window is the last W = 24 hours. Only surprises count:
  the price stayed below 0.5 for the first 144 hours.
- **No-change streams.** Consecutive 168-hour windows of NO markets, and the earlier windows of YES markets.
- **Events.** Markets that went public within 12 hours of each other form one event (ladders). That gives 64 events in
  105 markets. The test uses one stream per event, its largest market. Training uses all of an event's markets plus
  7 copies of each stream ending 3 to 21 hours earlier.
- **Split.** Markets that closed before 2026-02-01 are training: 970 markets, 34 events, 7,706 no-change streams. The
  rest are test: 1,113 markets, 30 events, 9,221 no-change streams.
- **Rates.** PFA is the probability of an alarm from hour 24 to tau_pub - 24 in any stream. PD is the probability that
  an event stream's first alarm falls in the window.

## Results

PD at the threshold that keeps PFA at or below the given level. For DeepQCD, the mean of three seeds with each seed in
brackets:

| Detector | PFA 0.01 | 0.05 | 0.1 | median lead at 0.05 |
|---|---|---|---|---|
| DeepQCD, all features | 0.01 | 0.40 (0.47, 0.20, 0.53) | 0.53 | 9 h |
| DeepQCD, flow only | 0.00 | 0.30 (0.67, 0.17, 0.07) | 0.48 | 1 h |
| WL-CUSUM, all features | 0.03 | 0.17 | 0.37 | 1 h |
| WL-CUSUM, flow only | 0.03 | 0.03 | 0.10 | 1 h |
| Price level | 0.03 | 0.10 | 0.27 | 13 h |
| Price rise over 24 h | 0.00 | 0.43 | 0.47 | 2 h |

With 30 events, the standard error of a PD near 0.4 is about 0.09.

**Large trades per hour (test data):**

| | no event | event market, days 2-7 before | event market, last 24 h |
|---|---|---|---|
| hours with a large trade | 5% | 12% | 20% |
| large trades per hour | 0.11 | 0.57 | 6.77 |
| large-trade $ per hour | $501 | $3,380 | $35,001 |
| net toward YES | -0.27 | -0.36 | +0.03 |
| net toward YES, wallets new to the market | -0.17 | -0.15 | -0.03 |
| net toward YES, wallets new to Polymarket | -0.07 | -0.04 | -0.06 |
| mean YES price | 0.20 | 0.16 | 0.25 |

**Backtest.** Buy YES at the first alarm in each test market while the price is below 0.8, hold to resolution, and
pay 1 cent. Thresholds are the PFA 0.05 ones. Profit is in cents per $1 contract:

| Entry | Trades | Profit (cents) |
|---|---|---|
| DeepQCD, all features | 470 | -4.6 ± 1.8 |
| DeepQCD, flow only | 695 | +2.6 ± 1.4 |
| WL-CUSUM, all features | 275 | +0.9 ± 2.4 |
| WL-CUSUM, flow only | 256 | +1.9 ± 2.3 |
| Price rise over 24 h | 231 | -16.7 ± 3.1 |
| Random hour | 1,034 | +5.7 ± 1.1 |

## Findings

1. **Few events.** Only 64 surprise events, concentrated in 2026 and in a few themes (Iran, Venezuela,
   Russia-Ukraine). Training has 34 of them, and the DeepQCD seeds disagree by more than the detectors do.
2. **Detection is no better than the price.** DeepQCD is level with a rule on the 24-hour price rise. The flow-only
   detectors are weaker on average.
3. **Activity rises, direction follows the price.** Event markets are busier, and much busier in the last day. The net
   direction of large trades turns from NO-leaning to balanced as the price rises. New and fresh wallets do not lean
   toward YES.
4. **No tradable edge.** No detector's alarms beat buying at a random hour. Price run-up alarms lose because most
   run-ups are scares. In this period YES was cheap overall: many of the 2026 events did happen.
5. **Verdict.** At hourly resolution with trades of $1,000 or more, there is no evidence of exploitable informed flow
   before public events. Next steps:
   - minute-level data with all trades, around the reported insider cases;
   - wallet-level features, such as a wallet's track record;
   - pooling with Kalshi;
   - using activity surges for surveillance rather than trading.

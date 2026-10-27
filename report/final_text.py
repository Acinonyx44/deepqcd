"""The prose of report/final.html, keyed by the placeholders of final_template.html (build_final.py fills them in)."""

PM_TEXT = '''
  <p>Polymarket settles on chain, so every trade carries its wallet, and episodes a market maker or the venue cares about can be labelled exactly from public data. The data are every trade of 3,000 two-outcome markets with $1M–$30M volume that closed between January 2025 and October 2026: 24.0 million trades by 1.54 million wallets. Markets are split by closing date: 1,800 train (to April 2026), 450 calibrate (April–June 2026) and 750 test (June–October 2026). The clock is one minute; the detectors see the nine anonymous one-minute features of the earlier study (trades, signed and gross flow, largest trade, one-sidedness, repeated sizes, regularity, price change, activity).</p>
  <ul class="tight">
    <li><b>Metaorder.</b> One wallet works a large order in one direction: at least 5 trades and $2,000, trades at most 10 minutes apart.</li>
    <li><b>Churn.</b> One wallet trades back and forth: at least 6 trades and 4 changes of side, net at most 20% of gross.</li>
    <li><b>Informed flow.</b> A wallet episode of at least $2,000 in one direction after which the price is at least 10 cents further its way 24 hours later (the market's final value if it closed first). 22.6% of the 306,304 such episodes qualify, against 19.0% that moved 10 cents the other way, so the label is mostly luck with a small edge. A second detector also sees four features the venue can compute from wallets: flow from wallets first seen in the last week, the track record of the wallets trading, flow from proven wallets, and how concentrated the volume is.</li>
    <li><b>Price jump.</b> The hour before the price moves at least 15 cents within 30 minutes (from a level between 5 and 95 cents) and is still 10 cents away two hours later.</li>
  </ul>
  <p class="verdict"><b>Reading.</b>
    <ul class="tight">
      <li><b>Metaorders: win at scale.</b> With 19,421 training metaorders, the DeepQCD ensemble finds 0.43 at α = 0.05 against 0.38 for gradient boosting (difference +0.05, interval +0.03 to +0.07) and 0.32 for the CUSUM. On the 750-market sample of the earlier study the two tied. The ensemble beats each of its five seeds (0.38–0.41), and the lead is largest for $5k–20k orders (0.48 against 0.39).</li>
      <li><b>Churn: win at the strict level.</b> 0.44 against 0.37 at α = 0.01; both 0.46 at α = 0.05.</li>
      <li><b>Informed flow: hard for every detector.</b> At α = 0.05 the detectors find 0.15–0.20, three to four times chance, and gradient boosting is slightly ahead. Wallet features add nothing that the one-line rule "flow from proven wallets over 30 minutes" (0.19) does not. Only large episodes stand out: of those of $20,000 or more, DeepQCD finds 0.48 against 0.31.</li>
      <li><b>Price jumps: only in sports.</b> DeepQCD finds 0.13 at α = 0.01 against 0.04 for gradient boosting, and all detectors 0.12–0.15 at α = 0.05. In sports markets DeepQCD finds 0.31 of jumps an hour ahead; elsewhere 0.01. Jumps on news give no warning in the flow.</li>
    </ul>
  </p>
'''

R_TEXT = '''
  <p>We are a market maker that takes the other side of every taker trade in the 748 test markets while it quotes, scored by the markout 5 minutes (and 30 minutes) later. Each alarm pulls the side being hit (or both sides, for churn and jumps) while it is on; its threshold makes it ring in 2% (or 10%) of the quiet decision minutes of the 448 calibration markets. The oracles pull the hit side exactly during true metaorders or true informed episodes.</p>
  <p class="verdict"><b>Reading.</b>
    <ul class="tight">
      <li><b>Quoting against every taker roughly breaks even</b> at scale: +5.0 cents per $100 of volume at 5 minutes (interval −3.3 to +13.8). The loss of the earlier 298-market sample (−12.7) does not hold up.</li>
      <li><b>No alarm changes P&amp;L by an amount distinguishable from zero.</b> The paired differences with always quoting all have intervals that include zero, except stepping back on jump alarms at 10%, which costs money.</li>
      <li><b>The ceilings are real.</b> Pulling the hit side exactly during metaorders would add 6 to 25 cents per $100, and during informed episodes 40 to 54. The detectors find too few of these episodes (0.43 of metaorders and about 0.2 of informed episodes at 5% false alarms) to collect it.</li>
      <li><b>Heavy tails.</b> Markouts near resolution move by tens of cents, so P&amp;L per market is noisy; the intervals come from 1,000 bootstrap resamples of the test markets.</li>
    </ul>
  </p>
'''

TEXT = {'PM_TEXT': PM_TEXT, 'R_TEXT': R_TEXT}

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

DATA_TEXT = '''
  <p>Five public, synthetic bank data sets.</p>
  <ul class="tight">
    <li><b>Laundering: IBM AMLworld</b> (Altman et al., NeurIPS 2023), three banks. HI-Small has 5.1 million transfers between 515,000 accounts; LI-Small has 6.9 million transfers with ten times fewer laundering transfers; HI-Medium has 31.9 million transfers between 2.08 million accounts. An account's stream has one step per transfer it sends or receives (15 features), and the change runs from its first laundering transfer to one after its last.</li>
    <li><b>Card fraud: IBM TabFormer</b> (24.4 million transactions of 6,139 cards, 1991–2020) and the <b>Sparkov</b> simulator (1.05 million transactions of 965 cards). A card's stream has one step per transaction, and the change is a fraud burst.</li>
    <li><b>Periods.</b> AMLworld trains on the days before 6 September (HI-Medium: 10 September), calibrates on the next two days (three) and tests on the last three days (four) that carry normal traffic; the days after those, which hold only the ends of laundering schemes, are left out. TabFormer trains before 2014, calibrates on 2014–15 and tests on 2016–20. Sparkov trains to August 2012, calibrates on September–October and tests from November 2012 to March 2013.</li>
  </ul>
'''

W_TEXT = '''
  <p class="verdict"><b>Reading.</b>
    <ul class="tight">
      <li><b>Laundering: DeepQCD wins at the strict level.</b> At α = 0.01 the ensemble finds 0.55 against 0.48 for gradient boosting on HI-Small, 0.34 against 0.30 on LI-Small and 0.38 against 0.26 on HI-Medium; every interval of the difference excludes zero. At α = 0.05 the two are within 0.05 of each other. The logistic model, the CUSUM and the rules are far behind.</li>
      <li><b>Card fraud: gradient boosting is level or slightly ahead.</b> On TabFormer it finds 0.73 against 0.69 at α = 0.01 (difference −0.03, interval −0.05 to −0.02). On Sparkov every learner finds 0.99: the simulator's fraud is too plain to separate them.</li>
      <li><b>The thresholds carry over.</b> The PFA realized on test stays near the target: 0.006–0.017 at α = 0.01. The one miss is DeepQCD on TabFormer at α = 0.05, whose threshold turns out too strict (PFA 0.018); on the oracle curve the two learners tie there (0.79 and 0.78).</li>
      <li><b>The ensemble matters.</b> Single seeds spread by up to 0.13 at α = 0.01 (TabFormer 0.61–0.74, LI-Small 0.24–0.36); the ensemble is at or above the average seed on every data set.</li>
    </ul>
  </p>
'''

L_TEXT = '''
  <p>DeepQCD (two seeds) and gradient boosting retrained on 10%, 30% and 100% of the training events, with thresholds again from the calibration period.</p>
  <ul class="tight">
    <li><b>HI-Medium.</b> DeepQCD's PD at α = 0.01 is already 0.35 with 10% of the events and stays there (0.34 with all of them), while gradient boosting falls from 0.32 to 0.26 as the data grow. The lead on the largest bank comes from gradient boosting losing precision at the top of its scores, not from the network gaining with more labels. At α = 0.05 the two are level at every size (0.52–0.54).</li>
    <li><b>TabFormer.</b> Both improve with more fraud (DeepQCD 0.49, 0.69, 0.72 at α = 0.01; gradient boosting 0.51, 0.66, 0.73) and stay level.</li>
  </ul>
'''

S_TEXT = '''
  <p>The window test above asks whether an alarm rings during an event that follows 15 (cards: 30) normal transactions. A bank runs the alarm on every account at every transaction instead, and counts alerts per day. Here each detector scores every transaction of the calibration and test periods from the account's or card's last PRE + 1 transactions (HI-Medium: all laundering accounts and a fifth of the others, weighted back; TabFormer: half of the clean cards).</p>
  <p class="verdict"><b>Reading.</b>
    <ul class="tight">
      <li><b>HI-Small: DeepQCD catches more at the alert rates a bank can staff.</b> At 0.5 false alert-days per 1,000 active account-days it alerts on 0.13 of laundering accounts against 0.05 for gradient boosting, and at 1 per 1,000 on 0.17 against 0.10; the two meet at 5–10 per 1,000.</li>
      <li><b>LI-Small and HI-Medium: level.</b> The two learners stay within 0.05 of each other from 0.5 to 50 per 1,000 (HI-Medium at 1 per 1,000: 0.35 against 0.36). On HI-Medium the window test's lead at α = 0.01 does not carry over to whole histories.</li>
      <li><b>Card fraud: DeepQCD slightly ahead.</b> On TabFormer it alerts on 0.75 of fraud bursts against 0.74 at 1 per 1,000 and 0.85 against 0.83 at 5 per 1,000; on Sparkov both reach 0.99.</li>
      <li><b>Alerts come early.</b> When an account is caught, the median alert comes at its first laundering transfer; on TabFormer, after one fraudulent transaction. On TabFormer, 67% of the fraud money in the test bursts came at or after an alert (missed bursts count as 0).</li>
      <li><b>Alert volume drifts.</b> With thresholds set in the calibration period for 5 false alert-days per 1,000, DeepQCD let through 7.8 on the HI-Small test days and 7.2 on LI-Small, against 4.9 and 5.5 for gradient boosting. A deployed network's alert volume needs watching.</li>
    </ul>
  </p>
'''

KINDS_CAP = ('Streaming on HI-Medium: share of laundering accounts alerted while active, by the laundering pattern they take part in, '
             'at the threshold set for 5 false alert-days per 1,000 (accounts in brackets). Laundering in the listed patterns is '
             'caught at about 0.8 by both learners; accounts whose laundering is in no listed attempt (half of them) are almost '
             'never caught.')

SUMMARY = '''
  <p>The two applications that worked in the earlier studies, laundering and card fraud, are finalised and scaled to five bank data sets (up to 32 million transfers between 2 million accounts), and the prediction-market studies are scaled to every trade of 3,000 Polymarket markets (24 million trades). Every threshold is now set on a calibration period and tested on a later one; DeepQCD is a five-seed ensemble; gradient boosting with lagged inputs is the main competitor; every number has a bootstrap interval; and the bank alarms are also run as a bank would run them, on every account at every transaction.</p>
  <div class="tablewrap"><table class="wraptab">
    <tr><th>Study</th><th>Events (test)</th><th>DeepQCD vs gradient boosting</th><th>Verdict</th></tr>
    <tr><td>Laundering, AMLworld HI-Small</td><td class="num">967</td><td>0.55 vs 0.48 at PFA 0.01; streaming 0.17 vs 0.10 at 1 false alert-day per 1,000</td><td><b>DeepQCD wins</b></td></tr>
    <tr><td>Laundering, AMLworld LI-Small</td><td class="num">828</td><td>0.34 vs 0.30 at PFA 0.01; streaming level</td><td>win at PFA 0.01, else level</td></tr>
    <tr><td>Laundering, AMLworld HI-Medium</td><td class="num">6,372</td><td>0.38 vs 0.26 at PFA 0.01; streaming level</td><td>win at PFA 0.01, else level</td></tr>
    <tr><td>Card fraud, TabFormer</td><td class="num">1,677</td><td>0.69 vs 0.73 at PFA 0.01; streaming 0.85 vs 0.83 at 5 per 1,000</td><td>level</td></tr>
    <tr><td>Card fraud, Sparkov</td><td class="num">163</td><td>0.99 vs 0.99</td><td>both at the ceiling</td></tr>
    <tr><td>Polymarket metaorders</td><td class="num">1,891</td><td>0.43 vs 0.38 at PFA 0.05</td><td><b>DeepQCD wins</b></td></tr>
    <tr><td>Polymarket churn</td><td class="num">563</td><td>0.44 vs 0.37 at PFA 0.01; 0.46 vs 0.46 at 0.05</td><td>win at PFA 0.01</td></tr>
    <tr><td>Polymarket informed flow</td><td class="num">1,487</td><td>0.17 vs 0.19 at PFA 0.05 (with wallet features 0.17 vs 0.20)</td><td>not detectable</td></tr>
    <tr><td>Polymarket price jumps</td><td class="num">826</td><td>0.13 vs 0.04 at PFA 0.01; at 0.05, sports 0.31 and other markets 0.01</td><td>sports only</td></tr>
    <tr><td>Polymarket market making</td><td class="num">748 markets</td><td>no alarm changes P&amp;L measurably; perfect metaorder knowledge would add 6–25¢ per $100</td><td>no edge yet</td></tr>
  </table></div>
  <ul class="tight">
    <li><b>DeepQCD earns its place at strict alarm levels on sequential patterns:</b> laundering accounts on all three AMLworld banks in the window test, laundering on HI-Small in streaming, and Polymarket metaorders and churn at scale.</li>
    <li><b>Elsewhere a tree model with a short look-back is as good:</b> card fraud, laundering at loose levels and, on two of three banks, laundering run on whole histories.</li>
    <li><b>More labels do not explain the laundering lead:</b> on HI-Medium the network reaches its strict-level PD with 10% of the training events, while gradient boosting gets worse with more.</li>
    <li><b>Informed flow and news jumps cannot be read from the flow,</b> with or without wallet features, and detection does not yet pay a market maker.</li>
  </ul>
'''

END_TEXT = '''
  <ul class="tight">
    <li><b>Where DeepQCD wins:</b> laundering at PFA 0.01 on all three AMLworld banks (+0.04 to +0.13 in PD, every interval above zero), laundering on HI-Small run on every account (up to 2.4 times the detection at 0.5–1 false alert-days per 1,000), and Polymarket metaorders and churn at scale. In each the event is a change in how an account or a wallet behaves over a run of steps, and the strict alarm level is where it shows.</li>
    <li><b>Where it ties:</b> card fraud (TabFormer, Sparkov), laundering at PFA 0.05, laundering on LI-Small and HI-Medium run on whole histories.</li>
    <li><b>Where nothing works:</b> informed flow and news jumps on Polymarket; laundering outside ACH transfers (never caught) and outside the listed patterns (half the accounts, caught at 0.03–0.30 at most).</li>
    <li><b>For deployment:</b> average several seeds (single seeds spread by up to 0.13 in PD); set thresholds on a separate period, and watch the alert volume, which drifted more for DeepQCD than for gradient boosting.</li>
    <li><b>Limits:</b> the bank data are synthetic (AMLworld, TabFormer and Sparkov are simulators; Polymarket is real); streaming counts an alert as false whenever no flagged span covers it, including alerts on an account before its first laundering transfer; the market-making replay ignores queue position and liquidity rewards, and its P&amp;L is heavy-tailed near resolution.</li>
  </ul>
'''

TEXT = {'PM_TEXT': PM_TEXT, 'R_TEXT': R_TEXT, 'DATA_TEXT': DATA_TEXT, 'W_TEXT': W_TEXT, 'L_TEXT': L_TEXT, 'S_TEXT': S_TEXT,
        'KINDS_CAP': KINDS_CAP, 'SUMMARY': SUMMARY, 'END_TEXT': END_TEXT}

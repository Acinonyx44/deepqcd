# Defense guide: what we did, explained simply, with the charts and likely questions

## 1. The problem in one picture (ELI10)

You watch a stream of numbers, such as a sensor, a stock price or network traffic. At some unknown moment
something **changes**: a machine starts failing, an attack starts, the market panics. You want to **shout
"change!" as soon as possible after it happens**, but **not before it happens**. Shouting early is a *false
alarm*; shouting late is a *delay*.

This is **quickest change detection (QCD)**. There is always a trade-off:

- a **jumpy** detector catches changes fast but cries wolf a lot;
- a **calm** detector rarely cries wolf but is slow.

So you never judge a detector by one number. You judge it by its **trade-off curve**: how much delay it
needs for a given false-alarm level. **Lower curve = better.**

Every detector works the same way:

1. Keep a running **score** `d_t` that summarizes everything seen so far.
2. Raise the alarm the first time the score crosses a **threshold** `h`.
3. Turning `h` up or down slides you along the trade-off curve.

## 2. What the paper (DeepQCD) claims

- **Classical detectors** (CUSUM, Shiryaev, Shiryaev-Roberts) compute the score with a formula. The formula
  needs you to **know the exact statistics** of the data before and after the change. When you do know
  them, these detectors are provably optimal.
- **DeepQCD** says: don't write the formula, **learn it**. Train a small recurrent neural network (an LSTM)
  on streams where you know when the change happened.
  - Label every time step `0` (before the change) or `1` (after).
  - Train it like a classifier (binary cross-entropy).
  - Its output `d_t` learns to mean "**probability the change has already happened**", which is exactly
    what the optimal Shiryaev detector computes.
- **The paper's claims:**
  - (a) DeepQCD matches the optimal detectors when the model is known;
  - (b) it even **beats** CUSUM / Shiryaev-Roberts;
  - (c) it works on real data: surveillance video and IoT botnet attacks.

## 3. The key words (you will be asked)

| term | plain meaning |
|---|---|
| **tau (τ)** | the moment the change happens |
| **ADD** | average detection delay: how many steps after τ the alarm comes (lower = better) |
| **CADD** | conditional ADD: delay counted only for streams that did *not* false-alarm first |
| **PFA** | probability of false alarm: fraction of streams that alarm *before* τ (Bayesian view; lower = better) |
| **FAP** | false-alarm period: average steps between false alarms when nothing changes (minimax view; *higher* = better) |
| **DR** | detection rate: fraction of changes caught within the horizon (our real-data metric) |
| **Bayesian setting** | τ is random (geometric); compare ADD at the same PFA. Shiryaev is optimal here |
| **minimax setting** | worst case over τ; compare ADD at the same FAP. CUSUM is optimal here |
| **matched false-alarm rate** | always compare detectors at the *same* PFA / FAP, never at the same threshold, because thresholds mean different things for different detectors |

## 4. What we did, step by step

1. **Reproduced the paper in PyTorch.** The authors' code is old TensorFlow and very slow (one network call
   per time step). We rewrote the synthetic experiments with the same network and protocol, plus a fast
   simulator.
2. **Checked the protocol for fairness.** This is where the main criticism comes from (see 5.2).
3. **Added a finance case:** a volatility jump in GARCH returns.
4. **Surveyed ~40 real datasets.** We tested 21 real problems under one fair protocol against strong
   classical baselines.
5. **Built a hybrid:** DeepQCD fed the classical statistics as extra inputs.
6. **Tried zero-shot pretrained models** that need no training: Amazon Chronos, an open LLM, and TypeSafe Jev.

## 5. The charts and how to explain them

### 5.1 `figures/iid.png`: synthetic IID Gaussian (paper Figs. 5 and 6)

The data: 7 independent sensors with mean 0 that jump to mean 1.

**Left panel (Bayesian, paper Fig. 5).**
- x-axis: PFA, going right means *fewer* false alarms (log scale, reversed).
- y-axis: ADD.
- Blue (DeepQCD) sits just above orange (Shiryaev, the optimal).
- **Say:** "DeepQCD is within ~3 % of the optimal detector: at PFA 0.01, ADD 2.49 vs 2.41. The paper's
  first claim reproduces."

**Middle panel (minimax, paper Fig. 6), the important one.**
- x-axis: FAP, going right means *fewer* false alarms.
- **Solid lines** = the paper's protocol (change at τ = 1). Solid blue (DeepQCD) is far *below* CUSUM and
  Shiryaev-Roberts, which looks like DeepQCD "beats the optimal".
- **Dashed lines** = our check (change at τ = 200). Dashed blue lands *on top of* the classical curves.
- **Say:** "CUSUM is provably optimal in this setting, so a learned model can't really beat it. The paper put
  the change at the very first sample. A neural network can learn to be extra jumpy right at start-up at
  almost no cost in false alarms, because the false-alarm period is measured over long stretches. CUSUM
  can't do that; it behaves the same at every time step. Once the change happens later, after start-up, the
  advantage disappears: ADD 1.21 vs CUSUM 1.11 at FAP 1000. So the 'beats CUSUM' result is a start-up
  artifact of the protocol, not a real advantage."

**Right panel.**
- One example stream, change at t = 1000.
- DeepQCD's output (blue) almost exactly traces the true Shiryaev posterior (orange dashed).
- **Say:** "This is the cleanest evidence that the network really learned the posterior probability, as the
  theory says it should."

### 5.2 `figures/ar.png`: AR(1) data (paper Figs. 7 and 8)

Same three panels, but the data is correlated over time.

- **Same story:** within ~7 % of optimal in the Bayesian setting.
- The "beats CUSUM" gap again disappears at τ = 200; DeepQCD is ~5 % *slower* there.

### 5.3 `figures/transient.png`: transient change (paper Fig. 9)

The change only lasts ~25 steps, then disappears. So the metric is **PD** (probability of detecting it
while it lasts) vs PFA. Higher is better.

- **Left:** the blue (DeepQCD) and orange (window-limited CUSUM) curves lie on top of each other.
- **Say:** "The paper says DeepQCD slightly beats the window-limited CUSUM. We get a tie, with the benchmark
  0.5-2 % ahead, which is within noise."
- **Middle and right panels:** one stream each. The shaded band is when the change is present. DeepQCD's
  score ramps up slowly over the 25 steps; the windowed CUSUM reacts within a few steps. The network was
  trained on longer transients, so it is sluggish on short ones.

### 5.4 `figures/vol.png` / `vol_sq.png`: our finance extension (GARCH volatility)

Daily returns whose volatility doubles at the change. Volatility *clusters*, so calm and wild periods are
naturally mixed, which makes this hard.

- **Left (Bayesian):** DeepQCD is within ~7 % of the GARCH-aware optimal detector.
- **Middle (minimax):** again, solid lines (τ = 1) flatter DeepQCD and dashed lines (τ = 200) put everyone
  together.
- **`vol_sq` vs `vol`:** feeding the network both r and r² instead of r alone helps by 5-19 %. "The LSTM can
  learn to square, but not for free."

### 5.5 `figures/detect_*_tau500.png`: one realistic run at a fixed false-alarm budget

Instead of sweeping thresholds, we pick each detector's threshold so all of them make one false alarm per
~1000 steps. Then we watch what happens with the change at t = 500.

- **Top-left:** a single stream. Grey is the data, blue is DeepQCD's score, the red dotted line is the
  threshold. It alarms 3 steps after the change.
- **Top-right:** histogram of delays. Every detector catches most changes within 0-2 steps.
- **Bottom-left:** the trade-off curve with each detector's operating point marked. All curves overlap.
- **Bottom-right:** fraction of streams that have alarmed vs steps after the change. It tops out well
  below 1 because many streams already false-alarmed before t = 500. DeepQCD's line is lowest because it
  false-alarms *earlier* than the others.
- **The subtle finding:**
  - For a detector with no memory, the chance of a false alarm before t = 500 follows a known formula,
    1 − exp(−500 / FAP).
  - The classical detectors match it; **DeepQCD has 10-21 % more early false alarms** than it should.
  - That is the same start-up jumpiness as in 5.1, showing up in a different statistic.
  - **Say:** "For a recurrent detector, the average false-alarm period understates the risk of an early false
    alarm."

### 5.6 `figures/real.png`: all 21 real problems

One panel per dataset.

- x-axis: PFA (going right = fewer false alarms; the last tick "0" means no false alarms at all).
- y-axis: ADD, where a missed change counts as the full horizon, so missing is penalized.
- Lower and further right is better.
- Lines:
  - **blue** = DeepQCD (3 training seeds);
  - **purple** = hybrid;
  - **orange** = fitted CUSUM;
  - **green** = MEWMA chart;
  - **grey** = Shewhart chart;
  - **red** = the field's own standard rule.

The curves are step-shaped because they show the best delay achievable at each false-alarm budget.

Panels to point at:

- **Crypto pump-and-dump:** blue and purple are far below everything else. This is DeepQCD's clearest win.
- **Seismic:** grey (Shewhart) and red (STA/LTA) are far below blue. Simple charts win on sudden jumps.
- **Account takeover (keystrokes):** red (a chart that learns each user's normal from the first entries of
  the session) wins by far.
- **IoT Mirai:** everyone's delay is near 0. The attack is easy to detect.

Why `real.png` is not smooth like the paper's curves:

- The paper averages thousands of simulated or resampled streams. We test on the real recordings that exist.
- Each curve is drawn as exact steps, without smoothing.
- A missed change counts as the full horizon, which makes the flat ceilings.
- Every episode has its own p and q (a different machine, user or fault).

### 5.7 `figures/real_tau1.png`: the same 21 datasets under the paper's own protocol

This is the answer to "why don't your real-data charts look like the paper's?". We reran every dataset exactly
the way the paper runs N-BaIoT (Sec. 6.2):

- **The data is shuffled.** Every normal row goes into a "normal" bag and every changed row into a "changed" bag.
  Test streams are built by drawing rows at random from the bags.
- **False alarms use τ = ∞.** These streams use only the normal bag, and never change.
- **Delay uses τ = 1.** These streams use only the changed bag, so the very first observation is already
  post-change. "τ = 1" means the stream starts inside the anomaly.
- **The axes match the paper's Fig. 15.** x = FAP (mean steps to a false alarm, log scale), y = ADD.

What it shows:

- **The curves are smooth like the paper's,** because the bags can be resampled forever.
- **DeepQCD now wins 13 of 21 datasets** at FAP 100, compared with 5 of 21 on the real ordered recordings. Same
  data, same network, same rivals; only the protocol changed.
- **Why it gains.** Shuffling turns each dataset into an exact IID p → q problem. The network only has to tell
  q-rows from p-rows, and it can learn q's real shape, which a single Gaussian can't. What made real data hard is
  gone: slow drifts, bursts of correlated normal data, and the order in which a fault develops.
- **Where it still loses:** keystrokes (the change is in the order of entries), Yahoo, and at strict FAP SKAB and
  fish kill.

The sentence to say: "Under the paper's own protocol I reproduce the paper's kind of result on almost every
dataset. When the data keeps its real time order and the change comes after real normal data, it reverses on
most of them. The protocol, not the network, makes most of the difference." Details are in
`notes/realdata-tau1.md`.

## 6. The real-data results (the big table in the README)

**How we made it fair:**
- the same test windows for every detector, with the change at a *random* time;
- thresholds matched to the same false-alarm rate (PFA ≤ 0.1);
- 3 training seeds for the networks;
- strong baselines, including each field's own standard detector.

**Score: DeepQCD wins 5 / 21, loses 12. The hybrid wins 7, loses 8.**

| when DeepQCD wins | example |
|---|---|
| the change is a *pattern* (shape, rhythm, co-movement), not just a level shift | bee dance: 0.80 vs 0.30 |
| normal data is spiky, so formula-based charts drown in false alarms | pump-and-dump: 91 % caught in ~5 s vs 53 % in ~28 s |
| there are many labelled examples of the change (74-300) | SMD servers |

| when DeepQCD loses | example | who wins |
|---|---|---|
| sudden jump in level or variance | earthquakes 0.27 vs 0.93; Tennessee Eastman plant 0.66 vs 0.90 | Shewhart chart |
| few training examples (7-20) | occupancy, SKAB | classical charts |
| every stream has its own "normal" | keystroke takeover 0.43 vs 0.97 | self-calibrating chart |
| the "change" is a 1-20-step blip | fish kill, Yahoo | nobody detects these |

**The paper's IoT application (Mirai botnet, same 115 features as N-BaIoT):** every detector catches the
attack within 1-2 packets. DeepQCD's edge is about one packet. The task is easy, so it says little about
the method.

**Without labels (unsupervised):** detectors that need only normal data (Shewhart, MEWMA), or no data at all
(self-calibrating chart, Chronos, STA/LTA), win on about two thirds of the problems. Labels only pay off on
distinctive signatures (pump-and-dump, bee dance).

**Zero-shot Chronos** (Amazon's pretrained forecaster, no training): it beats DeepQCD on earthquakes
(0.84 vs 0.27) but not the simple Shewhart chart. It fails on slow drifts, because a forecaster adapts to
the drift, so it never looks surprising.

## 7. Questions the professor will likely ask, with answers

**"So is the paper wrong?"**
No. Its core idea works: the network really learns the optimal posterior (5.1, right panel), and it is near
optimal when the model is known. What we challenge is the *strength* of two claims:
- "beats CUSUM" comes from putting the change at t = 1;
- "beats window-limited CUSUM" is a tie.

On real data it is a specialist, not a universal winner.

**"How can you be sure the τ = 1 effect is an artifact?"**
- CUSUM is provably optimal in the minimax sense, so a real win would contradict a theorem.
- Moving the change to τ = 200 removes the gap.
- CUSUM's own numbers barely change between τ = 1 and τ = 200 (it is time-invariant), while DeepQCD's ADD
  jumps from 0.06 to 1.21.
- The front-loaded false alarms (5.5) are the same effect, seen from the other side.

**"Why compare at matched false-alarm rates?"**
A threshold of 0.5 on a probability and a threshold of 5 on a CUSUM sum mean completely different things.
The only fair comparison is how much delay each detector needs for the same false-alarm rate.

**"Isn't 21 datasets cherry-picked?"**
- They are every dataset from our survey that we could download, plus 6 new applications.
- One split per dataset, with 12-300 test episodes. Small differences (a few points) are noise; we only claim
  the big gaps.

**"Why did DeepQCD fail on Tennessee Eastman at first?"**
- The training set had a single 500-sample normal run, and the network memorized it. It then treated the
  normal part of every test run as "abnormal" and caught only 5 % of faults.
- Adding the dataset's second normal run fixed most of it (66 %).
- Lesson: neural detectors need diverse normal data; charts that only need normal statistics are much less
  sensitive to this.

**"What is the hybrid and why does it help?"**
- It is the same LSTM, but it also receives the classical statistics (the fitted log-likelihood ratio and
  the MEWMA chart value) and the data re-centred on the stream's own start.
- It starts from what the formulas already know and only has to learn what they miss.
- It is rarely worse than plain DeepQCD (the exception is S&P 500) and wins 7 instead of 5.

**"Why did the IoT result come out so different from the paper's Fig. 15?"**
- The paper compares against weaker baselines (PCA-CUSUM, QuantTree, and others), measures at τ = 1, and
  shuffles normal and attack samples independently.
- With strong charts and a random change time, everyone detects within 1-2 packets.

**"What are the limitations of your study?"**
- One train/test split per dataset.
- Thresholds were matched on the test windows. That is the same for every detector, but optimistic in
  absolute terms.
- No per-dataset tuning of the network.
- Some change times are conventions: C-MAPSS onset, S&P 500 dates.
- The paper's own datasets (UCF-Crime video, N-BaIoT) were not reachable; we used Mirai with the same
  features as a stand-in.

**"What would you do next?"**
- Run N-BaIoT itself under both protocols.
- Run the 500-run Tennessee Eastman set, to test whether more data closes the gap on sudden faults.
- Run STEAD (≈10⁵ earthquake traces instead of 92).
- Run the Jev and LLM zero-shot detectors with API access or a GPU.

## 8. One-paragraph summary to say out loud

"We reproduced DeepQCD in PyTorch. Its central idea holds: the network learns the change posterior and is
near-optimal when the data model is known. But its headline claim of beating CUSUM comes from putting the
change at the first sample, which rewards start-up jumpiness; with a later change it ties the optimal
detectors and false-alarms earlier than its average rate suggests. We then tested it on 21 real problems at
matched false-alarm rates against strong classical and domain baselines. It wins where the change is a
learnable pattern with many labelled examples (pump-and-dump, bee dance, server incidents). It loses on
sudden jumps, small data, and per-user baselines, where simple Shewhart, MEWMA or self-calibrating charts win.
A hybrid that feeds it the classical statistics is the most robust learned option. So DeepQCD is a useful
specialist, not a general replacement for classical change detection."

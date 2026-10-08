# The real datasets under the paper's own protocol (τ = 1, IID resampling)

`deepqcd_tau1.py`, figure `figures/real_tau1.png`. Companion to `notes/realdata.md`, which uses our protocol
(real, ordered recordings with the change after real normal data, `figures/real.png`).

## What the protocol is

This is exactly what the paper does for N-BaIoT (Sec. 6.2): "we simulate an infinite data stream by uniformly
sampling from the normal dataset for the pre-change case and the [attack] dataset for the post-change case. For
the ADD and the FAP calculations, we set τ = 1 and τ = ∞, respectively."

- **Pools.** Every normal row goes in a normal pool and every changed row in a changed pool. Rows come from the
  same standardized windows as `deepqcd_real.py`: training rows for training, test rows for testing.
- **τ = ∞ (false alarms).** 500 streams of 2,000 rows drawn at random from the normal test pool.
  FAP = steps observed / false alarms, i.e. the mean time to a false alarm.
- **τ = 1 (delay).** 500 streams drawn from the changed test pool, so **the very first row is already
  post-change**. ADD = alarm time − 1, and a miss counts 500.
- **Training.** Streams of 1,024 rows resampled the same way, with τ uniform in [64, 960] (the paper used 2,048
  and [128, 1920]). A quarter of the streams have no change, because the FAP streams are longer than the training
  streams. DeepQCD and the hybrid use the same network as before, 3 seeds each.
- **Rivals.** The same fitted-Gaussian CUSUM, MEWMA, Shewhart and field rules, fitted on the same training rows.
- **Plotting.** Curves are drawn only where at least 10 false alarms were seen, so every FAP is measured rather
  than extrapolated. That is also why some curves stop early.

## Results

Each cell is **ADD at FAP ≥ 100 / ADD at FAP ≥ 1000** (lower is better), in the dataset's own step. DeepQCD and
the hybrid are the median of 3 seeds. The verdict compares DeepQCD with the best classical detector at each level;
a difference under 10 % (or half a step) counts as a tie.

| dataset | step | DeepQCD | hybrid | CUSUM (fitted) | MEWMA | Shewhart | field rule | DeepQCD vs best classical |
|---|---|---|---|---|---|---|---|---|
| SKAB water pump | s | 6.3 / 84.6 | 5.7 / 199 | 5.8 / 6.0 | 7.5 / 8.5 | 5.8 / 6.0 | — | tie / loss |
| Tennessee Eastman (21 faults) | 3 min | 0.1 / 0.5 | 0.1 / 0.7 | 1.2 ᶠ | 1.7 / 1.8 | 0.6 / 0.6 | — | tie / tie |
| UCI room occupancy | min | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.7 | 2.9 / 3.4 | 1.1 / 3.0 | — | tie / **WIN** |
| UCI room occupancy (no light sensor) | min | 2.5 / 11.8 | 7.5 / 18.8 | 7.5 / 17.4 | 11.3 / 18.7 | 20.0 / 59.5 | — | **WIN** / **WIN** |
| C-MAPSS FD001 turbofans | cycle | 0.0 / 0.4 | 0.1 / 0.8 | 1.1 / 1.3 | 3.7 / 4.1 | 3.6 / 4.9 | — | **WIN** / **WIN** |
| SMD server machines | min | 1.3 / 6.7 | 5.0 / 31.7 | 6.3 / 11.9 | 17.5 / 53.6 | 5.0 / 17.2 | — | **WIN** / **WIN** |
| HAI 21.03 ICS attacks | s | 0.1 / 0.8 | 0.1 / 1.7 | 0.4 / 1.0 | 1.2 / 1.4 | 0.5 / 1.2 | — | tie / tie |
| NAB (58 series) | step | 9.3 / 93.3 | 18.0 / 112 | 21.3 / 88.8 | 37.6 / 94.9 | 20.9 / 92.5 | — | **WIN** / tie |
| TCPD (univariate, consensus CPs) | step | 20.7 / 186 | 166 / 185 | 71.4 / 500 | 47.5 / 175 | — / — | — | **WIN** / tie |
| Bee waggle dance (6 seqs) | frame | 25.1 / 157 | 26.2 / 197 | 56.3 / 250 | 65.1 / 336 | 98.3 / 410 | — | **WIN** / **WIN** |
| HASC accelerometer activities | sample | 3.6 / 34.9 | 4.5 / 46.4 | 37.6 / 112 | 188 / 467 | 197 / 475 | — | **WIN** / **WIN** |
| Dam water level (fish kills) | step | 5.0 / 78.0 | 5.0 / 78.1 | 15.0 / 32.2 | 498 / 500 | 500 / 500 | — | **WIN** / loss |
| Yahoo S5 subset (15 series) | hour | 21.3 / 219 | 21.3 / 333 | 10.0 / 38.3 | 11.3 / 39.7 | 4.6 / — | — | loss / loss |
| pmuBAGE grid events | 1/30 s | 0.1 / 0.5 | 0.0 / 0.2 | 1.2 / 1.9 | 1.2 / 2.1 | 1.3 / 2.7 | — | **WIN** / **WIN** |
| S&P 500 stress episodes | day | 3.7 / 14.7 | 3.8 / 16.3 | 11.2 / 22.3 | 8.7 / 21.3 | 11.2 / 27.5 | — | **WIN** / **WIN** |
| Seismic P-wave onset (PhaseNet) | 10 ms | 11.3 / 155 | 9.9 / 78.2 | 38.8 / 204 | 47.6 / 176 | 62.5 / — | STA/LTA trigger 395 / 409 | **WIN** / **WIN** |
| Freezing of gait (Daphnet) | 1/32 s | 11.0 / 99.0 | 7.9 / 98.7 | 37.5 / 206 | 191 / 460 | 72.8 / 315 | Freeze index 90.2 / 284 | **WIN** / **WIN** |
| Crypto pump-and-dump | 5 s chunk | 0.0 / 3.4 | 0.0 / 4.3 | 11.3 / 14.7 | 11.3 / 14.8 | 11.3 / 14.8 | — | **WIN** / **WIN** |
| Account takeover (keystrokes) | entry | 111 / 382 | 122 / 388 | 59.8 / 268 | 115 / 410 | 73.9 / 389 | Self-calibrating chart 83.4 / 275 | loss / loss |
| IoT Mirai botnet (temporal blocks) | packet | 0.0 / 0.1 | 0.0 / 0.1 | 0.2 / 0.2 | 0.2 / 0.2 | 0.2 / 0.2 | — | tie / tie |
| IoT Mirai botnet (paper protocol: IID splice) | packet | 0.0 / 0.1 | 0.0 / 0.1 | 0.2 / 0.2 | 0.2 / 0.2 | 0.2 / 0.2 | — | tie / tie |

ᶠ Fitted CUSUM on TEP raised fewer than 10 false alarms in all 10⁶ normal steps at every threshold, so its FAP is
beyond what we can measure; with no false alarms at all, its ADD is 1.2.

**Tally of DeepQCD against the best classical detector:**

| | WIN | tie | loss |
|---|---|---|---|
| FAP 100, DeepQCD | 13 | 6 | 2 |
| FAP 100, hybrid | 10 | 8 | 3 |
| FAP 1000, DeepQCD | 11 | 6 | 4 |
| FAP 1000, hybrid | 8 | 6 | 7 |
| *For comparison, our protocol (`notes/realdata.md`), DeepQCD* | *5* | *3 (+1 none)* | *12* |

## What it means

1. **Under the paper's protocol DeepQCD wins most problems: 13 of 21, versus 5 of 21 under ours.** Same data,
   same network, same rivals; only the protocol changed. That is the strongest single piece of evidence for our
   main claim: the paper's evaluation flatters the method.
2. **Why the curves are smooth.** Resampling gives unlimited streams from fixed pools, so every point averages
   500 × 2,000 rows. On real ordered recordings we only have the episodes that exist, so `real.png` is stepped.
3. **Why DeepQCD gains so much.** Shuffling turns each dataset into an exact IID p → q problem with p and q fixed.
   - The network then only has to learn which rows look like q. It can learn q's real shape (several clusters,
     heavy tails), which a single fitted Gaussian can't.
   - What made real data hard for it is gone: slow drifts, autocorrelated normal stretches that trigger bursts of
     false alarms, and machines or users that differ from the training ones in their order of events.
   - τ = 1 again rewards a jumpy start-up, as in the synthetic Fig. 6 (`notes/results.md`).
4. **Where it still loses.** Keystrokes (the change of typist is in the order of entries, not in single rows),
   Yahoo, and at strict FAP SKAB and fish kill.
   - At FAP 1000 several DeepQCD seeds jump up sharply (SKAB, Yahoo, TEP): to stay quiet for 1,000+ steps the
     network's output has to stay low over far longer stretches than in training, and it doesn't always.
   - The classical charts degrade smoothly instead.
5. **Two detectors are handicapped by τ = 1 by construction.**
   - The hybrid's re-referenced input and the self-calibrating chart (keystrokes) both take their baseline from
     the stream's first steps, which here are already post-change.
   - STA/LTA (seismic) has the same issue: its long-term average starts at the noise level and adapts to the quake.
6. **Bottom line.** Under the paper's protocol we reproduce the paper's kind of result on almost every dataset.
   Under a protocol that keeps the data's real time order and puts the change after real normal data, the result
   reverses on most of them. The paper's IoT result is real but tells you about IID-shuffled data, not about
   monitoring a live stream.

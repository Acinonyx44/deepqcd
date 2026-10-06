# Real datasets for DeepQCD

*Results of running DeepQCD on the 15 reachable ones: `notes/realdata.md`.*

A survey of public datasets (mostly on GitHub) that fit our setup: streams that start in a normal regime and
switch to an abnormal one at a time we know, so we can label every step 0 or 1 for training and measure
detection delay and false alarms when testing. Compiled 2026-10-06. "Reachable here" means the
data downloaded from a Claude Code cloud session. In those sessions only GitHub (`raw.githubusercontent.com`,
`git clone`) gets through the network policy. PhysioNet, UCI, Kaggle, Harvard Dataverse, Stanford and
thedatum.org all timed out, so those datasets have to be fetched on a laptop.

## What a dataset needs for our recipe

Algorithm 1 trains on labels `0 before tau, 1 after`, and testing needs both a change time and change-free
data. So a dataset is useful in proportion to how many of these it gives us:

1. **A known onset time `tau`**, not just "this window is anomalous". Window labels (NAB, SMD) can be turned
   into onsets by taking each window's first point, but the window edges are often loose.
2. **Many independent streams**, each with its own change. A single long series with three changes is fine for
   evaluating a detector but too little to train an LSTM on.
3. **A change that persists** (Secs. 5.1, 5.2) or a transient with a known end (Sec. 5.3).
4. **Long stretches with no change**, to estimate the false-alarm period (FAP) and to calibrate thresholds as
   `detect.py` does.
5. **A random `tau`.** If every stream changes at the same index, the network can learn the clock instead of
   the change. SKAB has this problem (see below); the fix is to crop a random amount of the normal prefix.

There are three ways to get QCD streams out of real data, roughly in order of how faithful they are:

- **Native runs**: each recording is one experiment, normal and then faulty, with the onset recorded
  (Tennessee Eastman, SKAB, STEAD, seizures). The best kind.
- **Splicing**: take a pool of normal samples and a pool of abnormal samples, and build a stream by drawing
  `tau - 1` normal samples followed by abnormal ones. This is exactly how the paper did N-BaIoT (Sec. 6.2:
  "uniformly sampling from the normal dataset for the pre-change case and the spam attack dataset for the
  post-change"). It gives unlimited streams and a random `tau`, but destroys the real time dependence across
  the change, so it tests whether the detector tells the two distributions apart, not whether it tracks
  realistic transitions. Any labelled "normal vs. fault/attack/activity" dataset works this way.
- **Long annotated series**: one or a few long recordings with marked change points (TCPD, NAB, UCR). Use them
  for evaluation, and either train on a related synthetic source or use a sliding-window protocol.

## Recommended shortlist

In the order I would try them. Each one adds something the synthetic experiments do not.

| # | dataset | why it fits | protocol | reachable here |
|---|---|---|---|---|
| 1 | **Tennessee Eastman, Rieth et al. 2017** | The canonical sequential fault-detection benchmark. 20 fault types × 500 independent simulation runs for training and 500 for testing. Training runs are 500 samples with the fault at sample 20; testing runs are 960 samples with the fault at sample 160 (3-min sampling). 52 sensors. Plenty of streams, real process dynamics, and known onsets. Detection delay is already the standard metric. | native runs; crop the normal prefix to randomize `tau`; fault-free runs for FAP | the Braatz version (1 run per fault) is on GitHub and reachable; the 500-run Rieth version is on Harvard Dataverse (~1.8 GB), or via `fddbenchmark` |
| 2 | **SKAB (Skoltech Anomaly Benchmark)** | Real testbed (water pump, 8 sensors at 1 Hz), 34 runs each with one labelled fault segment and a `changepoint` column, plus a 9405-step anomaly-free recording. Small and quick to iterate on. | native runs; must randomize `tau` (all runs but two change at index 557-578); anomaly-free file for FAP | **yes** (cloned and checked) |
| 3 | **N-BaIoT** and its successor **CICIoT2023** | Sec. 6.2 of the paper; reproducing it is the natural real-data next step. CICIoT2023 is 2023 traffic with 33 attacks in 7 families and 47 features. | splicing (benign pool → attack pool), as in the paper | no (UCI / CIC sites) |
| 4 | **STEAD** (Stanford Earthquake Dataset) | About 1 million three-component waveforms at 100 Hz, 60 s each, with a hand-picked P-wave arrival sample per trace, plus about 200k pure-noise traces. That gives an enormous number of native onsets. P-wave picking is classic QCD: the STA/LTA trigger is a windowed likelihood-ratio test, so it is the natural benchmark. CC-BY-4.0. | native runs (`tau = p_arrival_sample`); noise chunk for FAP | metadata/code on GitHub; data (~85 GB, or 15 GB chunks) via external links |
| 5 | **CHB-MIT scalp EEG** | 23 patients, 22 channels at 256 Hz, about 980 h with annotated seizure onsets. Seizure-onset latency is the clinical metric. Hard: few onsets per patient, strong patient differences. | native onsets; train across patients, test on a held-out one; seizure-free hours for FAP | no (PhysioNet) |
| 6 | **HAI** (ICS security) | Industrial control testbed (boiler, turbine, water treatment, hardware-in-the-loop sim). Hundreds of hours of normal operation and dozens of labelled attacks per version (52 in 23.05), CSV, versions 20.07-23.05. | native: attack start times inside long test files; normal files for FAP | **yes** (GitHub `icsdataset/hai`) |
| 7 | **Real volatility (VIX, S&P 500)** | Takes our GARCH experiment to real data. | train on the GARCH source, evaluate on real returns around dated stress events | VIX daily 1990-2026 **yes** (`datasets/finance-vix`); daily stock prices need yfinance / Stooq on a laptop |

If I had to pick one: **Tennessee Eastman (Rieth)**. It is the only real-dynamics dataset with hundreds of
independent native runs per fault type, it has standard competitors (PCA T²/SPE, Hotelling charts, and recent
RNNs that report detection delay), and it has a second difficulty level built in: faults 3, 9 and 15 are
known to be nearly undetectable.

## Full catalog by domain

### Industrial process and machine faults

| dataset | change | size / streams | labels | link |
|---|---|---|---|---|
| Tennessee Eastman (Braatz) | 21 process faults | one run per fault, 52 vars: a 480-sample training file recorded after the fault is already on, and a 960-sample test file with the fault at 160; normal run `d00` is 500 samples | onset at 160 in the test files only | [camaramm/tennessee-eastman-profBraatz](https://github.com/camaramm/tennessee-eastman-profBraatz) (checked, reachable) |
| Tennessee Eastman (Rieth 2017) | 20 faults | 21 × 500 train + 21 × 500 test runs | onset at sample 20 / 160 | Harvard Dataverse; packaged by [AIRI-Institute/fddbenchmark](https://github.com/AIRI-Institute/fddbenchmark) (also `reinartz_tep`: 28 faults, 2800 runs of 2000 samples; `small_tep`: 210 runs, 19 MB) |
| SKAB | valve/pump faults | 34 runs of 745-1327 steps, 8 sensors, 1 Hz, plus 9405 normal steps | `anomaly` + `changepoint` columns | [waico/SKAB](https://github.com/waico/SKAB) (GPL-3.0; checked, reachable) |
| C-MAPSS turbofan (NASA) | gradual degradation to failure | 4 subsets, ~100-260 engines each, 21 sensors | failure time only; the fault onset is unlabelled | NASA PCoE; many GitHub mirrors |
| FEMTO / PRONOSTIA bearings | run-to-failure degradation | 17 bearings, 25.6 kHz vibration | end of life only | [awesome-bearing-dataset](https://github.com/VictorBauler/awesome-bearing-dataset) lists it and IMS, CWRU, Paderborn |
| CWRU / Paderborn / Lessmeier bearings | healthy → seeded fault | many short recordings per fault class | class labels | as above; `lessmeier_bearing` is in fddbenchmark — splice healthy → faulty |
| UCI Occupancy | room empty → occupied | ~20k rows, 5 sensors, minute data | occupancy column | [LuisM78/Occupancy-detection-data](https://github.com/LuisM78/Occupancy-detection-data) (reachable). The authors' own `SignalDetect_Occupancy.py` uses it, though the paper does not |

### Cyber-physical and network security

| dataset | change | size | labels | link |
|---|---|---|---|---|
| N-BaIoT | IoT botnet attack | 9 devices, benign + 10 attacks, 115 features | per-file class | UCI (paper Sec. 6.2) |
| CICIoT2023 | 33 IoT attacks, 7 families | ~47 flow features | per-flow class | CIC / IEEE DataPort |
| CIC-IDS2017 | 14 attacks over a 5-day capture | 2.8M flows, 78 features | per flow, with timestamps, so attacks have real start times | CIC |
| HAI 20.07-23.05 | ICS attacks | hundreds of hours normal; dozens of attacks per version (52 in 23.05) | per-second attack flag | [icsdataset/hai](https://github.com/icsdataset/hai) (reachable) |
| SWaT / WADI | water-plant attacks | 11 and 16 days, 51 and 123 sensors | attack periods | iTrust, on request; loaders in [ipal-ids/ipal_datasets](https://github.com/ipal-ids/ipal_datasets) |
| SMD (server machines) | server incidents | 28 machines × 38 metrics, 5 weeks | point labels | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |
| SMAP / MSL telemetry | spacecraft anomalies | 80 channels, labelled ranges | `labeled_anomalies.csv` | [khundman/telemanom](https://github.com/khundman/telemanom) (reachable) |

### Geophysics, power and environment

| dataset | change | size | labels | link |
|---|---|---|---|---|
| STEAD | earthquake P-wave arrival | ~1M 3-channel 60 s traces at 100 Hz, ~200k noise traces | `p_arrival_sample`, `s_arrival_sample`, magnitude, distance | [smousavi05/STEAD](https://github.com/smousavi05/STEAD) (CC-BY-4.0) |
| pmuBAGE | grid frequency / voltage events | 84 + 620 synthetic events, 100 PMUs, 30 Hz, 20 s windows | event type; onset not documented | [NanpengYu/pmuBAGE](https://github.com/NanpengYu/pmuBAGE) |
| PSML | grid disturbances | multi-scale sim + real data | event labels; has an "early detection" task | [tamu-engineering-research/Open-source-power-dataset](https://github.com/tamu-engineering-research/Open-source-power-dataset) |
| REDD / UK-DALE (NILM) | appliance switches on | months of household power | per-appliance submeters give exact on-times | via [nilmtk](https://github.com/nilmtk/nilmtk) |
| GOES X-ray flux | solar flare onset | 1-min flux since the 1980s | NOAA flare catalogue start times | NOAA; e.g. [solar-xray-statistical-detection-classification](https://github.com/Starcloud-retro/solar-xray-statistical-detection-classification) |

### Health

| dataset | change | size | labels | link |
|---|---|---|---|---|
| CHB-MIT | seizure onset | 23 patients, 22 ch, 256 Hz, ~980 h | onset / end seconds per file | PhysioNet; pipelines e.g. [marcusgaitan16-rice/chbmit](https://github.com/marcusgaitan16-rice/chbmit) (low-latency, causal) |
| MIT-BIH AF / Long-Term AF | atrial fibrillation onset | 25 × 10 h, and 84 long-term Holter records | rhythm annotations | PhysioNet |
| HASC 2011 activity | activity transitions (walk → run …) | 3-axis accelerometer, many subjects | transition times | hasc.jp; used by [KL-CPD](https://github.com/OctoberChang/klcpd_code) and [MC-TIRE](https://github.com/caozhenxiang/MC-TIRE) |
| JHU CSSE COVID-19 | outbreak waves | daily counts per country/state 2020-23 | none; use dated policy events | [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) (reachable) |

### Finance

| dataset | change | notes | link |
|---|---|---|---|
| VIX daily, 1990 to now | volatility regime | direct check of the GARCH experiment against dated stress events (1998, 2008, 2011, 2020) | [datasets/finance-vix](https://github.com/datasets/finance-vix) (reachable) |
| Daily equity / crypto returns | volatility jumps | yfinance, Stooq, Binance public data; we would label changes ourselves (dated events, or segments a fitted Markov-switching GARCH assigns to the high-vol regime) | laptop only |
| FI-2010 / LOBSTER limit order books | microstructure regime shifts | high-frequency; no change labels, needs a labelling convention | [lob-deep-learning](https://github.com/Jeonghwan-Cheon/lob-deep-learning) |

### Benchmarks of annotated series (evaluation, not training)

| benchmark | content | labels | link |
|---|---|---|---|
| TCPD (Turing) | 37 real series, many domains | change points from 5 human annotators each; `annotations.json` | [alan-turing-institute/TCPD](https://github.com/alan-turing-institute/TCPD), results in [TCPDBench](https://github.com/alan-turing-institute/TCPDBench) (reachable) |
| NAB | 58 series, 365k points, streaming | anomaly windows; scoring credits the earliest detection, which is close to a QCD delay | [numenta/NAB](https://github.com/numenta/NAB) (reachable) |
| UCR anomaly archive | 250 series, exactly one anomaly each | anomaly range in the filename | UCR (laptop) |
| TSB-AD (NeurIPS 2024) | 1070 series from 40 datasets, curated | point labels; `TSB-AD-U` / `TSB-AD-M` | [TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD) |
| Lists of more | | | [awesome-ts-anomaly-detection-datasets](https://github.com/OliverHennhoefer/awesome-ts-anomaly-detection-datasets), [awesome-TS-anomaly-detection](https://github.com/rob-med/awesome-TS-anomaly-detection) |
| GutenTAG | synthetic generator with configurable anomaly types | exact labels | [HPI-Information-Systems/gutentag](https://github.com/HPI-Information-Systems/gutentag) |

## Pitfalls already spotted

- **SKAB's onsets are almost all at the same index.** Measured on the 34 labelled files: 31 change between
  steps 557 and 578, one at 104, one at 495, one at 796. Trained as-is, an LSTM can score well just by
  counting to ~570. Crop a random-length normal prefix (or prepend normal data from the anomaly-free file).
  Also, its `changepoint` column marks several points around both the start and the end of a fault, so
  take the onset from the first `anomaly == 1`.
- **Fault data that ends.** SKAB faults and HAI attacks last minutes and then the process recovers. That is
  the Sec. 5.3 transient setting (score with detection probability within a window), not the persistent one.
- **Window labels are not onsets.** NAB windows are centred on the labelled anomaly, and together they cover 10 % of each file. A
  detection inside the window but before the true onset counts as a hit there; in QCD it would be a false
  alarm.
- **Weak faults.** In Tennessee Eastman, faults 3, 9 and 15 are known to be nearly indistinguishable from normal;
  report them separately or a single average delay hides it.
- **No model-based benchmark.** On real data there is no exact Shiryaev or CUSUM. Fair competitors are a
  CUSUM/Shiryaev on a density fitted to training data (Gaussian, or a GARCH for returns), Hotelling T² / PCA
  SPE charts (Tennessee Eastman), STA/LTA (STEAD), and the rolling-variance rule we already have.

## Fitting this into the code

`sources.py` already has the interface a real dataset needs: `training_set(n, T, tau)`, a `sampler(n, tau)`
for testing, and `benchmarks()`. A real source differs in two ways:

- `sampler` draws from a finite pool of runs instead of a generator. For native runs: pick a run, crop its
  normal prefix so the change lands at the requested `tau`. For splicing: concatenate `tau - 1` normal draws
  with abnormal draws.
- `benchmarks()` returns CUSUM/Shiryaev on fitted densities, and dataset-specific rules (T², STA/LTA).

`detect.py --source <name>` then works unchanged: calibrate on change-free streams, report early / caught /
delay at a fixed FAP. The natural first real source is SKAB (small, reachable here, already inspected),
followed by Tennessee Eastman for scale.

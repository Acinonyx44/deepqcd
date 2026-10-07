"""
Jev as a zero-shot quickest change detector.

Jev (TypeSafe AI, Sept. 2026) is a "System One" decision model: given a state (text or JSON) and typed
questions, it returns probabilities over the declared answers instead of generating text, in tens to
hundreds of milliseconds. A yes/no ("noul") question

    "Has the process already left its normal regime?"

asked about the stream observed so far returns P(change has happened | x_1..x_t). That is exactly the
decision statistic d_t that DeepQCD is trained to output (Eq. 6 of the paper). So Jev slots into the
generic QCD procedure with no training at all: s_t = the recent history serialized as JSON, d_t = Jev's
probability, alarm when d_t >= h.

    .venv/bin/python jev_qcd.py occupancy --mock          # plumbing check with a local stand-in, no API
    .venv/bin/python jev_qcd.py occupancy --estimate      # number of calls and tokens, no API
    .venv/bin/python jev_qcd.py occupancy pumpdump        # the real thing (needs TYPESAFE_API_KEY)

Protocol: the test windows are exactly those of deepqcd_real.py (same seeds), restricted to the first
--windows of them. Jev is asked every --stride steps (it is a network call), and its answer is held
until the next one. For a fair comparison every other detector is reported twice: at full rate, and
"on Jev's schedule", i.e. only allowed to alarm at the same decision times. DeepQCD (one seed) and the
classical rivals are fitted on the same training windows as in deepqcd_real.py.

API (from the official typesafe-sdk 0.7.2, called here with the standard library so the project needs no
new dependency):
    POST {TYPESAFE_BASE_URL or https://api.typesafe.ai}/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    {"state": ..., "model": "jev-latest", "questions": {"changed": {"type": "noul", "instructions": ...,
     "criteria": {"true": ..., "false": ...}}}}
    -> {"model": ..., "usage": {...}, "answers": {"changed": {"type": "noul", "noul": 0.97}}}
Answers are cached in runs/jev/<dataset>.jsonl by a hash of the request, so a rerun costs nothing.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import deepqcd_real as R
from qcd import DeepQCD, NetDetector, decision_statistics, train

MOCK = '--mock' in sys.argv
ESTIMATE = '--estimate' in sys.argv


def _arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


N_WINDOWS = _arg('--windows', 200)
WORKERS = _arg('--workers', 16)
HISTORY = 20  # most recent observations shown verbatim

# What Jev is told about each stream: what it is, what the features mean, and what the change looks like.
# Values are given standardized (normal operation ~ mean 0, unit variance per feature).
SPEC = {
    'occupancy': dict(
        stride=5,
        context='Minute-by-minute sensor readings in an office room.',
        features=['temperature', 'humidity', 'light', 'CO2', 'humidity_ratio'],
        change='Someone has entered and is now occupying the room (lights on, CO2 rising, warmer).',
        normal='The room is still empty; only slow drifts and sensor noise.'),
    'pumpdump': dict(
        stride=2,
        context='Market activity of one cryptocurrency on Binance in 5-second chunks.',
        features=['std_rush_orders', 'avg_rush_orders', 'std_trades', 'std_volume', 'avg_volume',
                  'std_price', 'avg_price', 'avg_price_max'],
        change='A coordinated pump-and-dump has started: a burst of rush buy orders, trades and volume '
               'with the price driven up.',
        normal='Ordinary trading, including the usual occasional spikes.'),
    'sp500': dict(
        stride=5,
        context='Daily S&P 500 log returns divided by their trailing one-year volatility.',
        features=['return', 'return_squared'],
        change='The market has entered a stress regime: volatility has jumped and stays high.',
        normal='The usual calm regime, including isolated large days.'),
    'skab': dict(
        stride=10,
        context='One-second readings of a water-circulation testbed (pump, valves, pipes).',
        features=['accelerometer1_rms', 'accelerometer2_rms', 'current', 'pressure', 'temperature',
                  'thermocouple', 'voltage', 'flow_rate'],
        change='A fault has occurred (e.g. a partially closed valve, cavitation, a leak) and persists.',
        normal='Normal operation.'),
    'keystroke': dict(
        stride=2,
        context='Successive entries of the same password; each entry is 31 keystroke timings '
                '(hold times and inter-key latencies), standardized.',
        features=[f't{i}' for i in range(31)],
        change='A different person has taken over the keyboard: the typing rhythm no longer matches '
               'the one at the start of the session.',
        normal='The same person who started the session is still typing.'),
}


# ---------------------------------------------------------------- the request

def state_at(x, t, names, ctx):
    """JSON the model sees at time t (0-based, x[:t+1] observed): context, a summary of the start of the
    stream (the reference) and of the recent past, and the last HISTORY observations verbatim."""
    seen = x[:t + 1]
    ref = seen[:max(1, min(len(seen), 30))]
    rec = seen[-HISTORY:]
    r = lambda v: [round(float(u), 2) for u in v]
    return {
        'stream': ctx['context'],
        'values': 'standardized per feature: normal operation is about mean 0, standard deviation 1',
        'observations_so_far': int(t + 1),
        'features': names,
        'start_of_stream': {'mean': r(ref.mean(0)), 'std': r(ref.std(0))},
        f'last_{len(rec)}': {'mean': r(rec.mean(0)), 'std': r(rec.std(0)),
                             'max_abs': r(np.abs(rec).max(0))},
        'recent_observations_oldest_first': [r(v) for v in rec],
    }


def question(ctx):
    return {'changed': {'type': 'noul',
                        'instructions': 'Has the process already left its normal regime, i.e. has the '
                                        'change described below happened by the latest observation?',
                        'criteria': {'true': ctx['change'], 'false': ctx['normal']}}}


# ---------------------------------------------------------------- the client (stdlib)

class Jev:
    def __init__(self, cache_path):
        self.base = os.environ.get('TYPESAFE_BASE_URL', '').strip() or 'https://api.typesafe.ai'
        self.model = os.environ.get('TYPESAFE_DEFAULT_MODEL', '').strip() or 'jev-latest'
        self.key = os.environ.get('TYPESAFE_API_KEY', '').strip()
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                k, v = json.loads(line)
                self.cache[k] = v
        self.calls = self.tokens = 0

    def body(self, state, questions):
        return {'state': state, 'model': self.model, 'questions': questions}

    def ask(self, body):
        k = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        if MOCK:
            p = mock_jev(body)
        else:
            p = self._post(body)
        self.cache[k] = p
        with open(self.cache_path, 'a') as f:
            f.write(json.dumps([k, p]) + '\n')
        return p

    def _post(self, body, tries=5):
        if not self.key:
            raise SystemExit('TYPESAFE_API_KEY is not set (add it to the environment), or use --mock / --estimate')
        req = urllib.request.Request(
            self.base.rstrip('/') + '/v1/systemone', data=json.dumps(body).encode(), method='POST',
            headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json',
                     'Accept': 'application/json', 'User-Agent': 'deepqcd-jev-trial'})
        for i in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    out = json.load(resp)
                self.calls += 1
                self.tokens += (out.get('usage') or {}).get('input_tokens') or 0
                return float(out['answers']['changed']['noul'])
            except urllib.error.HTTPError as e:
                if e.code in (408, 429) or e.code >= 500:
                    time.sleep(float(e.headers.get('retry-after') or 2 ** i))
                    continue
                raise SystemExit(f'Jev API error {e.code}: {e.read()[:300]!r}')
            except (urllib.error.URLError, TimeoutError) as e:
                if i == tries - 1:
                    raise SystemExit(f'Jev API unreachable ({e}); is the host allowed by the network policy?')
                time.sleep(2 ** i)


def mock_jev(body):
    """Local stand-in for --mock: a logistic of how far the recent mean sits from the stream's start.
    It only exercises the plumbing; its numbers say nothing about Jev."""
    s = body['state']
    a, b = np.array(s['start_of_stream']['mean']), np.array(next(v for k, v in s.items() if k.startswith('last_'))['mean'])
    z = np.sqrt(((b - a) ** 2).mean())
    return float(1 / (1 + np.exp(-4 * (z - 1))))


# ---------------------------------------------------------------- evaluation

def on_schedule(dstat, stride):
    """The detector still sees every observation, but may only raise an alarm at Jev's decision times
    (stride-1, 2*stride-1, ...): at each one it reports the largest d_t since the previous one, so a
    crossing is acted on at the next decision time."""
    out = np.full_like(dstat, -np.inf)
    for t0 in range(0, dstat.shape[1], stride):
        t1 = min(t0 + stride, dstat.shape[1])
        out[:, t1 - 1] = dstat[:, t0:t1].max(1)
    return out


def run(name):
    ctx = SPEC[name]
    p = R.prepare(name)
    d = p['d']
    n = min(N_WINDOWS, len(p['xs']))
    xs, tau, length = p['xs'][:n], p['tau'][:n], p['length'][:n]
    stride, names = ctx['stride'], ctx['features']
    print(f'\n=== {d.label} ({name}): {n} test windows, Jev asked every {stride} {d.unit}')

    times = [(i, t) for i in range(n) for t in range(stride - 1, int(length[i]), stride)]
    q = question(ctx)
    os.makedirs(os.path.join('runs', 'jev'), exist_ok=True)
    jev = Jev(os.path.join('runs', 'jev', f'{name}{"-mock" if MOCK else ""}.jsonl'))
    bodies = [jev.body(state_at(xs[i, :length[i]], t, names, ctx), q) for i, t in times]
    if ESTIMATE:
        chars = np.mean([len(json.dumps(b)) for b in bodies[:200]])
        new = sum(hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() not in jev.cache for b in bodies)
        print(f'  {len(bodies)} calls ({new} not cached), ~{chars / 4:.0f} input tokens each, '
              f'~{new * chars / 4 / 1e6:.1f} M tokens = ~${new * chars / 4 / 1e6 * 0.042:.2f} at $0.042 / M')
        return None
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        probs = list(ex.map(jev.ask, bodies))
    print(f'  {len(bodies)} answers in {time.time() - t0:.0f}s ({jev.calls} API calls, {jev.tokens} input tokens)')
    dj = np.full(xs.shape[:2], -np.inf)
    for (i, t), pr in zip(times, probs):
        dj[i, t:t + stride] = pr

    stats, _ = R.rivals({**p, 'xs': xs})
    torch.manual_seed(0)
    net = DeepQCD(p['P'], 16 if p['P'] <= 10 else 32)
    print('  training DeepQCD (1 seed) on the same windows ...')
    train(net, p['xt'], p['yt'], p['xv'], p['yv'], epochs=R.EPOCHS, w=p['mt'], wv=p['mv'])
    stats = {'DeepQCD (trained)': decision_statistics(NetDetector(net, None), xs), **stats}

    rows = {('Jev MOCK stand-in (not Jev)' if MOCK else 'Jev (zero-shot)'): R.tradeoff(dj, tau, length)}
    for k, s in stats.items():
        rows[k] = R.tradeoff(s, tau, length)
        rows[k + ' @ Jev schedule'] = R.tradeoff(on_schedule(s, stride), tau, length)
    print(f'  {"detector":42s}' + ''.join(f'  PFA<={lv:<4}: DR    ADD ' for lv in R.LEVELS))
    for k, (pfa, add, dr) in rows.items():
        print(f'  {k:42s}' + ''.join(f'        {R.at_level(pfa, dr, lv, add):5.2f} {R.at_level(pfa, add, lv, add):6.1f}'
                                     for lv in R.LEVELS))
    res = {k: {str(lv): {'DR': R.at_level(pf, dr, lv, ad), 'ADD': R.at_level(pf, ad, lv, ad)} for lv in R.LEVELS}
           for k, (pf, ad, dr) in rows.items()}
    if not MOCK:
        with open(os.path.join('runs', 'jev', f'{name}.json'), 'w') as f:
            json.dump({'name': name, 'stride': stride, 'windows': n, 'unit': d.unit, 'results': res}, f, indent=1)
    return res


def main():
    names = [a for a in sys.argv[1:] if not a.startswith('--') and not a.isdigit()] or list(SPEC)
    for n in names:
        if n not in SPEC:
            raise SystemExit(f'no Jev spec for {n!r}; choose from {", ".join(SPEC)}')
        run(n)


if __name__ == '__main__':
    main()

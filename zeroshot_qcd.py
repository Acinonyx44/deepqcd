"""
Zero-shot quickest change detection: decision models and a time-series foundation model, no training.

DeepQCD learns d_t = P(change has happened | x_1..x_t) from labelled episodes. This script asks pretrained
models for a d_t with no training at all, and scores them like every other detector in deepqcd_real.py
(same test windows, same PFA-matched operating points). Four backends:

  jev      TypeSafe AI's Jev, a "System One" decision model: a JSON state plus a typed yes/no question,
           answered with a probability, no text generated. Needs TYPESAFE_API_KEY.
  hf       the same interface rebuilt from an open-weight instruct model (default Qwen2.5-0.5B-Instruct,
           from Hugging Face): one forward pass on the same prompt, P(yes) read from the next-token logits
           over "Yes" vs "No". No generation and no key; runs locally (slow on CPU, fine on a GPU).
  openai   any OpenAI-compatible chat endpoint that returns logprobs (OpenAI, Groq, Together, Fireworks,
           vLLM, Ollama ...): max_tokens = 1 and P(yes) from the top logprobs. Needs OPENAI_API_KEY,
           optionally OPENAI_BASE_URL.
  chronos  Amazon's Chronos-Bolt, a zero-shot forecaster: at each decision time it forecasts the next block
           of every feature from the history, and a CUSUM accumulates how far the observations fall outside
           its predicted quantiles. Purely numeric, so it runs on every dataset in realdata.py.

    .venv/bin/python zeroshot_qcd.py --backend hf occupancy --mock      # plumbing, labelled stand-in
    .venv/bin/python zeroshot_qcd.py --backend jev --estimate           # calls and cost, nothing sent
    .venv/bin/python zeroshot_qcd.py --backend chronos                  # all datasets
    .venv/bin/python zeroshot_qcd.py --backend openai --model gpt-4o-mini occupancy pumpdump

Every backend is asked every --stride steps (dataset default), and its answer is held until the next one.
DeepQCD (one seed, trained) and the classical rivals are reported at full rate and "@ schedule", i.e.
allowed to alarm only at the same decision times: that is the like-for-like comparison. Text answers are
cached in runs/zeroshot/ by a hash of the request, so reruns cost nothing.

Jev wire format (official typesafe-sdk 0.7.2, called with the standard library):
    POST {TYPESAFE_BASE_URL or https://api.typesafe.ai}/v1/systemone, Authorization: Bearer <key>
    {"state": ..., "model": "jev-latest", "questions": {"changed": {"type": "noul", ...}}}
    -> {"answers": {"changed": {"type": "noul", "noul": 0.97}}, ...}
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
import realdata
from qcd import DeepQCD, NetDetector, decision_statistics, train

MOCK = '--mock' in sys.argv
ESTIMATE = '--estimate' in sys.argv


def _arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


BACKEND = _arg('--backend', 'jev')
MODEL = _arg('--model', {'jev': 'jev-latest', 'hf': 'Qwen/Qwen2.5-0.5B-Instruct', 'openai': 'gpt-4o-mini',
                         'chronos': 'amazon/chronos-bolt-small'}.get(BACKEND, ''))
N_WINDOWS = _arg('--windows', 60 if BACKEND == 'hf' else 200)
WORKERS = _arg('--workers', 16)
BATCH = _arg('--batch', 8)
HISTORY = 20  # most recent observations shown verbatim to the text models

# What the text models are told about each stream. Values are given standardized (normal ~ 0 mean, unit sd).
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


# ---------------------------------------------------------------- what the text models see

def state_at(x, t, names, ctx):
    """JSON state at time t (0-based, x[:t+1] observed): context, a summary of the start of the stream
    (the reference) and of the recent past, and the last HISTORY observations verbatim."""
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


def chat(state, q):
    """The same request as a chat prompt for the logprob backends (hf, openai)."""
    c = q['changed']
    return [{'role': 'system', 'content': 'You judge data streams. Answer with exactly one word: Yes or No.'},
            {'role': 'user', 'content': f'{json.dumps(state)}\n\nQuestion: {c["instructions"]}\n'
                                        f'Yes means: {c["criteria"]["true"]}\nNo means: {c["criteria"]["false"]}\n'
                                        'Answer Yes or No.'}]


# ---------------------------------------------------------------- text backends (all return P(yes))

class Cache:
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path):
                k, v = json.loads(line)
                self.d[k] = v

    @staticmethod
    def key(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def put(self, k, v):
        self.d[k] = v
        with open(self.path, 'a') as f:
            f.write(json.dumps([k, v]) + '\n')


def mock_answer(state):
    """Labelled stand-in for --mock: a logistic of how far the recent mean sits from the stream's start. It
    exercises the plumbing only; its numbers say nothing about any model."""
    a = np.array(state['start_of_stream']['mean'])
    b = np.array(next(v for k, v in state.items() if k.startswith('last_'))['mean'])
    return float(1 / (1 + np.exp(-4 * (np.sqrt(((b - a) ** 2).mean()) - 1))))


def _post(url, body, headers, tries=5):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                 headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                                          'User-Agent': 'deepqcd-zeroshot', **headers})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (408, 429) or e.code >= 500:
                time.sleep(float(e.headers.get('retry-after') or 2 ** i))
                continue
            raise SystemExit(f'API error {e.code} from {url}: {e.read()[:300]!r}')
        except (urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1:
                raise SystemExit(f'{url} unreachable ({e}); is the host allowed by the network policy?')
            time.sleep(2 ** i)


def _need(env):
    v = os.environ.get(env, '').strip()
    if not v and not MOCK and not ESTIMATE:
        raise SystemExit(f'{env} is not set (add it to the environment), or use --mock / --estimate')
    return v


class JevBackend:
    def __init__(self):
        self.base = os.environ.get('TYPESAFE_BASE_URL', '').strip() or 'https://api.typesafe.ai'
        self.key = _need('TYPESAFE_API_KEY')

    def request(self, state, q):
        return {'state': state, 'model': MODEL, 'questions': q}

    def answer_many(self, reqs):
        def one(body):
            out = _post(self.base.rstrip('/') + '/v1/systemone', body, {'Authorization': f'Bearer {self.key}'})
            return float(out['answers']['changed']['noul'])
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(one, reqs))


def _yes_no(pairs):
    """P(yes) from (token, logprob) pairs, pooling case and leading-space variants."""
    yes = sum(np.exp(lp) for tok, lp in pairs if tok.strip().lower() == 'yes')
    no = sum(np.exp(lp) for tok, lp in pairs if tok.strip().lower() == 'no')
    return float(yes / (yes + no)) if yes + no > 0 else 0.5


class OpenAIBackend:
    def __init__(self):
        self.base = os.environ.get('OPENAI_BASE_URL', '').strip() or 'https://api.openai.com/v1'
        self.key = _need('OPENAI_API_KEY')

    def request(self, state, q):
        return {'model': MODEL, 'messages': chat(state, q), 'max_tokens': 1, 'temperature': 0,
                'logprobs': True, 'top_logprobs': 10}

    def answer_many(self, reqs):
        def one(body):
            out = _post(self.base.rstrip('/') + '/chat/completions', body, {'Authorization': f'Bearer {self.key}'})
            top = out['choices'][0]['logprobs']['content'][0]['top_logprobs']
            return _yes_no([(c['token'], c['logprob']) for c in top])
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(one, reqs))


class HFBackend:
    """Jev's interface on an open-weight model: no generation, one forward pass, P(yes) from the logits."""

    def __init__(self):
        self.tok = self.model = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        vocab = self.tok.get_vocab()
        variants = lambda w: {w, w.lower(), ' ' + w, ' ' + w.lower(), 'Ġ' + w, 'Ġ' + w.lower(), '▁' + w, '▁' + w.lower()}
        self.yes = sorted({vocab[v] for v in variants('Yes') if v in vocab})
        self.no = sorted({vocab[v] for v in variants('No') if v in vocab})
        assert self.yes and self.no, 'could not find Yes/No tokens in the vocabulary'

    def request(self, state, q):
        return {'model': MODEL, 'messages': chat(state, q)}

    @torch.no_grad()
    def answer_many(self, reqs):
        if self.model is None:
            self._load()
        out = []
        for i in range(0, len(reqs), BATCH):
            texts = [self.tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True)
                     for r in reqs[i:i + BATCH]]
            enc = self.tok(texts, return_tensors='pt', padding=True).to(self.model.device)
            logp = torch.log_softmax(self.model(**enc).logits[:, -1].float(), -1)
            ly, ln = torch.logsumexp(logp[:, self.yes], -1), torch.logsumexp(logp[:, self.no], -1)
            out += torch.sigmoid(ly - ln).tolist()
            if i // BATCH % 10 == 0:
                print(f'    {i + len(texts)}/{len(reqs)}', flush=True)
        return out


def text_statistic(name, xs, length, stride, d):
    """d_t from a text backend: ask at every decision time, hold the answer until the next one."""
    ctx = SPEC[name]
    backend = {'jev': JevBackend, 'openai': OpenAIBackend, 'hf': HFBackend}[BACKEND]()
    times = [(i, t) for i in range(len(xs)) for t in range(stride - 1, int(length[i]), stride)]
    q = question(ctx)
    reqs = [backend.request(state_at(xs[i, :length[i]], t, ctx['features'], ctx), q) for i, t in times]
    cache = Cache(os.path.join('runs', 'zeroshot', f'{BACKEND}-{name}{"-mock" if MOCK else ""}.jsonl'))
    keys = [Cache.key([BACKEND, r]) for r in reqs]
    todo = [j for j, k in enumerate(keys) if k not in cache.d]
    if ESTIMATE:
        chars = np.mean([len(json.dumps(r)) for r in reqs[:200]])
        price = {'jev': 0.042, 'openai': 0.15}.get(BACKEND)
        cost = f' = ~${len(todo) * chars / 4 / 1e6 * price:.2f} at ${price} / M input tokens' if price else ''
        print(f'  {len(reqs)} calls ({len(todo)} not cached), ~{chars / 4:.0f} input tokens each, '
              f'~{len(todo) * chars / 4 / 1e6:.1f} M tokens{cost}')
        return None
    t0 = time.time()
    states = [r.get('state') or json.loads(r['messages'][1]['content'].split('\n\nQuestion:')[0]) for r in reqs]
    new = [mock_answer(states[j]) for j in todo] if MOCK else backend.answer_many([reqs[j] for j in todo])
    for j, v in zip(todo, new):
        cache.put(keys[j], v)
    print(f'  {len(reqs)} answers ({len(todo)} new) in {time.time() - t0:.0f}s')
    dz = np.full(xs.shape[:2], -np.inf)
    for (i, t), k in zip(times, keys):
        dz[i, t:t + stride] = cache.d[k]
    return dz


# ---------------------------------------------------------------- chronos backend

def chronos_statistic(xs, length, stride, min_context=16, cap=25.0, drift=2.0):
    """CUSUM of forecast surprise. At each decision time t the history up to the start of the latest block
    is the context; Chronos-Bolt forecasts the block's quantiles for every feature, and each observation
    scores z^2 with z = (x - median) / ((q90 - q10) / 2.563). The CUSUM adds mean_features(min(z^2, cap)) -
    drift per step (z^2 ~ 1 when the forecast is calibrated)."""
    n, L, P = xs.shape
    if not MOCK:
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map='cuda' if torch.cuda.is_available() else 'cpu',
                                                   torch_dtype=torch.float32)
    s = np.zeros(n)
    d = np.full((n, L), -np.inf)
    t0 = time.time()
    for t in range(stride - 1, L, stride):
        a = t - stride + 1  # first index of the block being scored
        act = np.flatnonzero((length > t) & (a >= min_context))
        if len(act) == 0:
            continue
        ctx = xs[act, :a].transpose(0, 2, 1).reshape(-1, a)  # (windows * features, a)
        if MOCK:  # labelled stand-in: last value as the median, recent spread as the scale
            med = np.repeat(ctx[:, -1:], stride, 1)
            half = np.repeat(1.2816 * ctx[:, -30:].std(1, keepdims=True), stride, 1)
            q10, q50, q90 = med - half, med, med + half
        else:
            q, _ = pipe.predict_quantiles(torch.as_tensor(ctx[:, -2048:]), prediction_length=stride,
                                          quantile_levels=[0.1, 0.5, 0.9])
            q10, q50, q90 = (q[..., k].numpy() for k in range(3))
        scale = np.maximum((q90 - q10) / 2.563, 1e-3)
        blk = xs[act, a:t + 1].transpose(0, 2, 1).reshape(-1, stride)
        z2 = np.minimum(((blk - q50) / scale) ** 2, cap).reshape(len(act), P, stride).mean(1)  # (act, stride)
        for j in range(stride):
            s[act] = np.maximum(0, s[act] + z2[:, j] - drift)
        d[act, t:t + stride] = s[act, None]
    print(f'  Chronos statistic computed in {time.time() - t0:.0f}s')
    return d


# ---------------------------------------------------------------- evaluation

def on_schedule(dstat, stride):
    """The detector still sees every observation but may only alarm at the decision times (stride-1,
    2*stride-1, ...): at each one it reports the largest d_t since the previous one."""
    out = np.full_like(dstat, -np.inf)
    for t0 in range(0, dstat.shape[1], stride):
        t1 = min(t0 + stride, dstat.shape[1])
        out[:, t1 - 1] = dstat[:, t0:t1].max(1)
    return out


def run(name):
    p = R.prepare(name)
    d = p['d']
    n = min(N_WINDOWS, len(p['xs']))
    xs, tau, length = p['xs'][:n], p['tau'][:n], p['length'][:n]
    stride = SPEC[name]['stride'] if name in SPEC else max(1, d.H // 10)
    who = f'{BACKEND}:{MODEL}'
    print(f'\n=== {d.label} ({name}): {n} test windows, {who} asked every {stride} {d.unit}')
    os.makedirs(os.path.join('runs', 'zeroshot'), exist_ok=True)
    if BACKEND == 'chronos':
        if ESTIMATE:
            print(f'  {len(range(stride - 1, xs.shape[1], stride))} forecast calls of up to {n * p["P"]} series each')
            return None
        dz = chronos_statistic(xs, length, stride)
    else:
        dz = text_statistic(name, xs, length, stride, d)
        if dz is None:
            return None

    stats, _ = R.rivals({**p, 'xs': xs})
    torch.manual_seed(0)
    net = DeepQCD(p['P'], 16 if p['P'] <= 10 else 32)
    print('  training DeepQCD (1 seed) on the same windows ...')
    train(net, p['xt'], p['yt'], p['xv'], p['yv'], epochs=R.EPOCHS, w=p['mt'], wv=p['mv'])
    stats = {'DeepQCD (trained)': decision_statistics(NetDetector(net, None), xs), **stats}

    label = f'{BACKEND} MOCK stand-in (not a model)' if MOCK else f'{who} (zero-shot)'
    rows = {label: R.tradeoff(dz, tau, length)}
    for k, s in stats.items():
        rows[k] = R.tradeoff(s, tau, length)
        rows[k + ' @ schedule'] = R.tradeoff(on_schedule(s, stride), tau, length)
    print(f'  {"detector":46s}' + ''.join(f'  PFA<={lv:<4}: DR    ADD ' for lv in R.LEVELS))
    for k, (pfa, add, dr) in rows.items():
        print(f'  {k[:46]:46s}' + ''.join(f'        {R.at_level(pfa, dr, lv, add):5.2f} {R.at_level(pfa, add, lv, add):6.1f}'
                                         for lv in R.LEVELS))
    res = {k: {str(lv): {'DR': R.at_level(pf, dr, lv, ad), 'ADD': R.at_level(pf, ad, lv, ad)} for lv in R.LEVELS}
           for k, (pf, ad, dr) in rows.items()}
    if not MOCK:
        with open(os.path.join('runs', 'zeroshot', f'{BACKEND}-{name}.json'), 'w') as f:
            json.dump({'name': name, 'backend': BACKEND, 'model': MODEL, 'stride': stride, 'windows': n,
                       'unit': d.unit, 'results': res}, f, indent=1)
    return res


def main():
    if BACKEND not in ('jev', 'hf', 'openai', 'chronos'):
        raise SystemExit(f'unknown backend {BACKEND!r}: jev, hf, openai or chronos')
    flags = {'--backend', '--model', '--windows', '--workers', '--batch'}
    args = sys.argv[1:]
    names = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] not in flags)]
    pool = list(realdata.LOADERS) if BACKEND == 'chronos' else list(SPEC)
    for n in names or pool:
        if n not in pool:
            raise SystemExit(f'{n!r} not available for {BACKEND}; choose from {", ".join(pool)}')
        run(n)


if __name__ == '__main__':
    main()

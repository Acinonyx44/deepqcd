"""
A synthetic E-mini-style futures market on the S&P 500, with six market makers whose stress unwinds are labelled.

Time runs in one-second steps over a 6.5-hour session. Prices live on a 0.25-point tick grid. The value of the index has
two parts:

    P_t  public value: a random walk calibrated to recent S&P 500 volatility (FRED SP500), seen by everyone at once
    I_t  private value: jumps at news times, known first to informed traders

Market makers quote around a common fair-value estimate F_t = P_t + G_t, where the flow-implied part G_t moves with
signed aggressive volume (price impact) and decays toward I_t at rate ETA:

    G_{t+1} = G_t + LAMBDA * (signed volume)_t + ETA * (I_t - G_t)

So informed buying after good news moves the price to where it belongs and stays there, while a market maker's forced
selling pushes the price below value, and the price comes back once the selling stops. That difference is what a
detector of forced unwinds could be worth.

Traders:
    noise      Poisson market orders, size 1 + geometric, side with slowly varying imbalance (herding)
    informed   after a news jump, market orders toward the new value until the price has caught up (label 0: a confounder)
    MM1..MM6   quote around F_t - KAPPA q (inventory skew); MM1-3 quote tight (always a 1-tick market, small size),
               MM4-6 wide (4 ticks, large size), so the inside spread is 1 tick almost always, as in E-mini futures

Stress events (label 1 while active). A market maker receives an inventory shock it must offload (for example the delta
of a client options trade done off-exchange), or loses its risk appetite, and then follows its own policy:

    MM1  tight  instant    market orders for the whole position within 1-3 s
    MM2  tight  twap       equal child market orders every second for 2-10 min (Almgren-Chriss with no urgency)
    MM3  tight  skew       passive: quotes only the exit side, two ticks through fair value, triple size, until flat
    MM4  wide   gamma      short-gamma hedging for 5-15 min: buys after up-ticks and sells after down-ticks (CTA-like)
    MM5  wide   lean       passive and slow: quotes only the exit side, a tick and a half through fair value
    MM6  wide   withdraw   pulls all quotes for 1-4 min, then resumes

The observer sees only the anonymous public tape and book, one row per second (FEATURES).

Use:  sim = simulate(n_seconds, seed, S0, ...)  ->  dict with X (n, 9), y (n,), mid (n,), events, news, inventory (n, 6),
      gap (n,): the price's distance from value in ticks, which only the simulator knows
"""
import numpy as np

TICK = 0.25
SESSION = 23400                     # seconds in a 6.5-hour session
FEATURES = ['mid return (ticks)', 'spread (ticks)', 'log best-bid size', 'log best-ask size', 'depth imbalance (3 levels)',
            'signed volume', 'log total volume', 'order-flow imbalance', 'trades']

LAMBDA = 0.008                      # points per contract of signed aggressive volume
ETA = 1 / 300                       # flow-implied price decays toward value with a 3.5-minute half-life
NEWS_RATE = 1 / 2400                # news jumps per second (about 10 a day)
NEWS_SD = 6.0                       # jump size, points
SHOCK_RATE = 1 / 1500               # stress events per second (about 15 a day)
NOISE_RATE = 2.0                    # noise market orders per second (before the intraday pattern)

MM = [  # name, style, half-spread (ticks), quote size, inventory skew (points per contract), policy
    ('MM1', 'tight', 0.0, 6, 0.005, 'instant'),
    ('MM2', 'tight', 0.0, 6, 0.005, 'twap'),
    ('MM3', 'tight', 0.0, 6, 0.005, 'skew'),
    ('MM4', 'wide', 2.0, 20, 0.003, 'gamma'),
    ('MM5', 'wide', 2.0, 20, 0.003, 'lean'),
    ('MM6', 'wide', 2.0, 20, 0.003, 'withdraw'),
]
POLICIES = [m[5] for m in MM]


def season(t):
    """Intraday activity: high at the open and the close (t in seconds since the open)."""
    return 0.7 + 0.9 * np.exp(-t / 1500) + 0.6 * np.exp(-(SESSION - t) / 1500)


def calibrate(path='data/mm/SP500.csv', window=60):
    """Last S&P 500 close and the daily volatility of the last `window` daily log-returns (FRED series SP500)."""
    import csv
    rows = [(d, float(v)) for d, v in list(csv.reader(open(path)))[1:] if v not in ('', '.')]
    p = np.array([v for _, v in rows])
    r = np.diff(np.log(p))[-window:]
    return {'date': rows[-1][0], 'close': float(p[-1]), 'sigma_daily': float(r.std())}


class Maker:
    def __init__(self, k, spec):
        self.k, (self.name, self.style, self.half, self.size, self.kappa, self.policy) = k, spec
        self.q, self.cash = 0, 0.0
        self.mode, self.until, self.child, self.acc, self.gamma = None, 0, 0.0, 0.0, 0.0

    def quotes(self, F, best_other_bid, best_other_ask):
        """(bid, bid size, ask, ask size); a missing side has size 0."""
        if self.mode == 'withdraw':
            return 0.0, 0, 0.0, 0
        r, size = F - self.kappa * self.q, self.size
        exit_only, shift = False, 0.0
        if self.mode in ('skew', 'lean'):
            shift = (2.0 if self.mode == 'skew' else 1.5) * TICK * np.sign(self.q)
            size, exit_only = self.size * (3 if self.mode == 'skew' else 1.5), True
        r -= shift
        if self.half == 0:                                       # tight: always a one-tick market around r
            bid = np.floor(r / TICK + 1e-9) * TICK
            ask = bid + TICK
        else:
            bid = np.floor((r - self.half * TICK) / TICK + 1e-9) * TICK
            ask = np.ceil((r + self.half * TICK) / TICK - 1e-9) * TICK
        if ask - bid < TICK:
            ask = bid + TICK
        bid = min(bid, best_other_ask - TICK) if best_other_ask > 0 else bid   # never cross the others
        ask = max(ask, best_other_bid + TICK) if best_other_bid > 0 else ask
        bs, as_ = int(size), int(size)
        if exit_only:
            if self.q > 0:
                bs = 0
            else:
                as_ = 0
        return bid, bs, ask, as_


def simulate(n, seed, S0, sigma_daily, t0=0, shock_rate=SHOCK_RATE, news_rate=NEWS_RATE, force=None, force_news=None,
             quiet_before=None, kappa_scale=1.0):
    """n seconds of the market starting at second t0 of the session. force=(t, k): start a stress event for maker k at
    second t; force_news=t: a news jump at second t; quiet_before=t: no random stress events before t;
    kappa_scale multiplies every maker's inventory skew (smaller = more discreet quoting)."""
    rng = np.random.default_rng(seed)
    makers = [Maker(k, s) for k, s in enumerate(MM)]
    for m in makers:
        m.kappa *= kappa_scale
    sig_s = sigma_daily * S0 / np.sqrt(SESSION)                  # public value: points per second
    P, I, G = round(S0 / TICK) * TICK, 0.0, 0.0
    herd, volf = 0.0, 0.0
    X = np.zeros((n, len(FEATURES)), np.float32)
    y = np.zeros(n, np.int8)
    mid = np.zeros(n)
    inv = np.zeros((n, len(MM)), np.int32)
    gaps = np.zeros(n, np.float32)                               # (F - value) / tick: how far flow has pushed the price
    events, news = [], []
    prev_mid, prev_top = None, None
    lastF = P
    for i in range(n):
        tod = (t0 + i) % SESSION
        s = season(tod)
        # ---- values
        volf += -volf / 3600 + 0.01 * rng.standard_normal()
        P += sig_s * np.sqrt(s) * np.exp(volf) * rng.standard_normal()
        if (force_news is not None and i == force_news) or (force_news is None and rng.random() < news_rate):
            J = NEWS_SD * rng.standard_normal()
            I += J
            news.append({'t': i, 'jump': float(J)})
        G += ETA * (I - G)
        F = P + G
        dF = F - lastF
        lastF = F
        # ---- stress events
        start = None
        if force is not None and i == force[0]:
            start = force[1]
        elif force is None and (quiet_before is None or i >= quiet_before) and rng.random() < shock_rate:
            free = [m.k for m in makers if m.mode is None]
            start = int(rng.choice(free)) if free else None
        for m in makers:                                         # a hard inventory limit also forces the policy
            if m.mode is None and abs(m.q) > 500 and start is None and force is None:
                start = m.k
        if start is not None and makers[start].mode is None:
            m = makers[start]
            if m.policy != 'gamma':
                shock = int(rng.choice([-1, 1]) * rng.integers(150, 401))
                m.q += shock
                m.cash -= shock * F                              # the off-exchange trade, at fair value
            m.mode = m.policy
            if m.policy == 'instant':
                m.until, m.child = i + int(rng.integers(1, 4)), 0.0
            elif m.policy == 'twap':
                L = int(rng.integers(120, 601))
                m.until, m.child, m.acc = i + L, abs(m.q) / L, 0.0
            elif m.policy == 'gamma':
                m.until, m.gamma = i + int(rng.integers(300, 901)), float(rng.uniform(4, 10))
            elif m.policy in ('skew', 'lean'):
                m.until = i + (400 if m.policy == 'skew' else 900)
            elif m.policy == 'withdraw':
                m.until = i + int(rng.integers(60, 241))
            events.append({'mm': m.name, 'policy': m.policy, 'start': i, 'end': None, 'side': int(-np.sign(m.q)) if m.policy != 'gamma' else 0,
                           'size': abs(m.q) if m.policy != 'gamma' else 0})
        # ---- quotes
        book = []
        for m in makers:
            ob = max([b for (b, bs, a, as_, k) in book if bs > 0], default=0.0)
            oa = min([a for (b, bs, a, as_, k) in book if as_ > 0], default=0.0)
            b, bs, a, as_ = m.quotes(F, ob, oa)
            book.append([b, bs, a, as_, m.k])
        bids = sorted([(b, bs, k) for b, bs, a, as_, k in book if bs > 0], key=lambda z: -z[0])
        asks = sorted([(a, as_, k) for b, bs, a, as_, k in book if as_ > 0], key=lambda z: z[0])
        bb = bids[0][0] if bids else F - TICK
        ba = asks[0][0] if asks else F + TICK
        bbs = sum(z[1] for z in bids if z[0] == bb)
        bas = sum(z[1] for z in asks if z[0] == ba)
        dbid = sum(z[1] for z in bids if z[0] > bb - 3 * TICK)
        dask = sum(z[1] for z in asks if z[0] < ba + 3 * TICK)
        m_t = (bb + ba) / 2
        # ---- orders this second: (side +1 buy / -1 sell, size, owner maker or -1)
        orders = []
        herd += -herd / 60 + 0.15 * rng.standard_normal()
        pb = 0.5 + 0.1 * np.tanh(herd)
        for _ in range(rng.poisson(NOISE_RATE * s)):
            orders.append((1 if rng.random() < pb else -1, 1 + rng.geometric(0.35) - 1, -1))
        gap = I - G
        if abs(gap) > TICK:
            for _ in range(rng.poisson(1.2)):
                orders.append((int(np.sign(gap)), 1 + rng.geometric(0.25) - 1, -1))
        for m in makers:
            if m.mode == 'instant':
                left = max(1, m.until - i + 1)
                v = int(np.ceil(abs(m.q) / left))
                if v > 0:
                    orders.append((-int(np.sign(m.q)), v, m.k))
            elif m.mode == 'twap':
                m.acc += m.child
                v = min(int(m.acc), abs(m.q))
                m.acc -= v
                if v > 0:
                    orders.append((-int(np.sign(m.q)), v, m.k))
            elif m.mode == 'gamma':
                v = int(round(m.gamma * abs(dF) / TICK))
                if v > 0:
                    orders.append((int(np.sign(dF)), v, m.k))
        rng.shuffle(orders)
        # ---- matching against the posted quotes (depleted within the second)
        rem = {('b', k): bs for b, bs, a, as_, k in book}
        rem.update({('a', k): as_ for b, bs, a, as_, k in book})
        buy_v = sell_v = ntr = 0
        for side, size, owner in orders:
            levels = asks if side > 0 else bids
            need = size
            for price in sorted(set(z[0] for z in levels), key=(lambda p: p) if side > 0 else (lambda p: -p)):
                if need <= 0:
                    break
                at = [z for z in levels if z[0] == price and z[2] != owner and rem[('a' if side > 0 else 'b', z[2])] > 0]
                tot = sum(rem[('a' if side > 0 else 'b', z[2])] for z in at)
                if tot <= 0:
                    continue
                take = min(need, tot)
                for z in at:                                     # pro rata at the price level
                    key = ('a' if side > 0 else 'b', z[2])
                    f = min(rem[key], int(round(take * rem[key] / tot)) or 1, need)
                    if f <= 0:
                        continue
                    rem[key] -= f
                    need -= f
                    mk = makers[z[2]]
                    mk.q -= side * f
                    mk.cash += side * f * price
                    if owner >= 0:
                        makers[owner].q += side * f
                        makers[owner].cash -= side * f * price
                    if need <= 0:
                        break
            done = size - need
            if done > 0:
                ntr += 1
                if side > 0:
                    buy_v += done
                else:
                    sell_v += done
        G += LAMBDA * (buy_v - sell_v)
        # ---- end of stress modes
        for m in makers:
            if m.mode is None:
                continue
            fin = i >= m.until
            if m.mode in ('instant', 'twap', 'skew', 'lean') and abs(m.q) <= (0 if m.mode == 'instant' else 10):
                fin = True
            if fin:
                m.mode = None
                for e in reversed(events):
                    if e['mm'] == m.name and e['end'] is None:
                        e['end'] = i + 1
                        break
        active = any(e['end'] is None or e['end'] > i for e in events if e['start'] <= i)
        y[i] = 1 if active else 0
        # ---- public features
        ofi = 0.0
        if prev_top is not None:
            pbb, pbs, pba, pas = prev_top
            ofi = (bbs if bb >= pbb else 0) - (pbs if bb <= pbb else 0) - (bas if ba <= pba else 0) + (pas if ba >= pba else 0)
        prev_top = (bb, bbs, ba, bas)
        X[i] = [0.0 if prev_mid is None else (m_t - prev_mid) / TICK, (ba - bb) / TICK, np.log1p(bbs), np.log1p(bas),
                (dbid - dask) / max(dbid + dask, 1), buy_v - sell_v, np.log1p(buy_v + sell_v), ofi, ntr]
        prev_mid = m_t
        mid[i] = m_t
        inv[i] = [m.q for m in makers]
        gaps[i] = (G - I) / TICK
    for e in events:
        if e['end'] is None:
            e['end'] = n
    return {'X': X, 'y': y, 'mid': mid, 'events': events, 'news': news, 'inventory': inv, 'gap': gaps, 'pnl': [m.cash + m.q * mid[-1] for m in makers]}

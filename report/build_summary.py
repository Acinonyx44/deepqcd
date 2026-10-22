"""Assemble report/summary.html: the whole study in one paper-style page (reproduction, real-data transient, and the
finance problems), with a signals-and-systems block diagram of how each data stream is built.
Numbers come from runs/ and notes/; the two detector figures are reused from report/template.html.
Run:  .venv/bin/python report/build_summary.py"""
import json
import math
import os
import re
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GH = 'https://github.com/Acinonyx44/deepqcd/blob/claude/awesome-allen-8e9wvl/'
FONT = 'Source Sans 3, Helvetica Neue, Arial, sans-serif'


def J(*p):
    return json.load(open(os.path.join(REPO, 'runs', *p)))


# ------------------------------------------------------------------ SVG helpers
def box(x, y, w, h, title, lines=(), bold=True):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="currentColor" stroke-width="1.2"/>'
    cy = y + 15
    s += f'<text x="{x + w / 2}" y="{cy}" text-anchor="middle" font-size="12" font-weight="{600 if bold else 400}">{escape(title)}</text>'
    for k, ln in enumerate(lines):
        s += f'<text x="{x + w / 2}" y="{cy + 13 * (k + 1)}" text-anchor="middle" font-size="10.5" fill-opacity="0.8">{escape(ln)}</text>'
    return s


def arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="currentColor" stroke-width="1.2"{d} marker-end="url(#ah)"/>'


def text(x, y, s, size=11, anchor='start', weight=400, op=1.0, italic=False):
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill-opacity="{op}"{st}>'
            f'{escape(s)}</text>')


DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>')


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" font-family="{FONT}" fill="currentColor">'
            f'{DEFS}{body}</svg>')


def card(sp):
    """Signals-and-systems view of one problem: raw records -> entity -> clock -> features -> x[n]; hindsight rule ->
    y[n]; a bit strip showing what tau1 and tau2 mean; the split."""
    b = ''
    b += box(8, 8, 156, 52, 'Source', sp['source'])
    b += arrow(164, 34, 186, 34)
    b += box(188, 8, 112, 52, 'Entity', sp['entity'])
    b += arrow(300, 34, 322, 34)
    b += box(324, 8, 112, 52, 'Clock', sp['clock'])
    b += arrow(436, 34, 458, 34)
    b += box(460, 8, 236, 52, 'Features φ(·)', sp['features'])
    b += arrow(696, 34, 744, 34)
    b += text(748, 31, 'x[n] ∈ ℝ', 12, weight=600) + text(748, 46, f'P = {sp["P"]}', 10.5, op=0.8)
    b += box(8, 78, 292, 50, 'Event rule (hindsight)', sp['event'])
    b += arrow(300, 103, 744, 103, dash=True)
    b += text(748, 100, 'y[n] ∈ {0, 1}', 12, weight=600) + text(748, 115, 'labels', 10.5, op=0.8)
    b += box(632, 140, 184, 50, 'Split', sp['split'], bold=True)
    # bit strip
    bits = sp.get('bits', [0] * 7 + [1] * 5 + [0] * 4)
    x0, dx, base, top = 70, 34, 210, 168
    b += text(60, 192, 'y[n]', 11.5, 'end', 600)
    b += f'<line x1="{x0 - 6}" y1="{base}" x2="{x0 + dx * len(bits)}" y2="{base}" stroke="currentColor" stroke-width="1"/>'
    on = [k for k, v in enumerate(bits) if v]
    if on and on[0] > 0:
        xa, xb = x0 + dx * on[0] - dx / 2, x0 + dx * (on[-1] + 1) - dx / 2
        b += f'<rect x="{xa}" y="{top - 14}" width="{xb - xa}" height="{base - top + 18}" fill="currentColor" fill-opacity="0.07"/>'
        for xx, lab, t in ((xa, 'τ₁', sp['tau1']), (xb, 'τ₂', sp['tau2'])):
            b += f'<line x1="{xx}" y1="{top - 18}" x2="{xx}" y2="{base + 6}" stroke="currentColor" stroke-dasharray="3 3" stroke-opacity="0.7"/>'
            b += text(xx, top - 22, lab, 12, 'middle', 600)
        b += text(632, 210, f'τ₁: {sp["tau1"]}', 10.5, op=0.85)
        b += text(632, 225, f'τ₂: {sp["tau2"]}', 10.5, op=0.85)
    else:
        for k, ln in enumerate(sp['tau1'].split('\n')):
            b += text(632, 210 + 15 * k, ln, 10.5, op=0.85)
    for k, v in enumerate(bits):
        xx = x0 + dx * k
        yy = top if v else base - 2
        b += f'<line x1="{xx}" y1="{base}" x2="{xx}" y2="{yy}" stroke="currentColor" stroke-width="1.6"/>'
        b += f'<circle cx="{xx}" cy="{yy}" r="3.6" fill="currentColor"/>'
        b += text(xx, base + 16, str(v), 10.5, 'middle', op=0.75)
    b += text(x0, 238, sp.get('strip', 'test stream: real history before the event, then the event; no splicing'), 10.5, op=0.75, italic=True)
    return svg(820, 246, b, f'Data construction for {sp["name"]}: ' + '; '.join(sp['source'] + sp['features'] + sp['event']))


def pipeline():
    b = ''
    b += box(8, 20, 150, 56, 'Raw records', ['trades, transfers,', 'card swipes, prices'])
    b += arrow(158, 48, 180, 48)
    b += box(182, 20, 118, 56, 'Split by entity', ['market, account,', 'card, stock'])
    b += arrow(300, 48, 322, 48)
    b += box(324, 20, 118, 56, 'Clock Δ', ['aggregate per s, min,', 'h, day or event'])
    b += arrow(442, 48, 464, 48)
    b += box(466, 20, 118, 56, 'Feature map φ', ['what the business', 'already computes'])
    b += arrow(584, 48, 606, 48)
    b += box(608, 20, 118, 56, 'Standardize', ['mean, sd of normal', 'training data'])
    b += arrow(726, 48, 770, 48)
    b += text(774, 45, 'x[n]', 12.5, weight=600)
    b += box(182, 104, 260, 50, 'Hindsight event rule', ['what happened later marks [τ₁, τ₂)', '(price run, fraud, laundering, …)'])
    b += arrow(442, 129, 770, 129, dash=True)
    b += text(774, 126, 'y[n]', 12.5, weight=600)
    b += f'<line x1="240" y1="76" x2="240" y2="104" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/>'
    b += box(8, 184, 214, 56, 'Training windows', ['fixed length L, events at', 'random offsets, plus normal'])
    b += box(240, 184, 250, 56, 'Native test streams', ['PRE steps of real history + the event;', 'event-free windows of the same length'])
    b += box(508, 184, 218, 56, 'Split by time / entity', ['test events after training ones', '(or other banks, markets, cards)'])
    b += f'<line x1="790" y1="58" x2="790" y2="212" stroke="currentColor" stroke-width="1" stroke-opacity="0.5"/>'
    b += arrow(790, 212, 728, 212)
    return svg(820, 252, b, 'Generic data pipeline: raw records split by entity, aggregated on a clock, mapped to features and '
                            'standardized give x[n]; a hindsight rule gives the labels y[n]; windows are cut for training and '
                            'native test streams, split by time.')


def rare_fig():
    b = ''
    b += box(8, 26, 150, 54, 'DeepQCD', ['trained where changes', 'are common (ρ_tr)'])
    b += arrow(158, 53, 182, 53)
    b += text(170, 44, 'd[n]', 11, 'middle', 600)
    b += box(184, 26, 96, 54, 'logit', ['z = log d/(1−d)'])
    b += arrow(280, 53, 304, 53)
    b += box(306, 26, 150, 54, 'Calibrate', ['a·z + b, fitted on', 'labelled data at ρ_tr'])
    b += arrow(456, 53, 480, 53)
    b += box(482, 26, 140, 54, 'Shift the prior', ['+ log(ρ_dep / ρ_tr)'])
    b += arrow(622, 53, 646, 53)
    b += box(648, 26, 96, 54, '≥ 1 − α ?', ['alarm a[n]'])
    b += text(8, 112, 'Raw output read as a probability: PFA 0.137 at α = 0.1 (too many alarms).  Shift only: 0.004 (25× too strict).', 11, op=0.85)
    b += text(8, 130, 'Calibrate, then shift: 0.028, below α, with no labelled data needed at the rare rate.', 11, op=0.85)
    return svg(820, 140, b, 'Rare-event deployment: network output, logit, calibration at the training rate, prior shift, threshold')


def mm_fig():
    b = ''
    b += box(8, 30, 120, 50, 'Taker trades', ['Polymarket, 298', 'held-out markets'])
    b += arrow(128, 55, 150, 55)
    b += box(152, 30, 112, 50, 'Minute features', ['9 anonymous'])
    b += arrow(264, 45, 290, 22)
    b += arrow(264, 65, 290, 88)
    b += box(292, 4, 170, 40, 'Metaorder alarm', ['WL-CUSUM'])
    b += box(292, 70, 170, 40, 'Churn alarm', ['DeepQCD'])
    b += arrow(462, 24, 490, 46)
    b += arrow(462, 90, 490, 66)
    b += box(492, 30, 130, 50, 'Quote switch', ['pull hit side / both'])
    b += arrow(622, 55, 646, 55)
    b += box(648, 30, 96, 50, 'Fills', ['our side −s'])
    b += arrow(744, 55, 768, 55)
    b += text(772, 52, 'markout', 11, weight=600) + text(772, 66, '5 min', 10.5, op=0.8)
    b += f'<polyline points="196,30 196,14 290,14" fill="none" stroke="currentColor" stroke-width="0" />'
    b += text(8, 132, 'Decisions at minute m use data up to m − 1. Thresholds: each alarm on in 2% of quiet training minutes.', 11, op=0.85)
    return svg(820, 142, b, 'Market-making replay: trades to features to two alarms to a quote switch to fills to markouts')


def dotplot(rows):
    """rows: (label, deep, other, other_name, pfa, verdict)"""
    W, rh, top, left, right = 820, 24, 30, 300, 40
    H = top + rh * len(rows) + 40
    sx = lambda v: left + v * (W - left - right)
    b = ''
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        X = sx(v)
        b += f'<line x1="{X}" y1="{top - 8}" x2="{X}" y2="{top + rh * len(rows)}" stroke="currentColor" stroke-opacity="0.15"/>'
        b += text(X, top + rh * len(rows) + 16, f'{v:g}', 10.5, 'middle', op=0.8)
    b += text(sx(0.5), top + rh * len(rows) + 34, 'probability of detection (PD) at the stated false-alarm level', 11, 'middle', op=0.85)
    col = {'win': '#00704F', 'tie': '#4D4D4D', 'loss': '#B2182B'}
    for k, (lab, d, o, on, pfa, vd) in enumerate(rows):
        y = top + rh * k + 12
        b += text(left - 10, y + 4, f'{lab}', 11.5, 'end')
        b += text(left - 10, y + 4, '', 11)
        b += f'<line x1="{sx(min(d, o))}" y1="{y}" x2="{sx(max(d, o))}" y2="{y}" stroke="{col[vd]}" stroke-width="2.2"/>'
        b += f'<circle cx="{sx(o)}" cy="{y}" r="4.5" fill="#FFFFFF" stroke="#4D4D4D" stroke-width="1.6"/>'
        b += f'<circle cx="{sx(d)}" cy="{y}" r="5" fill="{col[vd]}"/>'
    b += f'<circle cx="{left}" cy="12" r="5" fill="#00704F"/>' + text(left + 10, 16, 'DeepQCD (colour = verdict)', 11)
    b += f'<circle cx="{left + 210}" cy="12" r="4.5" fill="#FFFFFF" stroke="#4D4D4D" stroke-width="1.6"/>' + text(left + 220, 16, 'best competitor', 11)
    return svg(W, H, b, 'Dot plot of DeepQCD against the best competitor for every study')


def scatter(pts):
    """pts: (label, n_train_events, advantage, kind) kind in {'combo', 'single'}"""
    W, H, l, r, t, bm = 820, 330, 60, 20, 20, 50
    lx = lambda n: l + (math.log10(n) - math.log10(5)) / (math.log10(20000) - math.log10(5)) * (W - l - r)
    ly = lambda a: t + (0.3 - a) / 0.6 * (H - t - bm)
    b = f'<rect x="{l}" y="{t}" width="{W - l - r}" height="{H - t - bm}" fill="none" stroke="currentColor" stroke-opacity="0.6"/>'
    for n in (10, 100, 1000, 10000):
        X = lx(n)
        b += f'<line x1="{X}" y1="{t}" x2="{X}" y2="{H - bm}" stroke="currentColor" stroke-opacity="0.12"/>'
        b += text(X, H - bm + 16, f'{n:,}', 10.5, 'middle', op=0.8)
    for a in (-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3):
        Y = ly(a)
        b += f'<line x1="{l}" y1="{Y}" x2="{W - r}" y2="{Y}" stroke="currentColor" stroke-opacity="{0.45 if a == 0 else 0.12}"/>'
        b += text(l - 6, Y + 4, f'{a:+.1f}' if a else '0', 10.5, 'end', op=0.8)
    b += text((l + W - r) / 2, H - 12, 'labelled training events (log scale)', 11, 'middle', op=0.85)
    b += f'<text x="14" y="{(t + H - bm) / 2}" transform="rotate(-90 14 {(t + H - bm) / 2})" text-anchor="middle" font-size="11" fill-opacity="0.85">PD advantage over the best competitor</text>'
    for lab, n, a, kind, dx, dy in pts:
        X, Y = lx(n), ly(a)
        if kind == 'combo':
            b += f'<circle cx="{X}" cy="{Y}" r="5.5" fill="#009E73"/>'
        else:
            b += f'<circle cx="{X}" cy="{Y}" r="5" fill="#FFFFFF" stroke="#222222" stroke-width="1.6"/>'
        b += text(X + dx, Y + dy, lab, 10.5, 'start' if dx >= 0 else 'end', op=0.9)
    b += f'<circle cx="{l + 12}" cy="{t + 14}" r="5.5" fill="#009E73"/>' + text(l + 22, t + 18, 'the event shows as a combination of cues', 11)
    b += f'<circle cx="{l + 12}" cy="{t + 32}" r="5" fill="#FFFFFF" stroke="#222222" stroke-width="1.6"/>' + text(l + 22, t + 36, 'one statistic carries the event', 11)
    return svg(W, H, b, 'Scatter of DeepQCD advantage against the number of labelled training events')


# ------------------------------------------------------------------ data
def numbers():
    U = {k: J('usecases', f'{k}.json') for k in ('toxic', 'amlworld', 'mm_pnl', 'squeeze', 'credit', 'fraud')}
    A = J('aml', 'results.json')
    ME, CH = J('polymarket_meta', 'minute', 'results.json'), J('polymarket_meta', 'churn_minute', 'results.json')
    PN = J('polymarket', 'results.json')
    R = J('rare', 'results.json')
    return U, A, ME, CH, PN, R


def table(head, rows, cls=None):
    h = '<div class="tablewrap"><table><tr>' + ''.join(f'<th{" class=\"num\"" if i else ""}>{c}</th>' for i, c in enumerate(head)) + '</tr>'
    for r in rows:
        mid = r[0].startswith('|')
        cells = [r[0].lstrip('|')] + list(r[1:])
        h += f'<tr{" class=\"mid\"" if mid else ""}>' + ''.join(f'<td{" class=\"num\"" if i else ""}>{c}</td>' for i, c in enumerate(cells)) + '</tr>'
    return h + '</table></div>'


def f2(v):
    return '—' if v is None else f'{v:.2f}'


def main():
    U, A, ME, CH, PN, R = numbers()
    tpl = open(os.path.join(HERE, 'template.html')).read()
    style = re.search(r'<style>.*?</style>', tpl, re.S).group(0)
    figs = re.findall(r'<svg viewBox="0 (?:-10 820 482|0 820 300)".*?</svg>', tpl, re.S)
    assert len(figs) == 2
    D = lambda k: U[k]['detectors']
    pdx = lambda d, a: d['pd_at'][a]
    seeds = lambda d, a: '(' + ', '.join(f'{v:.2f}' for v in d['pd_at_seeds'][a]) + ')'

    cards = {
        'real': dict(name='real-data transient', source=['HAI, SMD, SKAB, FoG,', 'Occupancy recordings'], entity=['one recording'],
                     clock=['Δ = 1 sample'], features=['all sensors (or the Gaussian-LLR z)', 'standardized by normal training data'],
                     P='5 – 86', event=['labelled anomaly interval;', 'its first K = 25 samples'],
                     split=['blocks: spliced (IoT code)', 'native: one recording'], tau1='event onset (τ₁ = 1000)',
                     tau2='τ₁ + K', strip='blocks: normal data from other times glued before and after the event; native: one real recording'),
        'mmsim': dict(name='synthetic market makers', source=['agent-based market;', 'S&P level and vol (FRED)'], entity=['one market,', '6 makers + news'],
                      clock=['Δ = 1 second'], features=['signed volume, depth, imbalance,', 'spread, quote moves'], P=9,
                      event=['label 1 while any maker is', 'under stress (simulator truth)'], split=['30 days train, 10 test', 'τ₁ = 1200 s, K = 120 s'],
                      tau1='stress starts', tau2='stress ends'),
        'pmnews': dict(name='Polymarket news', source=['Polymarket prices + trades', '≥ $1,000 with wallets'], entity=['one deadline', 'market'],
                       clock=['Δ = 1 hour'], features=['price, Δlogit, signed and gross flow,', 'new / fresh-wallet flow, concentration'], P=8,
                       event=['τ_pub: YES ≥ 0.8 until close;', 'window = the 24 h before'], split=['closed before Feb 2026 train', '30 surprise events test'],
                       tau1='τ_pub − 24 h', tau2='τ_pub (public move)'),
        'amlsim': dict(name='AMLSim', source=['IBM AMLSim (built', 'from source), 6 banks'], entity=['one account'], clock=['Δ = 1 day'],
                       features=['transfers and amounts in / out,', 'new and distinct counterparties, hub'], P=9,
                       event=['first laundering transfer', 'to 3 days after the last'], split=['banks 1–4 train, 5–6 test', '2,394 laundering accounts'],
                       tau1='first laundering transfer', tau2='last + 3 days'),
        'flow': dict(name='Polymarket wallet labels', source=['Polymarket trades with', 'wallets: 750 markets'], entity=['one market'], clock=['Δ = 1 minute'],
                     features=['anonymous: trades, signed $, sizes,', 'one-sidedness, same sizes, gaps'], P=9,
                     event=['one wallet: ≥ 5 same-side ≥ $2k', '(metaorder) or ≥ 4 flips (churn)'], split=['447 markets train, 298 test', '865 metaorders, 206 churn'],
                     tau1="the wallet's first trade", tau2="the wallet's last trade"),
        'toxic': dict(name='crypto toxic flow', source=['Binance ETHUSDT perp', 'aggregated trades'], entity=['one market'], clock=['Δ = 1 second'],
                      features=['trades, signed / gross notional, largest', 'trade, levels, returns, imbalance, vol'], P=10,
                      event=['fast run: zigzag swing ≥ 15 bps', 'completed within 60 s'], split=['Jul–Sep train (10,776 runs)', 'Oct 1–20 test (360 runs)'],
                      tau1='the swing starts', tau2='the swing ends'),
        'fraud': dict(name='card fraud', source=['IBM TabFormer, 24.4M', 'card transactions'], entity=['one card'], clock=['Δ = 1 transaction'],
                      features=['amount, online, new merchant / city /', 'state, errors, gap, velocity, hour'], P=15,
                      event=['burst: frauds at most 20', 'transactions apart'], split=['before 2016 train (3,104)', '2016 on test (1,680)'],
                      tau1='first fraud', tau2='last fraud + 1'),
        'amlworld': dict(name='AMLworld', source=['IBM AMLworld HI-Small,', '5.1M transfers'], entity=['one account'], clock=['Δ = 1 transaction'],
                         features=['direction, USD, payment format, cross-', 'currency / bank, new counterparty, hub'], P=15,
                         event=['first laundering transfer to', 'one transfer after the last'], split=['before 6 Sep train (2,799)', 'after it test (1,676)'],
                         tau1='first laundering transfer', tau2='last + 1'),
        'squeeze': dict(name='short squeezes', source=['Yahoo daily prices + FINRA', 'short volume, 1,398 stocks'], entity=['one stock'],
                        clock=['Δ = 1 trading day'], features=['return, volume ratio, short share and', 'jump, range, gap, 5/20-day return, vol'], P=10,
                        event=['rally ≥ 50% within 10 days', 'on ≥ 3× normal volume'], split=['before 2025 train (968)', '2025 on test (260)'],
                        tau1='the day after the low', tau2='the day after the peak'),
        'credit': dict(name='credit default', source=['UCI default of credit', 'card clients (30,000)'], entity=['one client'], clock=['Δ = 1 month (6)'],
                       features=['months late, utilization, bill,', 'payment, payment ratio, limit, age'], P=8,
                       event=['defaults in month 7: label 1', 'in every month (Bayesian)'], split=['70% of clients train', '30% test (2,019 defaulters)'],
                       bits=[1, 1, 1, 1, 1, 1], tau1='defaulter: 1 in all six months;\nothers: 0 in all six',
                       strip='one stream per client, April to September; an alarm in any month flags the client'),
    }
    C = {k: card(v) for k, v in cards.items()}

    # numbers for the overview figures
    dt, df, da, dsq, dcr = D('toxic'), D('fraud'), D('amlworld'), D('squeeze'), D('credit')
    rows = [
        ('Crypto toxic flow (PFA 0.05)', pdx(dt['DeepQCD'], '0.05'), pdx(dt['Rule: price move, 5 s'], '0.05'), 'price rule', '0.05', 'win'),
        ('Card fraud (PFA 0.01)', pdx(df['DeepQCD'], '0.01'), pdx(df['Logistic (this step only)'], '0.01'), 'logistic', '0.01', 'win'),
        ('AMLworld laundering (PFA 0.01)', pdx(da['DeepQCD'], '0.01'), pdx(da['Logistic (this step only)'], '0.01'), 'logistic', '0.01', 'win'),
        ('AMLSim laundering (PFA 0.05)', pdx(A['detectors']['DeepQCD'], '0.05'), pdx(A['detectors']['Rules combined'], '0.05'), 'rules', '0.05', 'win'),
        ('Polymarket churn (PFA 0.05)', pdx(CH['detectors']['DeepQCD'], '0.05'), pdx(CH['detectors']['WL-CUSUM'], '0.05'), 'CUSUM', '0.05', 'win'),
        ('Synthetic MM, passive skew', 0.92, 0.06, 'CUSUM', '0.05', 'win'),
        ('Polymarket metaorders (PFA 0.05)', pdx(ME['detectors']['DeepQCD'], '0.05'), pdx(ME['detectors']['WL-CUSUM'], '0.05'), 'CUSUM', '0.05', 'tie'),
        ('Paper transient, Fig. 9 (PFA 0.05)', 0.87, 0.86, 'WL-CUSUM', '0.05', 'tie'),
        ('Short squeezes (PFA 0.05)', pdx(dsq['DeepQCD'], '0.05'), pdx(dsq['WL-CUSUM'], '0.05'), 'CUSUM', '0.05', 'tie'),
        ('Card default (PFA 0.05)', pdx(dcr['DeepQCD'], '0.05'), pdx(dcr['WL-CUSUM'], '0.05'), 'CUSUM', '0.05', 'tie'),
        ('AMLworld laundering (PFA 0.05)', pdx(da['DeepQCD'], '0.05'), pdx(da['Rule: ACH with a new counterparty'], '0.05'), 'ACH rule', '0.05', 'tie'),
        ('SKAB sensors (PFA 0.05)', 0.15, 0.15, 'WL-CUSUM', '0.05', 'tie'),
        ('Polymarket news (PFA 0.05)', PN['detectors']['DeepQCD, all features']['pd_at']['0.05'], PN['detectors']['Price rise over 24 h']['pd_at']['0.05'], 'price rule', '0.05', 'tie'),
        ('Occupancy sensors (PFA 0.05)', 0.95, 1.00, 'WL-CUSUM', '0.05', 'tie'),
        ('HAI sensors (PFA 0.05)', 0.43, 0.51, 'WL-CUSUM', '0.05', 'loss'),
        ('SMD sensors (PFA 0.05)', 0.03, 0.26, 'WL-CUSUM', '0.05', 'loss'),
        ('Synthetic MM, instant unwind', 0.54, 0.94, 'CUSUM', '0.05', 'loss'),
    ]
    F1 = dotplot(rows)
    sc = [('toxic flow', 10776, pdx(dt['DeepQCD'], '0.05') - pdx(dt['Rule: price move, 5 s'], '0.05'), 'combo', -10, -8),
          ('card fraud @1%', 3104, pdx(df['DeepQCD'], '0.01') - pdx(df['Logistic (this step only)'], '0.01'), 'combo', -10, -8),
          ('AMLworld @1%', 2799, pdx(da['DeepQCD'], '0.01') - pdx(da['Logistic (this step only)'], '0.01'), 'combo', -10, 14),
          ('AMLSim', 6442, pdx(A['detectors']['DeepQCD'], '0.05') - pdx(A['detectors']['Rules combined'], '0.05'), 'combo', 9, 4),
          ('churn', 651, pdx(CH['detectors']['DeepQCD'], '0.05') - pdx(CH['detectors']['WL-CUSUM'], '0.05'), 'combo', 9, 4),
          ('metaorders', 4634, pdx(ME['detectors']['DeepQCD'], '0.05') - pdx(ME['detectors']['WL-CUSUM'], '0.05'), 'single', 9, 4),
          ('squeezes', 968, pdx(dsq['DeepQCD'], '0.05') - pdx(dsq['WL-CUSUM'], '0.05'), 'single', 9, -6),
          ('credit', 4645, pdx(dcr['DeepQCD'], '0.05') - pdx(dcr['WL-CUSUM'], '0.05'), 'single', 9, 14),
          ('Polymarket news', 34, PN['detectors']['DeepQCD, all features']['pd_at']['0.05'] - PN['detectors']['Price rise over 24 h']['pd_at']['0.05'], 'single', 9, 4),
          ('HAI', 30, -0.08, 'single', 9, 4), ('SMD', 155, -0.23, 'single', 9, 4), ('SKAB', 20, 0.0, 'single', 9, -6),
          ('Occupancy', 7, -0.05, 'single', 9, 4), ('FoG', 151, 0.06, 'single', 9, -6)]
    F2 = scatter(sc)

    ms = U['mm_pnl']['strategies']
    sec = {}
    sec['summary_table'] = table(
        ['Study', 'Data (entity, clock)', 'Labelled events', 'DeepQCD vs best other', 'Verdict'],
        [['Paper §5 (synthetic)', 'simulated streams, step', 'unlimited', 'Bayesian within 6% of Shiryaev; transient 0.87 vs 0.86', 'reproduced; ties the optimum'],
         ['Real sensors (Prof. Wang)', 'recording, sample', '7–155', 'HAI 0.43 vs 0.51, SMD 0.03 vs 0.26, SKAB 0.15 vs 0.15', '<span class="n">loss</span>'],
         ['|Synthetic market makers', 'market, second', 'simulated', 'skew 0.92 vs 0.06; instant 0.54 vs 0.94', 'mixed'],
         ['Rare changes', 'synthetic, step', 'unlimited', 'train at 10× rate costs nothing; calibrate + shift gives PFA ≤ α', '<span class="y">works</span>'],
         ['Polymarket news', 'market, hour', '34', f'{f2(PN["detectors"]["DeepQCD, all features"]["pd_at"]["0.05"])} vs {f2(PN["detectors"]["Price rise over 24 h"]["pd_at"]["0.05"])} (price rule)', 'tie; no trading edge'],
         ['AMLSim', 'account, day', '6,442', f'{f2(pdx(A["detectors"]["DeepQCD"], "0.05"))} vs {f2(pdx(A["detectors"]["Rules combined"], "0.05"))} (rules)', '<span class="y">win</span>'],
         ['Polymarket metaorders', 'market, minute', '4,634', f'{f2(pdx(ME["detectors"]["DeepQCD"], "0.05"))} vs {f2(pdx(ME["detectors"]["WL-CUSUM"], "0.05"))}; CUSUM saves more', 'tie'],
         ['Polymarket churn', 'market, minute', '651', f'{f2(pdx(CH["detectors"]["DeepQCD"], "0.05"))} vs {f2(pdx(CH["detectors"]["WL-CUSUM"], "0.05"))}', '<span class="y">win</span>'],
         ['|Crypto toxic flow', 'market, second', '10,776', f'{f2(pdx(dt["DeepQCD"], "0.05"))} vs {f2(pdx(dt["Rule: price move, 5 s"], "0.05"))}; 2.6 vs 1.6 bps saved per run', '<span class="y">win</span>'],
         ['Card fraud', 'card, transaction', '3,104', f'at PFA 0.01: {f2(pdx(df["DeepQCD"], "0.01"))} vs {f2(pdx(df["Logistic (this step only)"], "0.01"))}', '<span class="y">win</span>'],
         ['AMLworld', 'account, transaction', '2,799', f'at 0.01: {f2(pdx(da["DeepQCD"], "0.01"))} vs {f2(pdx(da["Logistic (this step only)"], "0.01"))}; at 0.05: {f2(pdx(da["DeepQCD"], "0.05"))} vs {f2(pdx(da["Rule: ACH with a new counterparty"], "0.05"))}', '<span class="y">win at 1%</span>, tie at 5%'],
         ['Polymarket market making', 'market, minute', '—', f'P&amp;L {ms["always quote"]["pnl_per_100usd_of_all_volume"]:+.1f}¢ → {ms["step back on either alarm"]["pnl_per_100usd_of_all_volume"]:+.2f}¢ per $100; busy rule {ms["step back when busy"]["pnl_per_100usd_of_all_volume"]:+.2f}¢', 'alarms help; DeepQCD adds little'],
         ['Short squeezes', 'stock, day', '968', f'{f2(pdx(dsq["DeepQCD"], "0.05"))} vs {f2(pdx(dsq["WL-CUSUM"], "0.05"))}', 'tie'],
         ['Card default', 'client, month', '≈ 4,600', f'{f2(pdx(dcr["DeepQCD"], "0.05"))} vs {f2(pdx(dcr["WL-CUSUM"], "0.05"))}; all ≈ 0.63 at PFA 0.2', 'tie']])

    def dtab(dd, names, alphas=('0.01', '0.05', '0.1'), seeds_on=True):
        rs = []
        for i, n in enumerate(names):
            d = dd[n]
            row = [('|' if i == 3 else '') + n.replace('Rule: ', 'rule: ')]
            for a in alphas:
                s = f'<b>{f2(d["pd_at"][a])}</b>' if n == 'DeepQCD' else f2(d['pd_at'][a])
                if n == 'DeepQCD' and seeds_on and a == '0.05' and 'pd_at_seeds' in d:
                    s += f' <span class="note">{seeds(d, a)}</span>'
                row.append(s)
            rs.append(row)
        return table(['Detector'] + [f'PFA {a}' for a in alphas], rs)

    T = {}
    T['toxic'] = dtab(dt, ['DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', 'Rule: price move, 5 s', 'Rule: trades, 10 s', 'Rule: signed flow, 5 s'])
    T['fraud'] = dtab(df, ['DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', 'Rules combined', 'Rule: new merchant, above usual amount', 'Rule: online, above usual amount'])
    T['amlworld'] = dtab(da, ['DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', 'Rule: ACH with a new counterparty', 'Rule: new counterparties, last 5'])
    T['squeeze'] = dtab(dsq, ['DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', 'Rule: 2-day return', 'Rule: volume spike'])
    T['credit'] = dtab(dcr, ['DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', 'Rule: months late now', 'Rule: utilization'])
    T['amlsim'] = dtab(A['detectors'], ['DeepQCD', 'WL-CUSUM', 'Rules combined', 'Rule: dormant account reactivated', 'Rule: new counterparties (3 days)'])
    T['flow'] = table(['Detector', 'metaorders, PFA 0.05', 'churn, PFA 0.05', 'churn, PFA 0.01'],
                      [['DeepQCD', f'<b>{f2(pdx(ME["detectors"]["DeepQCD"], "0.05"))}</b>', f'<b>{f2(pdx(CH["detectors"]["DeepQCD"], "0.05"))}</b>', f'<b>{f2(pdx(CH["detectors"]["DeepQCD"], "0.01"))}</b>'],
                       ['WL-CUSUM', f2(pdx(ME['detectors']['WL-CUSUM'], '0.05')), f2(pdx(CH['detectors']['WL-CUSUM'], '0.05')), f2(pdx(CH['detectors']['WL-CUSUM'], '0.01'))],
                       ['rule: trades, 10 min', f2(pdx(ME['detectors']['Trades, last 10 min'], '0.05')), f2(pdx(CH['detectors']['Trades, last 10 min'], '0.05')), f2(pdx(CH['detectors']['Trades, last 10 min'], '0.01'))],
                       ['rule: signed flow, 5 min', f2(pdx(ME['detectors']['Flow, last 5 min'], '0.05')), f2(pdx(CH['detectors']['Flow, last 5 min'], '0.05')), f2(pdx(CH['detectors']['Flow, last 5 min'], '0.01'))]])
    T['mm'] = table(['Strategy', 'volume filled', 'markout on fills (¢/share)', 'P&amp;L (¢ per $100 of all volume)'],
                    [[('|' if n == 'oracle' else '') + n, f'{100 * v["share_of_volume"]:.0f}%', f'{v["cents_per_share"]:+.3f}', f'{v["pnl_per_100usd_of_all_volume"]:+.2f}'] for n, v in ms.items()])
    PD = PN['detectors']
    T['pmnews'] = table(['Detector', 'PFA 0.05', 'buy YES at first alarm (¢ per contract)'],
                        [['DeepQCD', f'<b>{f2(PD["DeepQCD, all features"]["pd_at"]["0.05"])}</b>', f'{100 * PD["DeepQCD, all features"]["backtest"]["0.05"]["mean"]:+.1f}'],
                         ['WL-CUSUM', f2(PD['WL-CUSUM, all features']['pd_at']['0.05']), f'{100 * PD["WL-CUSUM, all features"]["backtest"]["0.05"]["mean"]:+.1f}'],
                         ['rule: price rise over 24 h', f2(PD['Price rise over 24 h']['pd_at']['0.05']), f'{100 * PD["Price rise over 24 h"]["backtest"]["0.05"]["mean"]:+.1f}'],
                         ['|buy at a random hour', '—', f'{100 * PN["backtest_random"]["mean"]:+.1f}']])

    html = open(os.path.join(HERE, 'summary_template.html')).read()
    rep = {'__STYLE__': style, '__FIG_BITS__': figs[0], '__FIG_DET__': figs[1], '__PIPE__': pipeline(), '__RARE__': rare_fig(),
           '__MMFIG__': mm_fig(), '__F1__': F1, '__F2__': F2, '__SUMTAB__': sec['summary_table'], '__GH__': GH}
    for k, v in C.items():
        rep[f'__C_{k.upper()}__'] = v
    for k, v in T.items():
        rep[f'__T_{k.upper()}__'] = v
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    left = re.findall(r'__[A-Z0-9_]+__', html)
    assert not left, left
    open(os.path.join(HERE, 'summary.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

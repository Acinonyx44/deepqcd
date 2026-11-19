"""Assemble report/formal.html: the five finance use cases (laundering, crypto toxic flow, card fraud, short squeezes,
card default) and the prediction-market studies, in the form of the study summary (one section per problem with its
data diagram, results table, PD-PFA curves and a reading).
The run outputs behind these results are not in git; the figures and tables come from the published summary and final
pages and are kept in report/formal_src (report/extract_published.py made them).
Run:  python report/build_formal.py"""
import json
import math
import os
import re
import sys
from html import unescape

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_final import fig_protocol  # noqa: E402
from build_summary import COL, GH, card, legend, pipeline, svg, text  # noqa: E402

SRC = os.path.join(HERE, 'formal_src')
T = json.load(open(os.path.join(SRC, 'tables.json')))
VCOL = {'win': '#00704F', 'tie': '#4D4D4D', 'loss': '#B2182B'}
LEARNERS = [('DeepQCD (ensemble)', COL['deep'], None, 2.4), ('gradient boosting', COL['gb'], None, 2),
            ('logistic', COL['lr'], None, 2), ('WL-CUSUM', COL['cusum'], None, 2), ('best rule', COL['rule'], '6 4', 1.8)]


def fig(name):
    return open(os.path.join(SRC, name + '.svg')).read().strip()


def paper_axes(s):
    """The paper's Fig. 9 convention for a figure of curve panels on a log false-alarm axis: the axis runs from the
    largest value on the left to the strictest on the right, with a dotted minor grid. Every panel is mirrored inside
    its frame (curves, vertical grid lines, tick labels, the alpha marker), so the curves themselves are unchanged."""
    frames = [tuple(map(float, f)) for f in re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="none" stroke="currentColor" stroke-width="1"/>', s)]
    num = r'(-?[\d.]+)'
    within = lambda x, L, R: L - 0.5 <= x <= R + 0.5

    def panel_of(x, y):
        return next(((L, T, L + w, T + h) for L, T, w, h in frames if within(x, L, L + w) and T - 1 <= y <= T + h + 20), None)

    def poly(m):
        p = [float(v) for v in re.split(r'[ ,]+', m.group(1).strip())]
        f = panel_of(p[0], p[1])
        if not f:
            return m.group(0)
        L, R = f[0], f[2]
        pts = ' '.join(f'{L + R - x:.1f},{y:.1f}' for x, y in zip(p[0::2], p[1::2]))
        return f'<polyline points="{pts}"'

    def vline(m):
        x, y1, y2 = float(m.group(1)), float(m.group(2)), float(m.group(4))
        f = panel_of(x, y1)
        if not f or float(m.group(3)) != x or y1 < f[1] - 1 or y2 > f[3] + 1:
            return m.group(0)
        X = f[0] + f[2] - x
        return f'<line x1="{X:.2f}" y1="{m.group(2)}" x2="{X:.2f}" y2="{m.group(4)}"'

    def label(m):
        x, y = float(m.group(1)), float(m.group(2))
        f = panel_of(x, y)
        if not f or y < f[3]:
            return m.group(0)
        return f'<text x="{f[0] + f[2] - x:.2f}" y="{m.group(2)}" text-anchor="middle"'

    def alpha(m):
        x, y = float(m.group(1)), float(m.group(2))
        f = panel_of(x, y)
        return m.group(0) if not f else f'<text x="{f[0] + f[2] - (x - 3) + 3:.2f}" y="{m.group(2)}" text-anchor="start" font-size="9.5"'

    ticks = re.findall(rf'<text x="{num}" y="{num}" text-anchor="middle" font-size="10" font-weight="400" fill-opacity="0.8">([\d.]+)</text>', s)
    s = re.sub(r'<polyline points="([^"]+)"', poly, s)
    s = re.sub(rf'<line x1="{num}" y1="{num}" x2="{num}" y2="{num}"', vline, s)
    s = re.sub(rf'<text x="{num}" y="{num}" text-anchor="middle"(?= font-size="10" font-weight="400" fill-opacity="0.8">[\d.]+</text>)', label, s)
    s = re.sub(rf'<text x="{num}" y="{num}" text-anchor="start" font-size="9.5"', alpha, s)
    minor = ''
    for L, T, w, h in frames:
        vals = sorted({float(v) for x, y, v in ticks if within(float(x), L, L + w) and T + h < float(y) < T + h + 20})
        lo, hi = math.log10(vals[0]), math.log10(vals[-1])
        for k in range(int(lo), int(hi)):
            for j in range(2, 10):
                X = L + w - (math.log10(j * 10 ** k) - lo) / (hi - lo) * w
                minor += (f'<line x1="{X:.1f}" y1="{T}" x2="{X:.1f}" y2="{T + h}" stroke="currentColor" stroke-opacity="0.16" '
                          f'stroke-dasharray="1 2"/>')
    if all(float(v) <= 1 for _, _, v in ticks):                     # a PFA axis: the paper's 10^k labels
        sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
        s = re.sub(r'(text-anchor="middle" font-size="10" font-weight="400" fill-opacity="0.8">)(1|0\.1|0\.01|0\.001)(</text>)',
                   lambda m: m.group(1) + '10' + str(round(math.log10(float(m.group(2))))).translate(sup) + m.group(3), s)
    head = re.match(r'<svg[^>]*>(?:<defs>.*?</defs>)?', s).group(0)
    return head + minor + s[len(head):]


def inside(el, x0, y0, w, h):
    """Whether an SVG element's bounding box meets the region (elements without coordinates are kept)."""
    a = dict(re.findall(r'\b(x|y|x1|y1|x2|y2|cx|cy|width|height)="([-\d.]+)"', el))
    a = {k: float(v) for k, v in a.items()}
    if 'points' in el:
        p = [float(v) for v in re.split(r'[ ,]+', re.search(r'points="([^"]+)"', el).group(1).strip())]
        xs, ys = p[0::2], p[1::2]
    elif 'x1' in a:
        xs, ys = [a['x1'], a['x2']], [a['y1'], a['y2']]
    elif 'cx' in a:
        xs, ys = [a['cx']], [a['cy']]
    elif 'x' in a:
        xs, ys = [a['x'], a['x'] + a.get('width', 0)], [a['y'] - 12, a['y'] + a.get('height', 0)]
    else:
        return True
    return max(xs) >= x0 and min(xs) <= x0 + w and max(ys) >= y0 and min(ys) <= y0 + h


def crop(s, x0, y0, w, h, items=None):
    """Part of a published figure: the region (x0, y0, w, h), under a fresh legend when the region has none."""
    inner = re.sub(r'^<svg[^>]*>', '', s)[:-len('</svg>')]
    inner = re.sub(r'<(line|text|polyline|rect|circle)\b[^>]*?(?:/>|>[^<]*</\1>)',
                   lambda m: m.group(0) if inside(m.group(0), x0, y0, w, h) else '', inner)
    label = unescape(re.search(r'aria-label="([^"]*)"', s).group(1))
    lg, lh = legend(items, w) if items else ('', 0)
    body = lg + f'<svg x="0" y="{lh}" width="{w}" height="{h}" viewBox="{x0} {y0} {w} {h}">{inner}</svg>'
    return svg(w, h + lh, body, label)


def html_table(head, rows, cls='wraptab', mid=(), left=(0,)):
    num = lambda i: '' if i in left else ' class="num"'
    h = f'<div class="tablewrap"><table class="{cls}"><tr>' + ''.join(f'<th{num(i)}>{c}</th>' for i, c in enumerate(head)) + '</tr>'
    for k, r in enumerate(rows):
        h += f'<tr{" class=\"mid\"" if k in mid else ""}>' + ''.join(
            f'<td{num(i)}>{minus(c)}</td>' for i, c in enumerate(r)) + '</tr>'
    return h + '</table></div>'


def minus(c):
    return re.sub(r'(^|[\s\[(])-(?=\d)', '\\1−', c)


def plain(c):
    return c.replace('<b>', '').replace('</b>', '')


def bare(c):
    return re.sub(r'\s*<span class="note">.*?</span>', '', c)


def windows(names):
    """Calibrated-threshold results of the bank data sets in names, alpha 0.01 then 0.05."""
    head = ['Data (test events)', 'α', 'DeepQCD', 'gradient boosting', 'logistic', 'WL-CUSUM', 'best rule', 'DeepQCD − GB']
    rows = []
    for a, key in (('0.01', 'final_win01'), ('0.05', 'final_win05')):
        for r in T[key][1:]:
            if r[0] in names:
                rows.append([f'{names[r[0]]} ({r[1]})', a] + [plain(c) for c in r[2:4]] + [bare(c) for c in r[4:6]] + r[6:])
    return html_table(head, rows, cls='wraptab res', mid=(len(rows) // 2,))


def pm_table(key):
    """Polymarket results at one alpha; intervals for the two learners only."""
    t = T[key]
    rows = [r[:2] + [plain(c) for c in r[2:4]] + [bare(c) for c in r[4:6]] + r[6:] for r in t[1:]]
    return html_table(t[0], rows, cls='wraptab res pmres')


def stream_tables(names):
    seq = T['final_stream_equal']
    eq = html_table(seq[0], [[names[r[0]]] + r[1:] for r in seq[1:] if r[0] in names])
    s5, rows, keep = T['final_stream5'], [], False
    for r in s5[1:]:
        if r[0]:
            keep = r[0] in names
        if keep:
            rows.append([names.get(r[0], '')] + r[1:])
    mid = tuple(k for k, r in enumerate(rows) if r[0] and k)
    return eq, html_table(s5[0], rows, mid=mid, left=(0, 1))


def detector_table(key):
    """A summary-page table with detectors as rows; the rules start a new block."""
    t = T[key]
    first_rule = next(k for k, r in enumerate(t[1:]) if r[0].lower().startswith('rule'))
    rows = [[r[0].replace('Logistic (this step only)', 'logistic, this step only').replace('Rules combined', 'rules combined')] + r[1:]
            for r in t[1:]]
    return html_table(t[0], rows, cls='', mid=(first_rule,))


def dots(groups):
    """DeepQCD (filled, coloured by verdict) against the best other detector (open) at one false-alarm level, with the
    paired difference written on the right. groups: [(heading, [(label, deep, other, other_name, diff, verdict)])]."""
    W, rh, left, axr, top = 820, 24, 292, 600, 46
    n = sum(len(rows) + 1 for _, rows in groups)
    H = top + rh * n + 40
    sx = lambda v: left + v * (axr - left)
    b = f'<circle cx="{left}" cy="12" r="5" fill="#00704F"/>' + text(left + 10, 16, 'DeepQCD (colour = verdict)', 11)
    b += f'<circle cx="{left + 190}" cy="12" r="4.5" fill="#FFFFFF" stroke="#4D4D4D" stroke-width="1.6"/>' + text(left + 200, 16, 'best other detector', 11)
    b += text(axr + 16, top - 8, 'against', 10.5, op=0.75) + text(axr + 76, top - 8, 'DeepQCD − other', 10.5, op=0.75)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        X = sx(v)
        b += f'<line x1="{X}" y1="{top - 4}" x2="{X}" y2="{top + rh * n}" stroke="currentColor" stroke-opacity="0.15"/>'
        b += text(X, top + rh * n + 16, f'{v:g}', 10.5, 'middle', op=0.8)
    b += text(sx(0.5), top + rh * n + 34, 'PD at the stated false-alarm level', 11, 'middle', op=0.85)
    k = 0
    for head, rows in groups:
        b += text(8, top + rh * k + 16, head, 11.5, weight=600)
        k += 1
        for lab, d, o, on, diff, vd in rows:
            y = top + rh * k + 12
            b += text(left - 12, y + 4, lab, 11.5, 'end')
            b += f'<line x1="{sx(min(d, o))}" y1="{y}" x2="{sx(max(d, o))}" y2="{y}" stroke="{VCOL[vd]}" stroke-width="2.2"/>'
            b += f'<circle cx="{sx(o)}" cy="{y}" r="4.5" fill="#FFFFFF" stroke="#4D4D4D" stroke-width="1.6"/>'
            b += f'<circle cx="{sx(d)}" cy="{y}" r="5" fill="{VCOL[vd]}"/>'
            b += text(axr + 16, y + 4, on, 11, op=0.85) + text(axr + 76, y + 4, diff, 11, weight=600 if vd != 'tie' else 400)
            k += 1
    return svg(W, H, b, 'DeepQCD against the best other detector in each use case and prediction-market study')


CARDS = {
    'aml': dict(name='AMLworld laundering', source=['IBM AMLworld: HI-Small,', 'LI-Small, HI-Medium'], entity=['one account'],
                clock=['Δ = 1 transfer'], features=['direction, USD, payment format, cross-', 'currency / bank, new counterparty, hub'], P=15,
                event=['first laundering transfer to', 'one transfer after the last'], split=['train / calibrate / test by day', 'HI-Small: –5 / 6–7 / 8–10 Sep'],
                tau1='first laundering transfer', tau2='last + 1'),
    'toxic': dict(name='crypto toxic flow', source=['Binance ETHUSDT perp', 'aggregated trades'], entity=['one market'], clock=['Δ = 1 second'],
                  features=['trades, signed / gross notional, largest', 'trade, levels, returns, imbalance, vol'], P=10,
                  event=['fast run: zigzag swing ≥ 15 bps', 'completed within 60 s'], split=['Jul–Sep train (10,776 runs)', 'Oct 1–20 test (360 runs)'],
                  tau1='the swing starts', tau2='the swing ends'),
    'fraud': dict(name='card fraud', source=['IBM TabFormer (24.4M) and', 'Sparkov (1.05M) transactions'], entity=['one card'], clock=['Δ = 1 transaction'],
                  features=['amount, online, new merchant / place,', 'errors, gap, velocity, hour, distance'], P='15 / 12',
                  event=['burst: frauds at most 20', 'transactions apart'], split=['train / calibrate / test by date', 'TabFormer: –2013 / 14–15 / 16–20'],
                  tau1='first fraud', tau2='last fraud + 1'),
    'squeeze': dict(name='short squeezes', source=['Yahoo daily prices + FINRA', 'short volume, 1,398 stocks'], entity=['one stock'],
                    clock=['Δ = 1 trading day'], features=['return, volume ratio, short share and', 'jump, range, gap, 5/20-day return, vol'], P=10,
                    event=['rally ≥ 50% within 10 days', 'on ≥ 3× normal volume'], split=['before 2025 train (968)', '2025 on test (260)'],
                    tau1='the day after the low', tau2='the day after the peak'),
    'credit': dict(name='credit default', source=['UCI default of credit', 'card clients (30,000)'], entity=['one client'], clock=['Δ = 1 month (6)'],
                   features=['months late, utilization, bill,', 'payment, payment ratio, limit, age'], P=8,
                   event=['defaults in month 7: label 1', 'in every month (Bayesian)'], split=['70% of clients train', '30% test (2,019 defaulters)'],
                   bits=[1, 1, 1, 1, 1, 1], tau1='defaulter: 1 in all six months;\nothers: 0 in all six',
                   strip='one stream per client, April to September; an alarm in any month flags the client'),
    'pm': dict(name='Polymarket order flow', source=['Polymarket trades with', 'wallets: 3,000 markets'], entity=['one market'], clock=['Δ = 1 minute'],
               features=['anonymous: trades, signed $, sizes,', 'one-sidedness, same sizes, gaps'], P='9 (+4)',
               event=['one wallet: metaorder, churn or', 'informed flow; or a price jump'], split=['by closing date: 1,800 train,', '450 calibrate, 750 test'],
               tau1="the wallet's first trade", tau2="the wallet's last trade"),
}

AML = {'AMLworld HI-Small': 'HI-Small', 'AMLworld LI-Small': 'LI-Small', 'AMLworld HI-Medium': 'HI-Medium'}
CARDS_DATA = {'TabFormer cards': 'TabFormer', 'Sparkov cards': 'Sparkov'}

DOTS = [
    ('Five use cases', [
        ('Laundering, HI-Small (PFA 0.01)', 0.55, 0.47, 'GB', '+0.08 [+0.05, +0.10]', 'win'),
        ('Laundering, LI-Small (PFA 0.01)', 0.34, 0.30, 'GB', '+0.04 [+0.02, +0.06]', 'win'),
        ('Laundering, HI-Medium (PFA 0.01)', 0.38, 0.26, 'GB', '+0.13 [+0.12, +0.14]', 'win'),
        ('Crypto toxic flow (PFA 0.05)', 0.46, 0.29, 'GB', '+0.17', 'win'),
        ('Card fraud, TabFormer (PFA 0.01)', 0.69, 0.73, 'GB', '−0.03 [−0.05, −0.02]', 'loss'),
        ('Card fraud, Sparkov (PFA 0.01)', 0.99, 0.99, 'GB', '+0.00 [−0.02, +0.02]', 'tie'),
        ('Short squeezes (PFA 0.05)', 0.76, 0.75, 'CUSUM', '+0.01', 'tie'),
        ('Card default (PFA 0.05)', 0.31, 0.30, 'CUSUM', '+0.01', 'tie')]),
    ('Prediction markets (Polymarket)', [
        ('Metaorders (PFA 0.05)', 0.43, 0.38, 'GB', '+0.05 [+0.03, +0.07]', 'win'),
        ('Churn (PFA 0.01)', 0.44, 0.37, 'GB', '+0.07 [+0.04, +0.11]', 'win'),
        ('Informed flow (PFA 0.05)', 0.17, 0.19, 'GB', '−0.02 [−0.04, +0.00]', 'tie'),
        ('Hour before a price jump (PFA 0.05)', 0.14, 0.15, 'GB', '−0.01 [−0.03, +0.01]', 'tie')]),
]


def main():
    tpl = open(os.path.join(HERE, 'template.html')).read()
    style = re.search(r'<style>.*?</style>', tpl, re.S).group(0)
    det = re.findall(r'<svg viewBox="0 0 820 300".*?</svg>', tpl, re.S)
    assert len(det) == 1
    bank_curves, bank_stream = paper_axes(fig('bank_curves')), paper_axes(fig('bank_stream'))
    narrow = lambda s: s.replace('<svg viewBox', '<svg style="max-width:560px;min-width:470px" viewBox', 1)
    eq_aml, s5_aml = stream_tables(AML)
    eq_card, s5_card = stream_tables(CARDS_DATA)
    data = T['final_data']
    rep = {
        '__STYLE__': style, '__GH__': GH, '__F_DOTS__': dots(DOTS), '__F_PIPE__': pipeline(), '__F_DET__': det[0],
        '__F_PROTOCOL__': fig_protocol(),
        '__T_AML_DATA__': html_table(data[0], [[AML[r[0]]] + r[1:] for r in data[1:] if r[0] in AML]),
        '__T_CARD_DATA__': html_table(data[0], [[CARDS_DATA[r[0]]] + r[1:] for r in data[1:] if r[0] in CARDS_DATA]),
        '__T_AML__': windows(AML), '__T_CARD__': windows(CARDS_DATA),
        '__F_AML_CURVES__': crop(bank_curves, 0, 0, 826, 259),
        '__F_CARD_CURVES__': narrow(crop(bank_curves, 0, 259, 550, 237, LEARNERS)),
        '__F_AML_STREAM__': crop(bank_stream, 0, 0, 826, 269),
        '__F_CARD_STREAM__': narrow(crop(bank_stream, 0, 269, 550, 247, LEARNERS)),
        '__T_AML_SEQ__': eq_aml, '__T_AML_S5__': s5_aml, '__T_CARD_SEQ__': eq_card, '__T_CARD_S5__': s5_card,
        '__F_PATTERNS__': fig('aml_patterns'), '__F_LEARN__': fig('bank_learning'),
        '__T_TOXIC__': detector_table('summary_toxic'), '__F_TOXIC__': paper_axes(fig('toxic_curves')),
        '__T_SQUEEZE__': detector_table('summary_squeeze'), '__F_SQUEEZE__': paper_axes(fig('squeeze_curves')),
        '__T_CREDIT__': detector_table('summary_credit'), '__F_CREDIT__': paper_axes(fig('credit_curves')),
        '__T_PM05__': pm_table('final_pm05'), '__T_PM01__': pm_table('final_pm01'),
        '__T_PMG__': html_table(T['final_pm_sports'][0], T['final_pm_sports'][1:]),
        '__F_PM__': paper_axes(fig('pm_curves')),
        '__T_REPLAY__': html_table(T['final_replay'][0], T['final_replay'][1:], mid=(len(T['final_replay']) - 3,)),
        '__F_REPLAY__': fig('pm_replay'),
    }
    for k, v in CARDS.items():
        rep[f'__C_{k.upper()}__'] = card(v)
    html = open(os.path.join(HERE, 'formal_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    left = re.findall(r'__[A-Z0-9_]+__', html)
    assert not left, left
    open(os.path.join(HERE, 'formal.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

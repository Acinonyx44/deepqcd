"""Assemble report/final.html: the final bank studies (AMLworld HI-Small, LI-Small, HI-Medium; TabFormer; Sparkov) and
the prediction-market studies (Polymarket metaorders, churn, informed flow, jumps, market-making replay) at scale.
Numbers come from runs/bank and runs/pm (bank_final.py, pm_final.py, pm_replay.py); studies not yet run are left out.
Run:  .venv/bin/python report/build_final.py"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_summary import COL, GH, arrow, box, grid, legend, panel, svg, table, text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GB = 'Gradient boosting (lags)'
LR = 'Logistic (this step only)'
BANKS = [('aml-hi-small', 'AMLworld HI-Small'), ('aml-li-small', 'AMLworld LI-Small'), ('aml-hi-medium', 'AMLworld HI-Medium'),
         ('tabformer', 'TabFormer cards'), ('sparkov', 'Sparkov cards')]
TARGETS = [('metaorder', 'Metaorders'), ('churn', 'Churn'), ('informed', 'Informed flow'), ('informed_wallets', 'Informed flow, wallet features'),
           ('jump', 'Hour before a price jump')]
COLS = {'DeepQCD': COL['deep'], GB: '#CC79A7', LR: COL['lr'], 'WL-CUSUM': COL['cusum'], 'rule': COL['rule']}


def J(*p):
    path = os.path.join(REPO, 'runs', *p)
    return json.load(open(path)) if os.path.exists(path) else None


def f2(v):
    return '—' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.2f}'


def ci(v, lo, hi):
    return f'{v:.2f} <span class="note">[{lo:.2f}, {hi:.2f}]</span>'


def best_rule(dets, a='0.05', key='calibrated'):
    rs = [k for k in dets if k.startswith('Rule') or k in ('Rules combined',) or k[0].isupper() and ', last' in k]
    rs = [k for k in rs if k not in ('DeepQCD', GB, LR, 'WL-CUSUM') and not k.startswith('DeepQCD s')]
    return max(rs, key=lambda k: dets[k][key][a]['pd'] if key == 'calibrated' else dets[k]['oracle']['pd_at'][a]) if rs else None


# ------------------------------------------------------------------ figures
def fig_protocol():
    b = text(8, 16, 'One dataset in time', 12, weight=600)
    x0, y0, w = 20, 40, 760
    parts = [(0.0, 0.5, 'training', 'labelled windows', 'fit DeepQCD (5 seeds), GB, logistic, CUSUM'),
             (0.5, 0.7, 'calibration', 'set every threshold', 'Eq. 11 at α, or a target alert rate'),
             (0.7, 1.0, 'test', 'thresholds fixed', 'report realized PFA, PD, alerts')]
    for a, c, name, l1, l2 in parts:
        X, W = x0 + a * w, (c - a) * w
        b += f'<rect x="{X}" y="{y0}" width="{W}" height="30" fill="currentColor" fill-opacity="{0.05 if name != "test" else 0.12}" stroke="currentColor"/>'
        b += text(X + W / 2, y0 + 19, name, 12, 'middle', 600)
        b += text(X + W / 2, y0 + 48, l1, 10.5, 'middle', op=0.85)
        b += text(X + W / 2, y0 + 62, l2, 10.5, 'middle', op=0.85)
    b += text(x0 + 0.5 * w, y0 - 6, 'p₀', 11, 'middle', 600) + text(x0 + 0.7 * w, y0 - 6, 'p₁', 11, 'middle', 600)
    b += text(x0 + w, y0 - 6, 'p₂', 11, 'middle', 600) + text(x0, y0 - 6, 'time →', 10.5, op=0.75)
    y1 = 128
    b += text(8, y1, 'Streaming (every account or card, every transaction of the calibration and test periods)', 12, weight=600)
    y = y1 + 14
    b += box(8, y, 130, 52, 'Transaction n', ['of entity e'])
    b += arrow(138, y + 26, 160, y + 26)
    b += box(162, y, 150, 52, 'Last PRE + 1 steps', ['x[n − PRE] … x[n]'])
    b += arrow(312, y + 26, 334, y + 26)
    b += box(336, y, 120, 52, 'Detector', ['score s[n]'])
    b += arrow(456, y + 26, 478, y + 26)
    b += box(480, y, 100, 52, 's[n] ≥ h ?', ['h from calibration'])
    b += arrow(580, y + 26, 602, y + 26)
    b += box(604, y, 208, 52, 'Alert-day (e, day)', ['false if no flagged span', 'covers the transaction'])
    b += text(8, y + 76, 'Reported: false alert-days per 1,000 active entity-days; share of laundering accounts / fraud bursts alerted while active;', 10.5, op=0.85)
    b += text(8, y + 91, 'share of their money that came at or after the first alert.', 10.5, op=0.85)
    return svg(820, y + 100, b, 'Protocol: training, calibration and test periods in time; streaming scoring of every transaction')


def fig_forest(B, alphas=('0.01', '0.05')):
    names = [(k, n) for k, n in BANKS if k in B]
    dets = [('DeepQCD', 'DeepQCD (5-seed ensemble)'), (GB, 'gradient boosting'), (LR, 'logistic, this step only')]
    W, rh, left = 820, 22, 190
    pw = (W - left - 20) / len(alphas)
    H = 40 + rh * len(names) * len(dets) + 14 * len(names) + 40
    o = ''
    for j, a in enumerate(alphas):
        X0 = left + j * pw
        sx = lambda v, X0=X0: X0 + 10 + v * (pw - 30)
        o += text(X0 + pw / 2, 14, f'calibrated threshold, α = {a}', 11.5, 'middle', 600)
        for v in (0, 0.25, 0.5, 0.75, 1):
            o += f'<line x1="{sx(v)}" y1="24" x2="{sx(v)}" y2="{H - 40}" stroke="currentColor" stroke-opacity="0.12"/>'
            o += text(sx(v), H - 26, f'{v:g}', 10, 'middle', op=0.8)
        o += text(X0 + pw / 2, H - 10, 'PD on test (95% bootstrap interval)', 10.5, 'middle', op=0.85)
        y = 34
        for k, nm in names:
            for d, lab in dets:
                c = B[k]['windows'][d]['calibrated'][a]
                yy = y + 8
                o += f'<line x1="{sx(c["pd_ci"][0])}" y1="{yy}" x2="{sx(c["pd_ci"][1])}" y2="{yy}" stroke="{COLS[d]}" stroke-width="2"/>'
                o += f'<circle cx="{sx(c["pd"])}" cy="{yy}" r="4.2" fill="{COLS[d]}"/>'
                y += rh
            y += 14
    y = 34
    for k, nm in names:
        o += text(left - 10, y + 8 + rh, nm, 11.5, 'end', 600)
        y += rh * len(dets) + 14
    lg, _ = legend([(lab, COLS[d], None, 2.4) for d, lab in dets], 600, y=H - 4 + 18, x0=8)
    return svg(W, H + 24, o + lg, 'PD with bootstrap intervals for DeepQCD, gradient boosting and logistic on each bank dataset')


def fig_window_curves(B):
    pans = []
    for k, nm in BANKS:
        if k not in B:
            continue
        D = B[k]['windows']
        br = best_rule(D)
        cv = [(D[br]['oracle']['pfa'], D[br]['oracle']['pd'], COL['rule'], 1.8, '6 4')] if br else []
        cv += [(D['WL-CUSUM']['oracle']['pfa'], D['WL-CUSUM']['oracle']['pd'], COL['cusum'], 2, None),
               (D[LR]['oracle']['pfa'], D[LR]['oracle']['pd'], COL['lr'], 2, None),
               (D[GB]['oracle']['pfa'], D[GB]['oracle']['pd'], COLS[GB], 2, None),
               (D['DeepQCD']['oracle']['pfa'], D['DeepQCD']['oracle']['pd'], COL['deep'], 2.4, None)]
        pans.append((nm, cv, dict(alpha=0.01)))
    items = [('DeepQCD (ensemble)', COL['deep'], None, 2.4), ('gradient boosting', COLS[GB], None, 2), ('logistic', COL['lr'], None, 2),
             ('WL-CUSUM', COL['cusum'], None, 2), ('best rule', COL['rule'], '6 4', 1.8)]
    return grid(pans, 3, items, pw=272, ph=235, label='PD against PFA on the test streams of each bank dataset')


def fig_stream(B):
    pans = []
    for k, nm in BANKS:
        if k not in B or 'streaming' not in B[k]:
            continue
        S = B[k]['streaming']['detectors']
        rs = [r for r in S if r.startswith('Rule') or r == 'Rules combined']
        br = max(rs, key=lambda r: S[r]['targets']['20']['detected']) if rs else None
        cv = []
        for d, c, wdt, da in ((br, COL['rule'], 1.8, '6 4'), ('WL-CUSUM', COL['cusum'], 2, None), (LR, COL['lr'], 2, None),
                              (GB, COLS[GB], 2, None), ('DeepQCD', COL['deep'], 2.4, None)):
            if d:
                fa = [max(v, 0.1) for v in S[d]['curve']['fa']]
                cv.append((fa, S[d]['curve']['det'], c, wdt, da))
        pans.append((nm, cv, dict(xr=(0.1, 1000), xticks=(0.1, 1, 10, 100, 1000), alpha=None, xlabel='false alert-days per 1,000 active entity-days',
                                  ylabel='share of episodes alerted')))
    items = [('DeepQCD (ensemble)', COL['deep'], None, 2.4), ('gradient boosting', COLS[GB], None, 2), ('logistic', COL['lr'], None, 2),
             ('WL-CUSUM', COL['cusum'], None, 2), ('best rule', COL['rule'], '6 4', 1.8)]
    return grid(pans, 3, items, pw=272, ph=245, label='Streaming: share of episodes alerted against false alert-days per 1,000 active entity-days')


def fig_kinds(B, key='aml-hi-medium'):
    if key not in B:
        key = 'aml-hi-small'
    S = B[key]['streaming']['detectors']
    kinds = sorted(S['DeepQCD']['by_kind'], key=lambda g: -S['DeepQCD']['by_kind'][g][1])
    W, rh, left = 820, 30, 230
    H = 44 + rh * len(kinds) + 40
    sx = lambda v: left + v * (W - left - 30)
    o = ''
    for v in (0, 0.25, 0.5, 0.75, 1):
        o += f'<line x1="{sx(v)}" y1="36" x2="{sx(v)}" y2="{H - 34}" stroke="currentColor" stroke-opacity="0.12"/>'
        o += text(sx(v), H - 20, f'{v:g}', 10, 'middle', op=0.8)
    o += text((left + W) / 2, H - 4, 'share of laundering accounts alerted while active (5 false alert-days per 1,000)', 10.5, 'middle', op=0.85)
    for k, g in enumerate(kinds):
        y = 40 + rh * k
        n = S['DeepQCD']['by_kind'][g][1]
        o += text(left - 8, y + 14, f'{g.lower()} ({n})', 11, 'end')
        for j, (d, c) in enumerate((('DeepQCD', COL['deep']), (GB, COLS[GB]))):
            v = S[d]['by_kind'][g][0]
            o += f'<rect x="{sx(0)}" y="{y + 3 + 11 * j}" width="{sx(v) - sx(0)}" height="9" fill="{c}"/>'
    lg, _ = legend([('DeepQCD (ensemble)', COL['deep'], None, 6), ('gradient boosting', COLS[GB], None, 6)], W, y=16, x0=left)
    return svg(W, H, o + lg, 'Detection by laundering pattern for DeepQCD and gradient boosting')


def fig_learning(C):
    pans = []
    for k, nm in BANKS:
        if k not in C:
            continue
        fr = C[k]['fractions']
        xs = [fr[f]['n_events'] for f in sorted(fr, key=float)]
        cv = []
        for d, c in ((GB, COLS[GB]), ('DeepQCD', COL['deep'])):
            for a, da in (('0.01', None), ('0.05', '6 4')):
                cv.append((xs, [fr[f][d][a]['pd'] for f in sorted(fr, key=float)], c, 2.2, da))
        lo = 10 ** math.floor(math.log10(min(xs)))
        pans.append((nm, cv, dict(xr=(lo, lo * 100), xticks=(lo, lo * 10, lo * 100), alpha=None, xlabel='training windows with an event',
                                  ylabel='PD on test (calibrated)')))
    items = [('DeepQCD, α = 0.01', COL['deep'], None, 2.2), ('DeepQCD, α = 0.05', COL['deep'], '6 4', 2.2),
             ('gradient boosting, α = 0.01', COLS[GB], None, 2.2), ('gradient boosting, α = 0.05', COLS[GB], '6 4', 2.2)]
    return grid(pans, 2, items, pw=405, ph=260, label='Learning curves: PD against the number of training events')


def fig_pm_curves(P):
    pans = []
    for k, nm in TARGETS:
        if k not in P:
            continue
        D = P[k]['detectors']
        rs = [r for r in D if r not in ('DeepQCD', GB, LR, 'WL-CUSUM') and not r.startswith('DeepQCD s')]
        br = max(rs, key=lambda r: D[r]['oracle']['pd_at']['0.05']) if rs else None
        cv = [(D[br]['oracle']['pfa'], D[br]['oracle']['pd'], COL['rule'], 1.8, '6 4')] if br else []
        cv += [(D['WL-CUSUM']['oracle']['pfa'], D['WL-CUSUM']['oracle']['pd'], COL['cusum'], 2, None),
               (D[LR]['oracle']['pfa'], D[LR]['oracle']['pd'], COL['lr'], 2, None),
               (D[GB]['oracle']['pfa'], D[GB]['oracle']['pd'], COLS[GB], 2, None),
               (D['DeepQCD']['oracle']['pfa'], D['DeepQCD']['oracle']['pd'], COL['deep'], 2.4, None)]
        pans.append((f'{nm} ({P[k]["counts"]["test_events"]})', cv, {}))
    items = [('DeepQCD (ensemble)', COL['deep'], None, 2.4), ('gradient boosting', COLS[GB], None, 2), ('logistic', COL['lr'], None, 2),
             ('WL-CUSUM', COL['cusum'], None, 2), ('best rule', COL['rule'], '6 4', 1.8)]
    return grid(pans, 3, items, pw=272, ph=235, label='Polymarket: PD against PFA on the test markets for each target')


def fig_replay(R, key='0.02_300'):
    res = R['results'][key]
    names = list(res)
    W, rh, left = 820, 24, 290
    vals = [res[n]['pnl_per_100usd'] for n in names] + [x for n in names for x in res[n]['ci']]
    lo, hi = math.floor(min(vals) - 1), math.ceil(max(vals) + 1)
    sx = lambda v: left + (v - lo) / (hi - lo) * (W - left - 30)
    H = 16 + rh * len(names) + 46
    o = ''
    step = next(st for st in (1, 2, 5, 10, 20, 50, 100) if (hi - lo) / st <= 10)
    for v in range(math.ceil(lo / step) * step, hi + 1, step):
        o += f'<line x1="{sx(v)}" y1="8" x2="{sx(v)}" y2="{14 + rh * len(names)}" stroke="currentColor" stroke-opacity="{0.5 if v == 0 else 0.1}"/>'
        o += text(sx(v), 30 + rh * len(names), f'{v:+d}' if v else '0', 10, 'middle', op=0.8)
    for k, n in enumerate(names):
        y = 12 + rh * k
        v, (a, b) = res[n]['pnl_per_100usd'], res[n]['ci']
        col = '#00704F' if v >= 0 else '#B2182B'
        o += f'<rect x="{min(sx(0), sx(v))}" y="{y}" width="{abs(sx(v) - sx(0))}" height="{rh - 9}" fill="{col}" fill-opacity="0.8"/>'
        o += f'<line x1="{sx(a)}" y1="{y + (rh - 9) / 2}" x2="{sx(b)}" y2="{y + (rh - 9) / 2}" stroke="currentColor" stroke-width="1.4"/>'
        o += text(left - 8, y + 12, f'{label(n)} ({100 * res[n]["share_filled"]:.0f}%)', 10.5, 'end')
    o += text((left + W) / 2, 46 + rh * len(names), 'P&L, cents per $100 of all traded volume (95% interval over test markets)', 10.5, 'middle', op=0.85)
    return svg(W, H, o, 'Market-making replay: P&L by strategy with bootstrap intervals')


# ------------------------------------------------------------------ tables
def tab_data(B, P, R):
    rows = []
    for k, nm in BANKS:
        if k not in B:
            continue
        c, s = B[k]['counts'], B[k].get('streaming', {})
        ep = c['episodes']
        rows.append([nm, f'{c["steps"]:,}', f'{c["entities"]:,}', f'{ep["train"]:,} / {ep["calibration"]:,} / {ep["test"]:,}',
                     f'{c["cal_events"]:,} / {c["test_events"]:,}', f'{s.get("rows", 0):,}'])
    return table(['Dataset', 'transactions (steps)', 'accounts / cards', 'episodes: train / cal. / test', 'streams: cal. / test',
                  'streaming steps scored'], rows, cls='wraptab')


def tab_windows(B, a):
    rows = []
    for k, nm in BANKS:
        if k not in B:
            continue
        D = B[k]['windows']
        br = best_rule(D, a)
        r = [nm, f'{B[k]["counts"]["test_events"]:,}']
        for d in ('DeepQCD', GB, LR, 'WL-CUSUM'):
            c = D[d]['calibrated'][a]
            s = ci(c['pd'], *c['pd_ci']) + f' <span class="note">({c["pfa"]:.3f})</span>'
            r.append(f'<b>{s}</b>' if d == 'DeepQCD' else s)
        r.append(f2(D[br]['calibrated'][a]['pd']) if br else '—')
        dg = D['DeepQCD']['calibrated'][a]['diff_vs_gb']
        r.append(f'{dg[0]:+.2f} <span class="note">[{dg[1]:+.2f}, {dg[2]:+.2f}]</span>')
        rows.append(r)
    return table(['Dataset', 'test events', 'DeepQCD', 'gradient boosting', 'logistic', 'WL-CUSUM', 'best rule', 'DeepQCD − GB'], rows,
                 cls='wraptab')


def tab_stream(B, R='5'):
    rows = []
    for k, nm in BANKS:
        if k not in B or 'streaming' not in B[k]:
            continue
        S = B[k]['streaming']
        for d, lab in (('DeepQCD', 'DeepQCD'), (GB, 'gradient boosting'), (LR, 'logistic')):
            t = S['detectors'][d]['targets'][R]
            rows.append([nm if d == 'DeepQCD' else '', lab, f'{t["false_alert_days_per_1000"]:.1f}', ci(t['detected'], *t['detected_ci']),
                         f'{t["money_after_alert"]:.2f}', '' if d != 'DeepQCD' else
                         f'{t["diff_vs_gb"][0]:+.2f} <span class="note">[{t["diff_vs_gb"][1]:+.2f}, {t["diff_vs_gb"][2]:+.2f}]</span>'])
    return table(['Dataset', 'detector', 'false alert-days per 1,000 (test)', 'episodes alerted', 'money at or after the alert',
                  'DeepQCD − GB'], rows, cls='wraptab')


def at_rate(curve, R):
    fa, det = curve['fa'], curve['det']
    ok = [d for f, d in zip(fa, det) if f <= R]
    return max(ok) if ok else 0.0


def tab_stream_equal(B, rates=(5, 20)):
    """Detection read off each detector's test curve at the same realized false-alert rate (no calibration drift)."""
    rows = []
    dets = [('DeepQCD', 'DeepQCD'), (GB, 'gradient boosting'), (LR, 'logistic'), ('WL-CUSUM', 'WL-CUSUM')]
    for k, nm in BANKS:
        if k not in B or 'streaming' not in B[k]:
            continue
        S = B[k]['streaming']['detectors']
        r = [nm]
        for R in rates:
            vals = [at_rate(S[d]['curve'], R) for d, _ in dets]
            r += [f'<b>{v:.2f}</b>' if j == 0 else f'{v:.2f}' for j, v in enumerate(vals)]
        rows.append(r)
    head = ['Dataset'] + [f'{lab}, ≤ {R}' for R in rates for _, lab in dets]
    return table(head, rows, cls='wraptab')


def tab_pm(P, a):
    rows = []
    for k, nm in TARGETS:
        if k not in P:
            continue
        D = P[k]['detectors']
        rs = [r for r in D if r not in ('DeepQCD', GB, LR, 'WL-CUSUM') and not r.startswith('DeepQCD s')]
        br = max(rs, key=lambda r: D[r]['calibrated'][a]['pd']) if rs else None
        r = [nm, f'{P[k]["counts"]["test_events"]:,}']
        for d in ('DeepQCD', GB, LR, 'WL-CUSUM'):
            c = D[d]['calibrated'][a]
            s = ci(c['pd'], *c['pd_ci']) + f' <span class="note">({c["pfa"]:.3f})</span>'
            r.append(f'<b>{s}</b>' if d == 'DeepQCD' else s)
        r.append(f'{f2(D[br]["calibrated"][a]["pd"])} <span class="note">({br.lower()})</span>' if br else '—')
        dg = D['DeepQCD']['calibrated'][a]['diff_vs_gb']
        r.append(f'{dg[0]:+.2f} <span class="note">[{dg[1]:+.2f}, {dg[2]:+.2f}]</span>')
        rows.append(r)
    return table(['Target', 'test events', 'DeepQCD', 'gradient boosting', 'logistic', 'WL-CUSUM', 'best rule', 'DeepQCD − GB'], rows,
                 cls='wraptab')


def tab_groups_pm(P, a='0.05'):
    rows = []
    for k, nm in TARGETS:
        if k not in P:
            continue
        g = P[k]['detectors']['DeepQCD']['calibrated'][a]['groups']
        h = P[k]['detectors'][GB]['calibrated'][a]['groups']
        cell = lambda G, x: '—' if x not in G or G[x][0] is None else f'{G[x][0]:.2f} <span class="note">({G[x][1]})</span>'
        rows.append([nm] + [f'{cell(g, x)} / {cell(h, x)}' for x in ('sports', 'other')])
    return table(['Target', 'sports markets: DeepQCD / GB', 'other markets: DeepQCD / GB'], rows, cls='wraptab')


def label(n):
    return n.replace('informed_wallets alarm', 'informed alarm, wallet features')


def tab_replay(R):
    rows = []
    a, b = R['results']['0.02_300'], R['results']['0.02_1800']
    for n in a:
        d = a[n]['vs_always_ci']
        diff = '—' if n == 'always quote' else f'{a[n]["pnl_per_100usd"] - a["always quote"]["pnl_per_100usd"]:+.2f} <span class="note">[{d[0]:+.2f}, {d[1]:+.2f}]</span>'
        rows.append([label(n), f'{100 * a[n]["share_filled"]:.0f}%', ci(a[n]['pnl_per_100usd'], *a[n]['ci']), diff,
                     ci(b[n]['pnl_per_100usd'], *b[n]['ci'])])
    return table(['Strategy', 'volume filled', 'P&amp;L, 5-minute markout', 'against always quoting', 'P&amp;L, 30-minute markout'], rows,
                 cls='wraptab')


def main():
    B = {k: J('bank', f'{k}.json') for k, _ in BANKS}
    B = {k: v for k, v in B.items() if v}
    C = {k: J('bank', f'{k}_curve.json') for k, _ in BANKS}
    C = {k: v for k, v in C.items() if v}
    P = {k: J('pm', f'{k}.json') for k, _ in TARGETS}
    P = {k: v for k, v in P.items() if v}
    R = J('pm', 'replay.json')
    tpl = open(os.path.join(HERE, 'template.html')).read()
    style = re.search(r'<style>.*?</style>', tpl, re.S).group(0)
    rep = {'__STYLE__': style, '__GH__': GH, '__F_PROTOCOL__': fig_protocol(), '__T_DATA__': tab_data(B, P, R)}
    if B:
        rep.update({'__F_FOREST__': fig_forest(B), '__F_WCURVES__': fig_window_curves(B), '__T_W01__': tab_windows(B, '0.01'),
                    '__T_W05__': tab_windows(B, '0.05')})
        if any('streaming' in v for v in B.values()):
            rep.update({'__F_STREAM__': fig_stream(B), '__T_S5__': tab_stream(B, '5'), '__T_S20__': tab_stream(B, '20'),
                        '__T_SEQ__': tab_stream_equal(B), '__F_KINDS__': fig_kinds(B)})
    if C:
        rep['__F_LEARN__'] = fig_learning(C)
    if P:
        rep.update({'__F_PM__': fig_pm_curves(P), '__T_PM01__': tab_pm(P, '0.01'), '__T_PM05__': tab_pm(P, '0.05'),
                    '__T_PMG__': tab_groups_pm(P)})
    if R:
        rep.update({'__F_REPLAY__': fig_replay(R), '__T_REPLAY__': tab_replay(R)})
    from final_text import TEXT                         # report/final_text.py: the prose, keyed by placeholder
    rep.update({f'__{k}__': v for k, v in TEXT.items()})
    html = open(os.path.join(HERE, 'final_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    left = re.findall(r'__[A-Z0-9_]+__', html)
    if left:
        print('missing:', left)
    open(os.path.join(HERE, 'final.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

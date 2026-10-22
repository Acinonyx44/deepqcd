"""Assemble report/usecases.html (six finance use cases) from runs/usecases/*.json and report/usecases_text.json.
Run:  .venv/bin/python report/build_usecases.py"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    U = {k: json.load(open(os.path.join(REPO, 'runs', 'usecases', f'{k}.json')))
         for k in ('toxic', 'amlworld', 'mm_pnl', 'squeeze', 'credit', 'fraud')}
    for r in U.values():
        r.pop('runs_per_day', None)

    def finite(o):                                   # thresholds can be inf (no alarm at that level): store as null
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [finite(v) for v in o]
        return None if isinstance(o, float) and not math.isfinite(o) else o
    U = finite(U)
    U['credit']['n_test_events'] = U['credit']['n_test_defaulters']
    U['credit']['n_test_no'] = U['credit']['n_test_others']
    for k, r in U.items():                           # keep curves only for what the chart draws
        D = r.get('detectors')
        if not D:
            continue
        rules = [n for n in D if n.startswith('Rule')]
        best = max(rules, key=lambda n: D[n]['pd_at']['0.05'])
        for n, d in D.items():
            if n not in ('DeepQCD', 'Logistic (this step only)', 'WL-CUSUM', best):
                d.pop('pfa', None)
                d.pop('pd', None)
    text = json.load(open(os.path.join(HERE, 'usecases_text.json')))
    style = re.search(r'<style>.*?</style>', open(os.path.join(HERE, 'template.html')).read(), re.S).group(0)
    prose = '\n'.join(f"$('{k}').innerHTML = {json.dumps(v)};" for k, v in text.items())
    rep = {'__STYLE__': style, '__REPO__': 'https://github.com/Acinonyx44/deepqcd', '__BRANCH__': 'claude/awesome-allen-8e9wvl',
           '__TXE__': str(U['toxic']['n_test_events']), '__AWE__': f"{U['amlworld']['n_test_events']:,}",
           '__SQS__': f"{U['squeeze']['n_stocks']:,}", '__SQE__': str(U['squeeze']['n_test_events']),
           '__CRT__': f"{U['credit']['n_test_defaulters'] + U['credit']['n_test_others']:,}",
           '__FRE__': f"{U['fraud']['n_test_events']:,}", '__HELPERS__': open(os.path.join(HERE, 'helpers.js')).read(),
           '__PROSE__': prose}
    html = open(os.path.join(HERE, 'usecases_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    html = html.replace('__DATA__', json.dumps(U, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    left = re.findall(r'__[A-Z]+__', re.sub(r'<script id="data".*?</script>', '', html, flags=re.S))
    assert not left, left
    open(os.path.join(HERE, 'usecases.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

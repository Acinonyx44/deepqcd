"""Assemble report/works.html (AMLSim laundering detection; Polymarket metaorders and churn) from runs/aml/results.json,
runs/polymarket_meta/minute/results.json and runs/polymarket_meta/churn_minute/results.json.
Run:  .venv/bin/python report/build_works.py"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load(*p):
    r = json.load(open(os.path.join(REPO, 'runs', *p)))
    for k in ('history', 'examples'):
        r.pop(k, None)
    return r


def main():
    A = load('aml', 'results.json')
    me = load('polymarket_meta', 'minute', 'results.json')
    ch = load('polymarket_meta', 'churn_minute', 'results.json')
    ms = json.load(open(os.path.join(REPO, 'data', 'polymarket', 'full_markets.json')))
    n_tr = sum(len(np.load(f)['t']) for f in glob.glob(os.path.join(REPO, 'data', 'polymarket', 'full', '*.npz')))
    style = re.search(r'<style>.*?</style>', open(os.path.join(HERE, 'template.html')).read(), re.S).group(0)
    rep = {'__STYLE__': style, '__REPO__': 'https://github.com/Acinonyx44/deepqcd', '__BRANCH__': 'claude/awesome-allen-8e9wvl',
           '__AEV__': f"{A['n_test_events']:,}", '__AMAIN__': str(A['n_main']), '__AMEM__': f"{A['n_member']:,}",
           '__PMK__': str(len(ms)), '__PTR__': f'{n_tr / 1e6:.1f}', '__HELPERS__': open(os.path.join(HERE, 'helpers.js')).read()}
    html = open(os.path.join(HERE, 'works_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    html = html.replace('__DATA__', json.dumps({'aml': A, 'pm': {'metaorder': me, 'churn': ch}}, separators=(',', ':'),
                                               allow_nan=False).replace('</', '<\\/'))
    left = re.findall(r'__[A-Z]+__', re.sub(r'<script id="data".*?</script>', '', html, flags=re.S))
    assert not left, left
    open(os.path.join(HERE, 'works.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

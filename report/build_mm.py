"""Assemble report/mm.html (market-maker stress detection) from runs/mm/results.json and runs/mm_kappa0.25/results.json.
Run:  .venv/bin/python report/build_mm.py"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    R = json.load(open(os.path.join(REPO, 'runs', 'mm', 'results.json')))
    R2 = json.load(open(os.path.join(REPO, 'runs', 'mm_kappa0.25', 'results.json')))
    style = re.search(r'<style>.*?</style>', open(os.path.join(HERE, 'template.html')).read(), re.S).group(0)
    texts = json.load(open(os.path.join(HERE, 'mm_text.json')))
    import sys
    sys.path.insert(0, REPO)
    import mmsim
    s, cal = R['settings'], R['calibration']
    rep = {
        '__STYLE__': style, '__CLOSE__': f"{cal['close']:,.2f}", '__DATE__': cal['date'], '__SIGMA__': f"{cal['sigma_daily']:.2%}",
        '__NDAYS__': str(s['train_days'] + s['test_days']), '__LAMBDA__': str(mmsim.LAMBDA), '__TAU1__': str(s['tau1']),
        '__K__': str(s['K']), '__NS__': str(s['n_streams']), '__WARM__': str(s['warm']), '__WLK__': str(s['wl_k']),
        '__FW__': str(s['flow_w']), '__RESULT__': texts['result'], '__LIMITS__': texts['limits'],
        '__HELPERS__': open(os.path.join(HERE, 'helpers.js')).read(),
    }
    html = open(os.path.join(HERE, 'mm_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    ALL = {'runs': {'1.0': R, '0.25': R2}, 'edgeread': texts['edge']}
    html = html.replace('__DATA__', json.dumps(ALL, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    open(os.path.join(HERE, 'mm.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

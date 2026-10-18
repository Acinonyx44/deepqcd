"""Assemble report/followup.html (rare changes, and Polymarket) from runs/rare/results.json and runs/polymarket/results.json.
Run:  .venv/bin/python report/build_followup.py"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    R = json.load(open(os.path.join(REPO, 'runs', 'rare', 'results.json')))
    M = json.load(open(os.path.join(REPO, 'runs', 'polymarket', 'results.json')))
    M.pop('history', None)
    style = re.search(r'<style>.*?</style>', open(os.path.join(HERE, 'template.html')).read(), re.S).group(0)
    a, b = R['nets']['A']['platt']
    for n in ('A', 'B'):
        R['nets'][n].pop('history', None)
    rep = {
        '__STYLE__': style, '__REPO__': 'https://github.com/Acinonyx44/deepqcd', '__BRANCH__': 'claude/awesome-allen-8e9wvl',
        '__PA__': f'{a:.2f}', '__PB__': f'{b:.2f}', '__NMK__': f'{M["n_markets"]:,}', '__NEV__': str(M['n_events']['all_events']),
        '__NTE__': str(M['n_events']['test_events']), '__NNO__': f'{M["n_test_no"]:,}', '__SPLIT__': M['split'], '__TDET__': '3',
        '__HELPERS__': open(os.path.join(HERE, 'helpers.js')).read(),
    }
    html = open(os.path.join(HERE, 'followup_template.html')).read()
    for k, v in rep.items():
        html = html.replace(k, v)
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    ALL = {'rare': R, 'pm': M}
    html = html.replace('__DATA__', json.dumps(ALL, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    assert '__' not in re.sub(r'<script id="data".*?</script>', '', html, flags=re.S).replace('__proto__', ''), \
        re.findall(r'__[A-Z]+__', html)
    open(os.path.join(HERE, 'followup.html'), 'w').write(html)
    print('size KB', len(html) // 1024)


if __name__ == '__main__':
    main()

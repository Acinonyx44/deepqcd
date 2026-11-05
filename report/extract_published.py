"""One-off: pull the figures and result tables the formal page needs out of the two published pages
(DeepQCD Study Summary, DeepQCD Final Studies), whose run outputs were lost with the container.
Usage: python report/extract_published.py <folder with summary.html and final.html as published> report/formal_src"""
import json
import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
S = open(os.path.join(src, 'summary.html')).read()
F = open(os.path.join(src, 'final.html')).read()
svg = lambda s: re.findall(r'<svg viewBox="[^"]+".*?</svg>', s, flags=re.S)
SS, FS = svg(S), svg(F)
assert len(SS) == 30 and len(FS) == 8, (len(SS), len(FS))
figs = {'toxic_curves': SS[19], 'credit_curves': SS[27], 'squeeze_curves': SS[29], 'amlsim_curves': SS[21],
        'bank_curves': FS[2], 'bank_stream': FS[3], 'aml_patterns': FS[4], 'bank_learning': FS[5], 'pm_curves': FS[6],
        'pm_replay': FS[7]}
for k, v in figs.items():
    assert 'aria-label' in v
    open(os.path.join(out, k + '.svg'), 'w').write(v + '\n')
    print(k, len(v) // 1024, 'KB', re.search(r'aria-label="([^"]{0,60})', v).group(1))


def tables(s):
    res = []
    for t in re.findall(r'<table[^>]*>(.*?)</table>', s, flags=re.S):
        rows = []
        for r in re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.S):
            rows.append([c.strip() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.S)])
        res.append(rows)
    return res


ST, FT = tables(S), tables(F)
print(len(ST), len(FT))
names_f = ['summary', 'data', 'win01', 'win05', 'stream_equal', 'stream5', 'pm05', 'pm01', 'pm_sports', 'replay']
names_s = {'toxic': 'rule: price move, 5 s', 'amlsim': 'rule: dormant account reactivated', 'credit': 'rule: months late now',
           'squeeze': 'rule: volume spike', 'gb': 'DeepQCD (seed range)', 'conditions': 'Condition', 'features': 'Features x[n]'}
assert len(FT) == len(names_f)
T = {'final_' + n: t for n, t in zip(names_f, FT)}
for n, key in names_s.items():
    hit = [t for t in ST if any(key in c for r in t for c in r) and (n in ('gb', 'conditions', 'features') or t[0][0] == 'Detector')]
    assert len(hit) == 1, (n, len(hit))
    T['summary_' + n] = hit[0]
json.dump(T, open(os.path.join(out, 'tables.json'), 'w'), indent=1, ensure_ascii=False)
for k, t in T.items():
    print(k, len(t), 'rows;', ' | '.join(t[0])[:110])

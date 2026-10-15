"""Assemble report/index.html (the interactive report) from the run outputs in runs/paper_sweep and runs/transient_real.
authors_outputs.json holds the result arrays saved in the authors' notebooks (drawn as open circles).
Run:  .venv/bin/python report/build_report.py"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

NOTES = {
    'hai': 'HAI 21.03: a hardware-in-the-loop industrial control testbed sampled at 1 Hz; attacks are split at random between training and test.',
    'fog': 'Daphnet: ankle accelerometer of Parkinson\'s patients at 32 Hz; freezing episodes; subjects split 6 / 4, so the test subjects are unseen. The GitHub copy is a .mat of the UCI Daphnet data.',
    'smd': 'SMD: server metrics at one reading per minute; machines split 14 / 14, so the test machines are unseen; only events lasting at least K count.',
    'occupancy': 'UCI Occupancy: one office, one reading per minute; training file datatraining.txt, test files datatest.txt and datatest2.txt (the split of the authors\' own Occupancy script).',
    'occupancy-nolight': 'UCI Occupancy without the light sensor: the same office, but presence must be inferred from CO₂, temperature and humidity.',
    'skab': 'SKAB: water-circulation testbed at 1 Hz; each 20-minute run has one valve or pump fault that is switched on and off.',
}
LABEL = {'hai': 'HAI 21.03 (ICS attacks)', 'fog': 'Daphnet (freezing of gait)', 'smd': 'SMD (server machines)',
         'occupancy': 'Occupancy (all five sensors)', 'occupancy-nolight': 'Occupancy (no light sensor)', 'skab': 'SKAB (water pump)'}
UNIT = {'hai': 's', 'fog': '1/32 s', 'smd': 'min', 'occupancy': 'min', 'occupancy-nolight': 'min', 'skab': 's'}
ORDER = ['hai', 'smd', 'skab', 'fog', 'occupancy', 'occupancy-nolight']
GH = 'https://github.com/'
SOURCES = {  # where each dataset is downloaded from (fetch_data.sh)
    'hai': GH + 'icsdataset/hai/tree/master/hai-21.03', 'smd': GH + 'NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset',
    'skab': GH + 'waico/SKAB', 'fog': GH + 'takotab/FOG', 'occupancy': GH + 'LuisM78/Occupancy-detection-data',
    'occupancy-nolight': GH + 'LuisM78/Occupancy-detection-data', 'fishkiller': GH + 'OctoberChang/klcpd_code',
    'yahoo': GH + 'OctoberChang/klcpd_code', 'beedance': GH + 'OctoberChang/klcpd_code', 'hasc': GH + 'OctoberChang/klcpd_code',
    'nab': GH + 'numenta/NAB', 'tcpd': GH + 'alan-turing-institute/TCPD', 'tep': GH + 'camaramm/tennessee-eastman-profBraatz',
    'cmapss': GH + 'edwardzjl/CMAPSSData', 'pmubage': GH + 'NanpengYu/pmuBAGE', 'sp500': GH + 'fja05680/dow-sp500-100-years',
    'seismic': GH + 'AI4EPS/PhaseNet', 'pumpdump': GH + 'SystemsLab-Sapienza/pump-and-dump-dataset',
    'keystroke': GH + 'bikramb98/Keystroke-dynamics', 'iot-mirai': GH + 'ymirsky/KitNET-py',
}


def thin(c, n=160):
    pfa, pd = c['pfa'], c['pd']
    if len(pfa) > n:
        idx = sorted(set([0, len(pfa) - 1] + [round(i * (len(pfa) - 1) / (n - 1)) for i in range(n)]))
        pfa, pd = [pfa[i] for i in idx], [pd[i] for i in idx]
    return {'pfa': [float(f'{v:.4g}') for v in pfa], 'pd': [round(v, 4) for v in pd], 'pd_at': c['pd_at']}


def thin_evals(ev):
    out = {}
    for key, cons in ev.items():
        out[key] = {}
        for con, r in cons.items():
            out[key][con] = {'n': r['n'], 'n_events': r['n_events'], 'DeepQCD': thin(r['DeepQCD']), 'WL-CUSUM': thin(r['WL-CUSUM'])}
    return out


def load(p):
    return json.load(open(p))


def main():
    D = {'authors': load(os.path.join(HERE, 'authors_outputs.json')), 'iid': {}, 'ar': {}, 'trans': {}, 'real': {}, 'sources': SOURCES}
    for f in glob.glob(os.path.join(REPO, 'runs', 'paper_sweep', '*.json')):
        r = load(f)
        r.pop('history', None)
        if r['kind'] == 'iid':
            D['iid'][f"p{r['p']}_mu{r['mu']}"] = r
        elif r['kind'] == 'ar':
            D['ar'][f"mu{r['mu1']}_lam{r['lam1']}"] = r
        else:
            for k, v in r['evals'].items():
                r['evals'][k] = {d: thin(c) for d, c in v.items()}
            D['trans'][f"{r['mu']}"] = r
    D['survey'] = load(os.path.join(REPO, 'runs', 'transient_real', 'survey.json')) if os.path.exists(os.path.join(REPO, 'runs', 'transient_real', 'survey.json')) else []
    for s in D['survey']:
        if s['dataset'] in LABEL:
            s['label'] = LABEL[s['dataset']]
        if s['kind'] == 'interval':  # C3: enough test events with the paper's 1024 normal steps before them
            s['C3'] = s['native_1000'] >= 5
    sp = os.path.join(REPO, 'runs', 'transient_real', 'splice.json')
    D['splice'] = load(sp) if os.path.exists(sp) else None
    for name in ORDER:
        runs = sorted(glob.glob(os.path.join(REPO, 'runs', 'transient_real', f'{name}.all.s*.json')))
        if not runs:
            continue
        rs = [load(p) for p in runs]
        zf = os.path.join(REPO, 'runs', 'transient_real', f'{name}.z.s0.json')
        z = load(zf) if os.path.exists(zf) else None
        D['real'][name] = {
            'label': LABEL[name], 'unit': UNIT[name], 'P': rs[0]['P_raw'], 'mu_post': rs[0]['mu_post'], 'notes': NOTES[name],
            'n_train_events': rs[0]['n_train_events'], 'n_test_events': rs[0]['n_test_events'],
            'all': [thin_evals(r['evals']) for r in rs], 'z': thin_evals(z['evals']) if z else None,
            'hist': [r['history'] for r in rs], 'hist_z': z['history'] if z else None,
            'examples': rs[0]['examples']}
    missing = [k for k in ('p7_mu1.0',) if k not in D['iid']]
    data = json.dumps(D, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    html = open(os.path.join(HERE, 'template.html')).read()
    for key, fn in (('__DISCUSSION__', 'discussion.html'),):
        p = os.path.join(HERE, fn)
        html = html.replace(key, open(p).read().strip() if os.path.exists(p) else '(pending)')
    html = html.replace('__HELPERS__', open(os.path.join(HERE, 'helpers.js')).read())
    html = html.replace('__DATA__', data)
    open(os.path.join(HERE, 'index.html'), 'w').write(html)
    print('real:', list(D['real']), '| iid:', sorted(D['iid']), '| ar:', sorted(D['ar']), '| trans:', sorted(D['trans']),
          '| size KB', len(html) // 1024, '| missing', missing)


if __name__ == '__main__':
    main()

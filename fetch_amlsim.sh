#!/usr/bin/env bash
# Build IBM AMLSim (https://github.com/IBM/AMLSim) and simulate the banks used by aml_qcd.py.
#
# Output: data/amlsim/qcd10K_s<seed>/{tx_log.csv, alert_members.csv} for seeds 1..6. Each is a bank of about 10,000
# accounts over 720 days (paramFiles/10K: accounts, degrees and normal models unchanged) with 300 laundering
# schemes across the eight AMLSim typologies, amounts 100-1,000 (the same range as normal transfers), 5-20 days each.
#
# Three things are needed to run the 2021 code today; nothing in the simulation logic is changed:
#   - MASON 20 is no longer downloadable as a jar; AMLSim uses only sim.engine.SimState and Steppable, which are
#     compiled from MASON's GitHub source.
#   - networkx 1.11 (required by AMLSim) imports fractions.gcd, removed in Python 3.9: patched to math.gcd.
#   - transaction_graph_generator.py calls random.sample on sets (an error since Python 3.11): a shim sorts them first;
#     and it crashes on an alert member without edges: such a member gets amounts 0 in alert_members.csv.
#
# Run:  bash fetch_amlsim.sh        (about 5 minutes; needs java, mvn, python3.11)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
D=$ROOT/data/amlsim
mkdir -p "$D"
cd "$D"

[ -d AMLSim ] || { git clone -q https://github.com/IBM/AMLSim.git && git -C AMLSim checkout -q 7338a4b; }
[ -d mason ] || git clone -q --depth 1 https://github.com/eclab/mason.git
if [ ! -f AMLSim/jars/mason.20.jar ]; then
  S=mason/mason/src/main/java
  rm -rf mbuild && mkdir mbuild
  javac -nowarn -d mbuild -sourcepath $S $S/sim/engine/SimState.java $S/sim/engine/Steppable.java
  (cd mbuild && jar cf ../AMLSim/jars/mason.20.jar .)
fi
cd AMLSim
if [ ! -f target/amlsim-1.0.0.jar ]; then
  mvn -q -B install:install-file -Dfile=jars/mason.20.jar -DgroupId=mason -DartifactId=mason -Dversion=20 -Dpackaging=jar -DgeneratePom=true
  mvn -q -B -DskipTests package
fi

if [ ! -d "$D/venv" ]; then
  python3.11 -m venv "$D/venv"
  "$D/venv/bin/pip" install -q numpy "networkx==1.11" "decorator<5" powerlaw python-dateutil faker
  grep -rl "from fractions import gcd" "$D"/venv/lib/python3.11/site-packages/networkx | xargs sed -i 's/from fractions import gcd/from math import gcd/'
fi
"$D/venv/bin/python" - <<'EOF'
p = 'scripts/transaction_graph_generator.py'
s = open(p).read()
if '_compat_sample' not in s:
    s = s.replace('import cProfile\n', '''import cProfile

# Python 3.11 compatibility (added for this run): random.sample no longer accepts sets or dict views
_random_sample = random.sample
def _compat_sample(population, k, *args, **kwargs):
    if not hasattr(population, '__getitem__'):
        population = sorted(population)
    return _random_sample(population, k, *args, **kwargs)
random.sample = _compat_sample
''', 1)
old = '''                    min_amt = '{:.2f}'.format(min(get_out_edge_attrs(sub_g, n, "amount")))
                    max_amt = '{:.2f}'.format(max(get_out_edge_attrs(sub_g, n, "amount")))'''
new = '''                    amts = get_out_edge_attrs(sub_g, n, "amount") or [0.0]  # a member can have no edge (fix for this run)
                    min_amt = '{:.2f}'.format(min(amts))
                    max_amt = '{:.2f}'.format(max(amts))'''
s = s.replace(old, new)
open(p, 'w').write(s)
EOF

rm -rf paramFiles/qcd10K && cp -r paramFiles/10K paramFiles/qcd10K
cat > paramFiles/qcd10K/alertPatterns.csv <<'EOF'
count,type,schedule_id,min_accounts,max_accounts,min_amount,max_amount,min_period,max_period,bank_id,is_sar
40,fan_in,2,5,10,100.0,1000.0,5,20,bank,True
40,fan_out,2,5,10,100.0,1000.0,5,20,bank,True
40,cycle,2,5,10,100.0,1000.0,5,20,bank,True
30,bipartite,2,5,10,100.0,1000.0,5,20,bank,True
30,stack,2,6,10,100.0,1000.0,5,20,bank,True
40,random,2,5,10,100.0,1000.0,5,20,bank,True
40,scatter_gather,2,5,10,100.0,1000.0,5,20,bank,True
40,gather_scatter,2,5,10,100.0,1000.0,5,20,bank,True
EOF

CP="target/amlsim-1.0.0.jar:target/classes:jars/mason.20.jar:$(find ~/.m2/repository/org/json ~/.m2/repository/it/unimi -name '*.jar' | tr '\n' ':')$(find ~/.m2/repository -name 'slf4j-api*.jar' | head -1)"
for seed in 1 2 3 4 5 6; do
  name=qcd10K_s$seed
  [ -f "$D/$name/tx_log.csv" ] && continue
  "$D/venv/bin/python" - "$seed" "$name" <<'EOF'
import json, sys
c = json.load(open('paramFiles/10K/conf.json'))
d = json.load(open('conf.json'))
for k, v in d.items():                      # keys the 10K conf lacks (normal_models, sar_interval, ...)
    if k not in c:
        c[k] = v
    elif isinstance(v, dict):
        for kk, vv in v.items():
            c[k].setdefault(kk, vv)
c['input']['directory'] = 'paramFiles/qcd10K'
c['input']['normal_models'] = 'normalModels.csv'
c['general']['random_seed'] = int(sys.argv[1])
c['general']['simulation_name'] = sys.argv[2]
json.dump(c, open(f'paramFiles/qcd10K/conf_{sys.argv[2]}.json', 'w'), indent=2)
EOF
  rm -rf tmp/$name outputs/$name
  "$D/venv/bin/python" scripts/transaction_graph_generator.py paramFiles/qcd10K/conf_$name.json > "$D/$name.gen.log" 2>&1
  java -Xms1g -Xmx6g -cp "$CP" amlsim.AMLSim paramFiles/qcd10K/conf_$name.json > "$D/$name.sim.log" 2>&1
  mkdir -p "$D/$name"
  cp outputs/$name/tx_log.csv tmp/$name/alert_members.csv "$D/$name/"
  echo "$name: $(($(wc -l < "$D/$name/tx_log.csv") - 1)) transactions, $(($(wc -l < "$D/$name/alert_members.csv") - 1)) alert members"
done

#!/usr/bin/env bash
# Download every dataset deepqcd_real.py uses into data/ (git-ignored). All of them are on GitHub, so this
# also works from a Claude Code cloud session. About 1.1 GB on disk.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data
RAW=https://raw.githubusercontent.com

get() { [ -s "$2" ] || curl -sSfL --retry 3 -o "$2" "$1"; }
clone() { [ -d "$2" ] || git clone -q --depth 1 "$1" "$2"; }

echo skab;      clone https://github.com/waico/SKAB.git SKAB
echo nab;       clone https://github.com/numenta/NAB.git NAB
echo tcpd;      clone https://github.com/alan-turing-institute/TCPD.git TCPD
echo kl-cpd;    clone https://github.com/OctoberChang/klcpd_code.git klcpd_code   # bee dance, HASC, fish kill, Yahoo
echo smd
if [ ! -d OmniAnomaly ]; then
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git
    git -C OmniAnomaly sparse-checkout set ServerMachineDataset
fi

echo tep
mkdir -p tep
for f in d00 d00_te $(for i in $(seq -w 1 21); do echo d$i d${i}_te; done); do
    get $RAW/camaramm/tennessee-eastman-profBraatz/master/$f.dat tep/$f.dat
done

echo occupancy
mkdir -p occupancy
for f in datatraining.txt datatest.txt datatest2.txt; do
    get $RAW/LuisM78/Occupancy-detection-data/master/$f occupancy/$f
done

echo hai
mkdir -p hai
for f in train1 test1 test2 test3 test4 test5; do
    get $RAW/icsdataset/hai/master/hai-21.03/$f.csv.gz hai/$f.csv.gz
done

echo cmapss
mkdir -p cmapss
get $RAW/edwardzjl/CMAPSSData/master/train_FD001.txt cmapss/train_FD001.txt

echo pmubage
mkdir -p pmubage
for i in $(seq 0 20); do get $RAW/NanpengYu/pmuBAGE/main/data/frequency/frequency_$i.npy pmubage/frequency_$i.npy; done
for i in 0 1 2 3 4; do get $RAW/NanpengYu/pmuBAGE/main/data/voltage/voltage_$i.npy pmubage/voltage_$i.npy; done

echo sp500
mkdir -p finance
get $RAW/fja05680/dow-sp500-100-years/master/SP500.csv finance/SP500.csv

echo seismic
mkdir -p phasenet/npz
get $RAW/AI4EPS/PhaseNet/master/dataset/waveform.csv phasenet/waveform.csv
tail -n +2 phasenet/waveform.csv | cut -d, -f1 | while read f; do
    get $RAW/AI4EPS/PhaseNet/master/dataset/waveform_train/$f phasenet/npz/$f
done

echo fog
mkdir -p daphnet
get "$RAW/takotab/FOG/master/Code%20SVM/daphnet_19062017.mat" daphnet/daphnet.mat

echo pumpdump
mkdir -p pumpdump
get $RAW/SystemsLab-Sapienza/pump-and-dump-dataset/master/labeled_features/features_5S.csv.gz pumpdump/features_5S.csv.gz

echo keystroke
mkdir -p keystroke
get $RAW/bikramb98/Keystroke-dynamics/master/DSL-StrongPasswordData.csv keystroke/DSL-StrongPasswordData.csv

echo iot-mirai
mkdir -p kitnet
if [ ! -s kitnet/mirai3.npy ]; then
    get $RAW/ymirsky/KitNET-py/master/dataset.zip kitnet/dataset.zip
    (cd kitnet && unzip -o -q dataset.zip mirai3.csv)
    ../.venv/bin/python -c "import numpy as np; np.save('kitnet/mirai3.npy', np.loadtxt('kitnet/mirai3.csv', delimiter=',', dtype=np.float32))"
fi
echo done

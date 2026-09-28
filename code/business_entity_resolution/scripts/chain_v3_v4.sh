#!/usr/bin/env bash
# v3 = word features (retrain A/B) -> export; v4 = + cross-encoder stage C -> export.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
L=~/ber/logs
FLAG=~/ber/logs/own_train_c.flag; rm -f $FLAG
# guard: the older chain would start stage C on the previous model after the cross-encoder; stop it
( while [ ! -f $FLAG ]; do pkill -f "ber.run --stage train_c" 2>/dev/null; sleep 20; done ) &
until grep -q -E "test: word features added|Traceback" $L/wordfeat.log 2>/dev/null && ! pgrep -f "stage wordfeat" >/dev/null; do sleep 30; done
grep -q Traceback $L/wordfeat.log && { echo "WORDFEAT FAILED"; exit 1; }
bash ber.sh -m ber.run --stage train --split train > $L/train_v3.log 2>&1 || { echo "TRAIN V3 FAILED"; exit 1; }
bash ber.sh -m ber.run --stage predict --split test > $L/predict_v3.log 2>&1 || { echo "PREDICT V3 FAILED"; exit 1; }
bash ber.sh -m ber.run --stage export --split test > $L/export_v3.log 2>&1 || { echo "EXPORT V3 FAILED"; exit 1; }
cp ~/ber/output/matching_results.tsv /mnt/c/Users/arnav/Projects/AmazonHackathon/output/matching_results_v3.tsv
mkdir -p ~/ber/artifacts/v3 && cp ~/ber/artifacts/summary.json ~/ber/artifacts/model_*.txt ~/ber/artifacts/v3/
echo "V3 READY $(date)"; grep -E "chosen" $L/train_v3.log | tail -1; grep -E "PASS|FAIL" $L/export_v3.log
# v4: wait for the cross-encoder's test scores, then stage C
until [ -f ~/ber/artifacts/test/ce.parquet ] && ! pgrep -f "stage cross" >/dev/null; do sleep 30; done
touch $FLAG; sleep 30
bash ber.sh -m ber.run --stage train_c --split train > $L/train_c.log 2>&1 || { echo "TRAIN C FAILED"; exit 1; }
bash ber.sh -m ber.run --stage predict_c --split test > $L/predict_c.log 2>&1 || { echo "PREDICT C FAILED"; exit 1; }
bash ber.sh -m ber.run --stage export --split test > $L/export_v4.log 2>&1 || { echo "EXPORT V4 FAILED"; exit 1; }
cp ~/ber/output/matching_results.tsv /mnt/c/Users/arnav/Projects/AmazonHackathon/output/matching_results_v4.tsv
echo "V4 READY $(date)"; grep -E "stage C holdout" $L/train_c.log; grep -E "PASS|FAIL" $L/export_v4.log

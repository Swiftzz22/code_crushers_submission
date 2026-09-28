#!/usr/bin/env bash
# v5 = France geography normalization + wider cross-encoder band + stage C on 50% of train.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
L=~/ber/logs; A=~/ber/artifacts
run() { local name=$1; shift; bash ber.sh -m ber.run "$@" > $L/v5_$name.log 2>&1 || { echo "FAILED at $name: $(grep -m1 -E 'Error' $L/v5_$name.log)"; exit 1; }; }
# back up v4 (the 0.982 leaderboard version)
mkdir -p $A/v4 && cp $A/summary.json $A/model_C_fold*.txt $A/v4/ && cp $A/test/pred.parquet $A/v4/test_pred.parquet \
  && cp $A/test/ce.parquet $A/v4/test_ce.parquet && cp $A/train/ce.parquet $A/v4/train_ce.parquet && echo "v4 backed up"
run geo --stage geo --split test
echo "geo map:"; grep -E "^\S.*INFO   \(" $L/v5_geo.log | sed "s/^.*INFO //"
# GPU: widen the cross-encoder band on train, while the CPU rebuilds the French test features
( run cross_ext_train --stage cross_ext --split train; echo "cross_ext train done" ) &
GPU=$!
run normalize --stage normalize --split test
run features --stage features --split test
run wordfeat --stage wordfeat --split test
run predict --stage predict --split test
wait $GPU
run cross_ext_test --stage cross_ext --split test
run train_c --stage train_c --split train --set model.train_s1_frac=0.5
grep -E "stage C holdout" $L/v5_train_c.log | sed "s/^.*INFO //"
run predict_c --stage predict_c --split test
run export --stage export --split test
cp ~/ber/output/matching_results.tsv /mnt/c/Users/arnav/Projects/AmazonHackathon/output/matching_results_v5.tsv
echo "V5 READY $(date)"
grep -A7 monitors $L/v5_export.log | grep -E "France|India|US"; grep -E "PASS|FAIL" $L/v5_export.log

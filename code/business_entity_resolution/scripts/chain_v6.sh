#!/usr/bin/env bash
# v6 = v5 + self-trained (France-adapted) cross-encoder scores for the unlabelled partition.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
L=~/ber/logs
run() { local name=$1; shift; bash ber.sh -m ber.run "$@" > $L/v6_$name.log 2>&1 || { echo "FAILED at $name: $(grep -m1 -E 'Error' $L/v6_$name.log)"; exit 1; }; }
run adapt --stage adapt --split test
grep -E "pseudo|guard|rescored" $L/v6_adapt.log | sed "s/^.*INFO //"
run predict_c --stage predict_c --split test
run export --stage export --split test
cp ~/ber/output/matching_results.tsv /mnt/c/Users/arnav/Projects/AmazonHackathon/output/matching_results_v6.tsv
echo "V6 READY $(date)"
grep -A7 monitors $L/v6_export.log | grep -E "France|India|US"; grep -E "PASS|FAIL" $L/v6_export.log

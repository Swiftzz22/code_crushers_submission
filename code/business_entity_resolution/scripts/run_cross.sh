#!/usr/bin/env bash
# Train + score the cross-encoders with up to 3 attempts (finished halves are skipped on retry).
cd "$(dirname "$0")/.."
for i in 1 2 3; do
  HF_HUB_OFFLINE=1 bash ber.sh -m ber.run --stage cross --split train > ~/ber/logs/cross.log 2>&1 && break
  echo "attempt $i failed: $(grep -m1 -E 'Error' ~/ber/logs/cross.log)"; sleep 20
done
grep -E "band|AUC|saved|test band" ~/ber/logs/cross.log

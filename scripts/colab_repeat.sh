#!/usr/bin/env bash
# Repeat measurements for the polarbear checks on Colab: base / polarbear / polarbear --route
# on freshly generated sets (seeds 2 and 3) plus the units set. One run per cell; prints compare tables.
# Usage (Colab, after cloning the repo; export HF_TOKEN in a Python cell first to keep reports):
#   bash scripts/colab_repeat.sh
cd "$(dirname "$0")/.."
ev() {  # name data-dir extra-flags (a finished report is kept; an interrupted run continues from its .partial.jsonl)
  [ -f "reports/applied/$1.json" ] && { echo "$1 already done"; return; }
  python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B --template-kwargs '{"enable_thinking": false}' \
    --data "$2/eval_test.jsonl" --name "$1" $3 --batch-size 16 --max-new-tokens 192 --gpu-hour-price 1.0 \
    --out "reports/applied/$1.json" --resume >> "/tmp/$1.log" 2>&1
  if [ -f "reports/applied/$1.json" ]; then echo "$1 done $(date +%H:%M)"; else echo "$1 FAILED $(date +%H:%M)"; tail -n 15 "/tmp/$1.log"; fi
}
trio() {  # prefix data-dir
  ev "$1-base" "$2" ""
  ev "$1-pb" "$2" "--pipeline polarbear"
  ev "$1-route" "$2" "--pipeline polarbear --route"
  python -m genesis.harness.evaluate compare "reports/applied/$1-base.json" "reports/applied/$1-pb.json" "reports/applied/$1-route.json" 2>&1 | tail -12
}
save() {
  # Colab secrets cannot be read from a subprocess: the notebook exports HF_TOKEN first
  # (os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")); without it nothing is uploaded.
  [ -n "$HF_TOKEN" ] || { echo "HF_TOKEN not set: reports stay in this session only"; return; }
  python - <<'PY' || echo "NOT saved to Hugging Face"
import os
from genesis import hubsync
token = os.environ["HF_TOKEN"]
hubsync.sync_up(f"{hubsync.username(token)}/genesis-fin-v1-runs", "reports", token, path_in_repo="reports-polarbear")
print("reports saved")
PY
}
python -m genesis_calc.build_tasks --set units --out data/applied/calc_units_v1 --per-kind 60 > /dev/null
trio units data/applied/calc_units_v1
save
for seed in 2 3; do
  for set in multi lookup basic; do
    dir="data/applied/calc_${set}_s${seed}"
    python -m genesis_calc.build_tasks --set "$set" --out "$dir" --per-kind 60 --seed "$seed" > /dev/null
    trio "rep-${set}-s${seed}" "$dir"
  done
  save
done
echo ALL DONE

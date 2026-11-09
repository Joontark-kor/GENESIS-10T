#!/usr/bin/env bash
# Overnight run: seeds 4-5 (multi/lookup/basic) then the full Korean GSM8K test set (1319). Repeat measurements for the polarbear checks on Colab: base / polarbear / polarbear --route
# on freshly generated sets (seeds 2 and 3) plus the units set. One run per cell; prints compare tables.
# Usage (Colab, after cloning the repo; export HF_TOKEN in a Python cell first to keep reports):
#   bash scripts/colab_repeat.sh
cd "$(dirname "$0")/.."
ev() {  # name data-dir extra-flags
  python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B --template-kwargs '{"enable_thinking": false}' \
    --data "$2/eval_test.jsonl" --name "$1" $3 --batch-size 16 --max-new-tokens ${EV_TOKENS:-192} --gpu-hour-price 1.0 \
    --out "reports/applied/$1.json" > "/tmp/$1.log" 2>&1
  if [ -f "reports/applied/$1.json" ]; then echo "$1 done $(date +%H:%M)"; else echo "$1 FAILED $(date +%H:%M)"; tail -n 15 "/tmp/$1.log"; fi
}
trio() {  # prefix data-dir
  ev "$1-base" "$2" ""
  ev "$1-pb" "$2" "--pipeline polarbear"
  ev "$1-route" "$2" "--pipeline polarbear --route"
  python -m genesis.harness.evaluate compare "reports/applied/$1-base.json" "reports/applied/$1-pb.json" "reports/applied/$1-route.json" 2>&1 | tail -5
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
for seed in 4 5; do
  for set in multi lookup basic; do
    dir="data/applied/calc_${set}_s${seed}"
    python -m genesis_calc.build_tasks --set "$set" --out "$dir" --per-kind 60 --seed "$seed" > /dev/null
    trio "rep-${set}-s${seed}" "$dir"
  done
  save
done
python -m genesis_calc.external --kind gsm8k-ko --out data/external/gsm8k-full --limit 1319 || { echo "build failed"; echo ALL DONE; exit; }
EV_TOKENS=384
trio ext-gsm8k-full data/external/gsm8k-full
save
echo ALL DONE

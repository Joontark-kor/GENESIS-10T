#!/usr/bin/env bash
# Public benchmarks on Colab (rows are downloaded at run time, never committed): KMMLU Accounting/Taxation/Economics
# (test split, up to 300 per subject) and a Korean GSM8K (first 300 test items), base / polarbear / polarbear --route.
# Usage (Colab, after cloning the repo): bash scripts/colab_external.sh
cd "$(dirname "$0")/.."
ev() {  # name data-dir extra-flags
  python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B --template-kwargs '{"enable_thinking": false}' \
    --data "$2/eval_test.jsonl" --name "$1" $3 --batch-size 16 --max-new-tokens 384 --gpu-hour-price 1.0 \
    --out "reports/applied/$1.json" > "/tmp/$1.log" 2>&1
  echo "$1 done $(date +%H:%M)"
}
trio() {
  ev "$1-base" "$2" ""
  ev "$1-pb" "$2" "--pipeline polarbear"
  ev "$1-route" "$2" "--pipeline polarbear --route"
  python -m genesis.harness.evaluate compare "reports/applied/$1-base.json" "reports/applied/$1-pb.json" "reports/applied/$1-route.json" 2>&1 | tail -5
}
save() {
  python - <<'PY' || echo "NOT saved to Hugging Face"
from google.colab import userdata
from genesis import hubsync
token = userdata.get("HF_TOKEN")
hubsync.sync_up(f"{hubsync.username(token)}/genesis-fin-v1-runs", "reports", token, path_in_repo="reports-polarbear")
print("reports saved")
PY
}
for kind in kmmlu gsm8k-ko; do
  python -m genesis_calc.external --kind "$kind" --out "data/external/$kind" --limit 300 || { echo "build failed: $kind"; continue; }
  trio "ext-$kind" "data/external/$kind"
  save
done
echo ALL DONE

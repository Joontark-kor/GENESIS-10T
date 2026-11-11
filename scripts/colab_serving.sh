#!/usr/bin/env bash
# Served evaluation = local evaluation (Colab T4). Same questions, same polarbear pipeline, once through the
# local model and once through genesis.serving.mini_server over HTTP. Prints both accuracies and how many
# items got a different verdict. One question at a time on the server, so keep the set to 240 items.
# Usage (after cloning the repo in Colab):  bash scripts/colab_serving.sh        (SEED=4 by default)
cd "$(dirname "$0")/.."
SEED=${SEED:-4}
DIR="data/applied/calc_multi_s${SEED}"
TK='{"enable_thinking": false}'
python -m genesis_calc.build_tasks --set multi --out "$DIR" --per-kind 60 --seed "$SEED" > /dev/null

echo "local run: $(date +%H:%M)"
python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B --template-kwargs "$TK" \
  --data "$DIR/eval_test.jsonl" --name srv-local-pb --pipeline polarbear --batch-size 16 --max-new-tokens 192 \
  --gpu-hour-price 1.0 --out reports/applied/srv-local-pb.json > /tmp/srv-local.log 2>&1 \
  || { echo "local run FAILED"; tail -n 15 /tmp/srv-local.log; exit 1; }
echo "local done $(date +%H:%M)"

python -m genesis.serving.mini_server --model Qwen/Qwen3-4B --port 8000 > /tmp/srv-server.log 2>&1 &
SERVER=$!
for _ in $(seq 1 120); do
  curl -s http://127.0.0.1:8000/v1/models > /dev/null && break
  sleep 5
done
curl -s http://127.0.0.1:8000/v1/models > /dev/null || { echo "server did not start"; tail -n 15 /tmp/srv-server.log; kill $SERVER; exit 1; }
echo "server up $(date +%H:%M)"

python -m genesis.harness.evaluate run --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-4B --template-kwargs "$TK" \
  --data "$DIR/eval_test.jsonl" --name srv-endpoint-pb --pipeline polarbear --max-new-tokens 192 \
  --gpu-hour-price 1.0 --out reports/applied/srv-endpoint-pb.json > /tmp/srv-endpoint.log 2>&1 \
  || { echo "endpoint run FAILED"; tail -n 15 /tmp/srv-endpoint.log; kill $SERVER; exit 1; }
kill $SERVER
echo "endpoint done $(date +%H:%M)"

python -m genesis.harness.evaluate compare reports/applied/srv-local-pb.json reports/applied/srv-endpoint-pb.json 2>&1 | tail -4
python - <<'PY'
import json
a = json.load(open("reports/applied/srv-local-pb.json"))["rows"]
b = json.load(open("reports/applied/srv-endpoint-pb.json"))["rows"]
ca, cb = [bool(x["correct"]) for x in a], [bool(x["correct"]) for x in b]
same_text = sum(x["output"] == y["output"] for x, y in zip(a, b))
print(f"items {len(a)}: local {sum(ca)} correct, served {sum(cb)} correct, "
      f"different verdict {sum(x != y for x, y in zip(ca, cb))}, identical output text {same_text}")
PY
echo ALL DONE

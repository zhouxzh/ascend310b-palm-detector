#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhong/Documents/ascend310b-palm-detector
RUN_ID=20261123-palm-reject-three-backbones
OUT="$ROOT/artifacts/$RUN_ID"
PY=/home/zhong/anaconda3/envs/torch/bin/python
TB=/home/zhong/anaconda3/envs/torch/bin/tensorboard

if ss -ltn | grep -q ':6006 '; then
  echo 'port 6006 is already in use' >&2
  exit 4
fi
mkdir -p "$OUT/logs" "$OUT/tensorboard"

nohup bash "$ROOT/scripts/run_palm_reject_classifier_matrix.sh" \
  > "$OUT/logs/matrix-launcher.log" 2>&1 < /dev/null &
runner_pid=$!
echo "$runner_pid" > "$OUT/runner.pid"

nohup "$TB" --logdir "$OUT/tensorboard" --host 0.0.0.0 --port 6006 --reload_interval 30 \
  > "$OUT/logs/tensorboard.log" 2>&1 < /dev/null &
tensorboard_pid=$!
echo "$tensorboard_pid" > "$OUT/tensorboard.pid"

nohup "$PY" "$ROOT/scripts/snapshot_palm_reject_classifier_results.py" \
  > "$OUT/logs/snapshot.log" 2>&1 < /dev/null &
snapshot_pid=$!
echo "$snapshot_pid" > "$OUT/snapshot.pid"

nohup /usr/bin/python3 -m http.server 6007 --bind 0.0.0.0 --directory "$OUT" \
  > "$OUT/logs/results-http.log" 2>&1 < /dev/null &
results_pid=$!
echo "$results_pid" > "$OUT/results-http.pid"

sleep 3
printf '{"runner_pid":%d,"tensorboard_pid":%d,"snapshot_pid":%d,"results_pid":%d,"started_at":"%s"}\n' \
  "$runner_pid" "$tensorboard_pid" "$snapshot_pid" "$results_pid" "$(date -u +%FT%TZ)" \
  > "$OUT/supervisor.json"
cat "$OUT/supervisor.json"

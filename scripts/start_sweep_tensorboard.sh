#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PORT="${1:-6006}"
readonly LOG_DIR="$ROOT/artifacts/sweeps/palm-v001/tensorboard"
readonly RUNS_DIR="$ROOT/artifacts/runs"

mkdir -p "$LOG_DIR"
pkill -f "[t]ensorboard.*--port ${PORT}" || true
for _ in {1..30}; do
  if ! ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    break
  fi
  sleep 1
done
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "TensorBoard port ${PORT} did not become available" >&2
  exit 1
fi
setsid "$HOME/anaconda3/envs/palm-nanodet/bin/tensorboard" \
  --logdir_spec "sweep:$RUNS_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$LOG_DIR/server.log" 2>&1 < /dev/null &
echo $! >"$LOG_DIR/server.pid"
for _ in {1..10}; do
  if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo "TensorBoard: http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
cat "$LOG_DIR/server.log" >&2 || true
exit 1

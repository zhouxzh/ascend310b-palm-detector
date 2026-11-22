#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PORT="${1:-6007}"
readonly LOG_DIR="$ROOT/artifacts/tensorboard"
readonly NANODET_RUNS_DIR="$ROOT/artifacts/runs"
readonly PICODET_SURFACE3_RUNS_DIR="$ROOT/artifacts/picodet-surface3-v003/runs"
readonly PICODET_SURFACE3_M_RUNS_DIR="$ROOT/artifacts/picodet-surface3-m-v003/runs"
readonly PICODET_SURFACE3_320_RUNS_DIR="$ROOT/artifacts/picodet-surface3-320-v004/runs"
readonly PICODET_SURFACE3_320_STRETCH_RUNS_DIR="$ROOT/artifacts/picodet-surface3-320-stretch-v005/runs"

mkdir -p "$LOG_DIR" "$NANODET_RUNS_DIR" "$PICODET_SURFACE3_RUNS_DIR" "$PICODET_SURFACE3_M_RUNS_DIR" "$PICODET_SURFACE3_320_RUNS_DIR" "$PICODET_SURFACE3_320_STRETCH_RUNS_DIR"
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "TensorBoard port ${PORT} is already in use" >&2
  exit 1
fi
# TensorBoard scans only retained final event files plus any active run event.
# tensorboardX files are compatible with the pinned TensorBoard server.
setsid "$HOME/anaconda3/envs/palm-nanodet/bin/tensorboard" \
  --logdir_spec "nanodet:$NANODET_RUNS_DIR,picodet-surface3:$PICODET_SURFACE3_RUNS_DIR,picodet-surface3-m:$PICODET_SURFACE3_M_RUNS_DIR,picodet-surface3-320:$PICODET_SURFACE3_320_RUNS_DIR,picodet-surface3-320-stretch:$PICODET_SURFACE3_320_STRETCH_RUNS_DIR" \
  --reload_interval 5 --host 127.0.0.1 --port "$PORT" \
  >"$LOG_DIR/server.log" 2>&1 < /dev/null &
echo $! >"$LOG_DIR/server.pid"
for _ in {1..15}; do
  if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo "TensorBoard: http://127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 1
done
cat "$LOG_DIR/server.log" >&2 || true
exit 1

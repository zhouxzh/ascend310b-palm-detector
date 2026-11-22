#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOG_ROOT="$ROOT/artifacts/single-palm-v001"
readonly PORT="${1:-6008}"

exec conda run --no-capture-output -n palm-nanodet tensorboard \
  --logdir_spec "nanodet-single:$LOG_ROOT/nanodet/runs,picodet-single:$LOG_ROOT/picodet/runs" \
  --reload_interval 15 --host 127.0.0.1 --port "$PORT"

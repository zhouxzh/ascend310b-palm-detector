#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 RUN_ID PRETRAINED_CHECKPOINT PRETRAINED_MANIFEST [SEED]" >&2
  exit 2
fi
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_ID="$1"
readonly PRETRAINED_CHECKPOINT="$2"
readonly PRETRAINED_MANIFEST="$3"
readonly SEED="${4:-42}"
readonly RUN_DIR="$ROOT/artifacts/runs/$RUN_ID"
readonly DATA_ROOT="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-dedup-v001"
readonly DATA_REPORT="$ROOT/reports/$RUN_ID-data-audit.json"
readonly CONFIG="$ROOT/configs/nanodet-plus-m-320-portable-hagridv2-palm.yml"

cd "$ROOT"
[[ -d third_party/nanodet/.git ]] || { echo "NanoDet is missing. Run ./scripts/fetch_nanodet.sh first." >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 2; }
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/check_coco_dataset.py \
  --split train "$DATA_ROOT/annotations/instances_train.json" "$DATA_ROOT/images" \
  --split val "$DATA_ROOT/annotations/instances_val.json" "$DATA_ROOT/images" \
  --split test "$DATA_ROOT/annotations/instances_test.json" "$DATA_ROOT/images" \
  --category-contract single-palm --allow-missing-subject-id --verify-image-hashes --output "$DATA_REPORT"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
  --base-config "$CONFIG" --run-id "$RUN_ID" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --pretrained-manifest "$PRETRAINED_MANIFEST" --data-report "$DATA_REPORT" --seed "$SEED"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python third_party/nanodet/tools/train.py "$RUN_DIR/config.yml" --seed "$SEED" |& tee "$RUN_DIR/train.log"

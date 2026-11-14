#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 TRAIN_ANN VAL_ANN IMAGES_ROOT PRETRAINED_CHECKPOINT PRETRAINED_MANIFEST DATA_AUDIT OUTPUT_DIR LABEL_QUALITY" >&2
  exit 2
fi
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/nanodet_pipeline_smoke.py \
  --base-config configs/nanodet-plus-m-320-portable-hagridv2-handed.yml \
  --train-annotations "$1" --val-annotations "$2" --images-root "$3" \
  --pretrained-checkpoint "$4" --pretrained-manifest "$5" --data-audit "$6" \
  --output-dir "$7" --label-quality "$8"

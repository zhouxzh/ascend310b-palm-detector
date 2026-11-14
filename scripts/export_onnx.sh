#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_DIR="$(cd "$1" && pwd)"
readonly CHECKPOINT="$RUN_DIR/model_best/model_best.ckpt"
readonly ONNX_PATH="$RUN_DIR/nanodet-plus-m-320-palm.onnx"
readonly ONNX_REPORT="$RUN_DIR/onnx-check.json"
[[ -f "$RUN_DIR/config.yml" && -f "$CHECKPOINT" ]] || { echo "Missing run config or best checkpoint" >&2; exit 2; }
cd "$ROOT"
NUM_CLASSES="$(conda run --no-capture-output -n palm-nanodet python -c "import yaml; print(yaml.safe_load(open('$RUN_DIR/config.yml', encoding='utf-8'))['model']['arch']['head']['num_classes'])")"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/export_nanodet_onnx.py --config "$RUN_DIR/config.yml" \
  --checkpoint "$CHECKPOINT" --output "$ONNX_PATH" |& tee "$RUN_DIR/export-onnx.log"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/verify_onnx.py --onnx "$ONNX_PATH" --num-classes "$NUM_CLASSES" --output "$ONNX_REPORT"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/generate_model_card.py --run-dir "$RUN_DIR" --onnx "$ONNX_PATH" \
  --onnx-report "$ONNX_REPORT" --output "$RUN_DIR/model-card.md"

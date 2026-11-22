#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_DIR="$(cd "$1" && pwd)"
readonly CHECKPOINT="$RUN_DIR/model_best/model_best.ckpt"
readonly ONNX_REPORT="$RUN_DIR/onnx-check.json"
[[ -f "$RUN_DIR/config.yml" && -f "$CHECKPOINT" ]] || { echo "Missing run config or best checkpoint" >&2; exit 2; }
cd "$ROOT"
MODEL_DETAILS="$(conda run --no-capture-output -n palm-nanodet python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); a=c["model"]["arch"]; print(a["name"], *c["data"]["val"]["input_size"], a["backbone"].get("model_size", "none"), a["head"]["num_classes"])' "$RUN_DIR/config.yml")"
read -r ARCH INPUT_WIDTH INPUT_HEIGHT MODEL_SIZE NUM_CLASSES <<< "$MODEL_DETAILS"
readonly ONNX_PATH="$RUN_DIR/model-${INPUT_WIDTH}x${INPUT_HEIGHT}.onnx"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/export_nanodet_onnx.py --config "$RUN_DIR/config.yml" \
  --checkpoint "$CHECKPOINT" --output "$ONNX_PATH" |& tee "$RUN_DIR/export-onnx.log"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/verify_onnx.py --onnx "$ONNX_PATH" --config "$RUN_DIR/config.yml" --output "$ONNX_REPORT"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/generate_model_card.py --run-dir "$RUN_DIR" --onnx "$ONNX_PATH" \
  --onnx-report "$ONNX_REPORT" --output "$RUN_DIR/model-card.md"

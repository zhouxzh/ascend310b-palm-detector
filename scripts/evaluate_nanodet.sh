#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$2" != "val" && "$2" != "test" ) ]]; then
  echo "Usage: $0 RUN_DIR {val|test}" >&2
  exit 2
fi
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_DIR="$(cd "$1" && pwd)"
readonly TASK="$2"
readonly CHECKPOINT="$RUN_DIR/model_best/model_best.ckpt"
readonly EVALUATION_DIR="$RUN_DIR/evaluations/$TASK"
readonly EVALUATION_CONFIG="$EVALUATION_DIR/config.yml"
[[ -f "$RUN_DIR/config.yml" && -f "$CHECKPOINT" ]] || { echo "Missing run config or best checkpoint" >&2; exit 2; }
cd "$ROOT"
test_annotations=()
if [[ "$TASK" == "test" && -n "${NANODET_TEST_ANNOTATIONS:-}" ]]; then
  test_annotations=(--test-annotations "$NANODET_TEST_ANNOTATIONS")
fi
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/create_evaluation_config.py --run-config "$RUN_DIR/config.yml" --task "$TASK" --output "$EVALUATION_CONFIG" "${test_annotations[@]}"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/test_nanodet_model.py --task "$TASK" --config "$EVALUATION_CONFIG" --model "$CHECKPOINT" |& tee "$EVALUATION_DIR/evaluate-$TASK.log"

if [[ "$TASK" == "test" ]]; then
  TEST_ANNOTATIONS="$(conda run --no-capture-output -n palm-nanodet python -c "import yaml; print(yaml.safe_load(open('$EVALUATION_CONFIG', encoding='utf-8'))['data']['val']['ann_path'])")"
  prediction_file="$(find "$EVALUATION_DIR" -mindepth 2 -maxdepth 2 -type f -name results.json -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "$prediction_file" ]]; then
    echo "NanoDet test did not write results.json under $EVALUATION_DIR" >&2
    exit 2
  fi
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
    python scripts/evaluate_coco_metrics.py --ground-truth "$TEST_ANNOTATIONS" \
    --predictions "$prediction_file" --output "$EVALUATION_DIR/coco-metrics.json"
fi

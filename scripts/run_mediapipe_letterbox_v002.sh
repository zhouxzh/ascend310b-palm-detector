#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_ID="nanodet-plus-m-320-mediapipe-letterbox-v002__data-v001__seed-42__20261117-1500"
readonly NANODET_RUN_DIR="$ROOT/artifacts/runs/$RUN_ID"
readonly NANO_AUDIT="$ROOT/reports/$RUN_ID-data-audit.json"
readonly PICODET_ROOT="$ROOT/artifacts/picodet-v002"
readonly PICODET_INVENTORY="$PICODET_ROOT/variant-inventory.json"
readonly PICODET_STATE="$PICODET_ROOT/sweep-state.json"
readonly PICODET_RUNS="$PICODET_ROOT/runs"
readonly PICODET_RUN_ID="picodet-v002-picodet-s-320-seed-42"
readonly PICODET_RUN_DIR="$PICODET_RUNS/$PICODET_RUN_ID"
readonly HANDED_DATASET="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-handed-v001"
readonly HANDED_TEST_ANNOTATIONS="$HANDED_DATASET/annotations/instances_test.json"

cd "$ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/check_mediapipe_letterbox_v002_readiness.py
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  python scripts/check_picodet_training_readiness.py --matrix configs/picodet-mediapipe-letterbox-v002.yml

if [[ ! -d "$NANODET_RUN_DIR" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/check_coco_dataset.py \
    --split train data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_train.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
    --split dev artifacts/sweeps/palm-v001/validation-audited/instances_dev.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
    --split gate artifacts/sweeps/palm-v001/validation-audited/instances_gate.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
    --split test data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_test.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
    --verify-image-hashes --output "$NANO_AUDIT"
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
    --base-config configs/nanodet-plus-m-320-mediapipe-letterbox-v002.yml --run-id "$RUN_ID" \
    --pretrained-checkpoint artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt \
    --pretrained-manifest data/manifests/nanodet-plus-m_416_checkpoint.json \
    --data-report "$NANO_AUDIT" --seed 42
fi

if [[ ! -f "$NANODET_RUN_DIR/training-summary.json" ]]; then
  resume=()
  if [[ -f "$NANODET_RUN_DIR/model_last.ckpt" ]]; then
    resume+=(--resume)
  fi
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/train_nanodet_sweep.py "$NANODET_RUN_DIR/config.yml" --seed 42 "${resume[@]}" |& tee -a "$NANODET_RUN_DIR/train.log"
fi

run_picodet_training() {
  local attempt
  for attempt in 1 2; do
    if conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
      --inventory "$PICODET_INVENTORY" --state "$PICODET_STATE" --runs-root "$PICODET_RUNS" --run-prefix picodet-v002 --seed 42; then
      return 0
    fi
    if [[ -f "$PICODET_RUN_DIR/training-summary.json" ]]; then
      return 0
    fi
    if (( attempt == 2 )); then
      return 1
    fi
    printf 'PicoDet v002 training exited before reaching a terminal summary; resuming its newest complete checkpoint.\n' >&2
  done
}

run_picodet_training

if [[ ! -f "$PICODET_RUN_DIR/picodet-onnx-check.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/export_picodet_onnx.py \
    --run-dir "$PICODET_RUN_DIR" --upstream third_party/PaddleDetection --samples 20
fi
if [[ ! -f "$PICODET_RUN_DIR/test-eval/coco-metrics.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/evaluate_picodet.py \
    --run-dir "$PICODET_RUN_DIR" --annotations "$HANDED_TEST_ANNOTATIONS" --output-dir "$PICODET_RUN_DIR/test-eval" --upstream third_party/PaddleDetection
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/evaluate_coco_metrics.py \
    --ground-truth "$HANDED_TEST_ANNOTATIONS" --predictions "$PICODET_RUN_DIR/test-eval/bbox.json" --output "$PICODET_RUN_DIR/test-eval/coco-metrics.json"
fi
if [[ ! -f "$PICODET_RUN_DIR/picodet-onnx-decoder-check.json" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/compare_picodet_onnx.py \
    --run-dir "$PICODET_RUN_DIR" --annotations "$HANDED_TEST_ANNOTATIONS" --images-root "$HANDED_DATASET/images" \
    --output "$PICODET_RUN_DIR/picodet-onnx-decoder-check.json"
fi

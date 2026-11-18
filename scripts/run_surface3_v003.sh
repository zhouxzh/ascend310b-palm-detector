#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_ID="nanodet-plus-m15-416-surface3-letterbox-v003__data-v001__seed-42__20261118-1500"
readonly NANODET_RUN_DIR="$ROOT/artifacts/runs/$RUN_ID"
readonly NANODET_ONNX="$NANODET_RUN_DIR/model-416x416.onnx"
readonly NANO_AUDIT="$ROOT/reports/surface3-v001-data-audit.json"
readonly PICODET_ROOT="$ROOT/artifacts/picodet-surface3-v003"
readonly PICODET_INVENTORY="$PICODET_ROOT/variant-inventory.json"
readonly PICODET_STATE="$PICODET_ROOT/sweep-state.json"
readonly PICODET_RUNS="$PICODET_ROOT/runs"
readonly PICODET_RUN_ID="picodet-surface3-v003-picodet-l-416-surface3-seed-42"

cd "$ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/check_surface3_training_readiness.py
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  python scripts/check_picodet_training_readiness.py --matrix configs/picodet-surface3-letterbox-v003.yml

if [[ ! -d "$NANODET_RUN_DIR" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
    --base-config configs/nanodet-plus-m15-416-surface3-letterbox-v003.yml --run-id "$RUN_ID" \
    --pretrained-checkpoint artifacts/pretrained/nanodet-plus-m15-416.ckpt \
    --pretrained-manifest data/manifests/nanodet-plus-m15-416_checkpoint.json \
    --data-report "$NANO_AUDIT" --seed 42
fi
if [[ ! -f "$NANODET_RUN_DIR/training-summary.json" ]]; then
  resume=()
  [[ -f "$NANODET_RUN_DIR/model_last.ckpt" ]] && resume+=(--resume)
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/train_nanodet_sweep.py "$NANODET_RUN_DIR/config.yml" --seed 42 "${resume[@]}" |& tee -a "$NANODET_RUN_DIR/train.log"
fi
if [[ ! -f "$NANODET_RUN_DIR/onnx-check.json" ]]; then
  ./scripts/export_onnx.sh "$NANODET_RUN_DIR"
fi
if [[ ! -f "$NANODET_RUN_DIR/onnx-comparison.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/compare_onnx.py \
      --config "$NANODET_RUN_DIR/config.yml" --checkpoint "$NANODET_RUN_DIR/model_best/model_best.ckpt" \
      --onnx "$NANODET_ONNX" --coco-annotations "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
      --images-root "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/images" --sample-count 20 \
      --sample-manifest "$NANODET_RUN_DIR/onnx-comparison-samples.json" --output "$NANODET_RUN_DIR/onnx-comparison.json"
fi
if [[ ! -f "$NANODET_RUN_DIR/evaluations/test/coco-metrics.json" ]]; then
  NANODET_TEST_ANNOTATIONS="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
    ./scripts/evaluate_nanodet.sh "$NANODET_RUN_DIR" test
fi

conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
  --inventory "$PICODET_INVENTORY" --state "$PICODET_STATE" --runs-root "$PICODET_RUNS" --run-prefix picodet-surface3-v003 --seed 42
readonly PICODET_RUN_DIR="$PICODET_RUNS/$PICODET_RUN_ID"
if [[ ! -f "$PICODET_RUN_DIR/picodet-onnx-check.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/export_picodet_onnx.py --run-dir "$PICODET_RUN_DIR" --upstream third_party/PaddleDetection --samples 3
fi
if [[ ! -f "$PICODET_RUN_DIR/test-eval/coco-metrics.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/evaluate_picodet.py \
    --run-dir "$PICODET_RUN_DIR" --annotations "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
    --output-dir "$PICODET_RUN_DIR/test-eval" --upstream third_party/PaddleDetection
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/evaluate_coco_metrics.py \
    --ground-truth "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
    --predictions "$PICODET_RUN_DIR/test-eval/bbox.json" --output "$PICODET_RUN_DIR/test-eval/coco-metrics.json"
fi

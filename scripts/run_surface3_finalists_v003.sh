#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly NANO_RUN_ID="nanodet-plus-m-416-surface3-letterbox-v003__data-v001__seed-42__20261118-1700"
readonly NANO_RUN_DIR="$ROOT/artifacts/runs/$NANO_RUN_ID"
readonly NANO_ONNX="$NANO_RUN_DIR/model-416x416.onnx"
readonly NANO_AUDIT="$ROOT/reports/surface3-v001-data-audit.json"
readonly PICO_ROOT="$ROOT/artifacts/picodet-surface3-m-v003"
readonly PICO_RUN_ID="picodet-surface3-m-v003-picodet-m-416-surface3-seed-42"
readonly PICO_RUN_DIR="$PICO_ROOT/runs/$PICO_RUN_ID"

cd "$ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/check_surface3_training_readiness.py
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  python scripts/check_picodet_training_readiness.py --matrix configs/picodet-surface3-m-416-letterbox-v003.yml

if [[ ! -d "$NANO_RUN_DIR" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
    --base-config configs/nanodet-plus-m-416-surface3-letterbox-v003.yml --run-id "$NANO_RUN_ID" \
    --pretrained-checkpoint artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt \
    --pretrained-manifest data/manifests/nanodet-plus-m_416_checkpoint.json \
    --data-report "$NANO_AUDIT" --seed 42
fi
if [[ ! -f "$NANO_RUN_DIR/training-summary.json" ]]; then
  resume=()
  [[ -f "$NANO_RUN_DIR/model_last.ckpt" ]] && resume+=(--resume)
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/train_nanodet_sweep.py "$NANO_RUN_DIR/config.yml" --seed 42 "${resume[@]}" |& tee -a "$NANO_RUN_DIR/train.log"
fi
if [[ ! -f "$NANO_RUN_DIR/onnx-check.json" ]]; then
  ./scripts/export_onnx.sh "$NANO_RUN_DIR"
fi
if [[ ! -f "$NANO_RUN_DIR/onnx-comparison.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/compare_onnx.py \
      --config "$NANO_RUN_DIR/config.yml" --checkpoint "$NANO_RUN_DIR/model_best/model_best.ckpt" --onnx "$NANO_ONNX" \
      --coco-annotations "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
      --images-root "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/images" --sample-count 20 \
      --sample-manifest "$NANO_RUN_DIR/onnx-comparison-samples.json" --output "$NANO_RUN_DIR/onnx-comparison.json"
fi
if [[ ! -f "$NANO_RUN_DIR/evaluations/test/coco-metrics.json" ]]; then
  NANODET_TEST_ANNOTATIONS="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
    ./scripts/evaluate_nanodet.sh "$NANO_RUN_DIR" test
fi

if [[ ! -f "$PICO_ROOT/variant-inventory.json" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_picodet_sweep.py \
    --matrix configs/picodet-surface3-m-416-letterbox-v003.yml --output "$PICO_ROOT/configs"
fi
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
    --inventory "$PICO_ROOT/variant-inventory.json" --state "$PICO_ROOT/sweep-state.json" --runs-root "$PICO_ROOT/runs" --run-prefix picodet-surface3-m-v003 --seed 42
if [[ ! -f "$PICO_RUN_DIR/picodet-onnx-check.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/export_picodet_onnx.py --run-dir "$PICO_RUN_DIR" --upstream third_party/PaddleDetection --samples 3
fi
if [[ ! -f "$PICO_RUN_DIR/test-eval/coco-metrics.json" ]]; then
  conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/evaluate_picodet.py \
      --run-dir "$PICO_RUN_DIR" --annotations "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
      --output-dir "$PICO_RUN_DIR/test-eval" --upstream third_party/PaddleDetection
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/evaluate_coco_metrics.py \
    --ground-truth "$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json" \
    --predictions "$PICO_RUN_DIR/test-eval/bbox.json" --output "$PICO_RUN_DIR/test-eval/coco-metrics.json"
fi

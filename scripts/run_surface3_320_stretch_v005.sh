#!/usr/bin/env bash
# Compare direct 320-square stretch with the completed surface3 LetterBox runs.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DATASET="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-surface3-v001"
readonly TEST_ANNOTATIONS="$DATASET/annotations/instances_test.json"
readonly DATA_AUDIT="$ROOT/reports/surface3-v001-data-audit.json"
readonly EXPERIMENT_ROOT="$ROOT/artifacts/surface3-320-stretch-v005"
readonly NANO_CONFIG_ROOT="$EXPERIMENT_ROOT/configs"
readonly PICO_ROOT="$ROOT/artifacts/picodet-surface3-320-stretch-v005"
readonly PICO_RUNS="$PICO_ROOT/runs"

cd "$ROOT"

run_nanodet() {
  local run_id="$1"
  local config="$2"
  local pretrained_checkpoint="$3"
  local pretrained_manifest="$4"
  local run_dir="$ROOT/artifacts/runs/$run_id"
  local onnx_path="$run_dir/model-320x320.onnx"
  if [[ ! -d "$run_dir" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
      --base-config "$config" --run-id "$run_id" --pretrained-checkpoint "$pretrained_checkpoint" \
      --pretrained-manifest "$pretrained_manifest" --data-report "$DATA_AUDIT" --seed 42
  fi
  if [[ ! -f "$run_dir/training-summary.json" ]]; then
    local resume=()
    [[ -f "$run_dir/model_last.ckpt" ]] && resume+=(--resume)
    conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/train_nanodet_sweep.py "$run_dir/config.yml" --seed 42 "${resume[@]}" |& tee -a "$run_dir/train.log"
  fi
  if [[ ! -f "$run_dir/onnx-check.json" ]]; then
    ./scripts/export_onnx.sh "$run_dir"
  fi
  if [[ ! -f "$run_dir/onnx-comparison.json" ]]; then
    conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/compare_onnx.py \
      --config "$run_dir/config.yml" --checkpoint "$run_dir/model_best/model_best.ckpt" --onnx "$onnx_path" \
      --coco-annotations "$TEST_ANNOTATIONS" --images-root "$DATASET/images" --sample-count 20 \
      --sample-manifest "$run_dir/onnx-comparison-samples.json" --output "$run_dir/onnx-comparison.json"
  fi
  if [[ ! -f "$run_dir/evaluations/test/coco-metrics.json" ]]; then
    NANODET_TEST_ANNOTATIONS="$TEST_ANNOTATIONS" ./scripts/evaluate_nanodet.sh "$run_dir" test
  fi
}

run_picodet_training() {
  local attempt
  for attempt in 1 2; do
    if conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
      --inventory "$PICO_ROOT/variant-inventory.json" --state "$PICO_ROOT/sweep-state.json" --runs-root "$PICO_RUNS" \
      --run-prefix picodet-surface3-320-stretch-v005 --only picodet-l-320-surface3-stretch picodet-m-320-surface3-stretch --seed 42; then
      return 0
    fi
    if [[ -f "$PICO_RUNS/picodet-surface3-320-stretch-v005-picodet-m-320-surface3-stretch-seed-42/training-summary.json" ]]; then
      return 0
    fi
    if (( attempt == 2 )); then
      return 1
    fi
    printf 'PicoDet direct-stretch training exited before a terminal summary; resuming newest complete checkpoints.\n' >&2
  done
}

run_picodet_postprocess() {
  local run_dir="$1"
  if [[ ! -f "$run_dir/picodet-onnx-check.json" ]]; then
    conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/export_picodet_onnx.py \
      --run-dir "$run_dir" --upstream third_party/PaddleDetection --samples 20
  fi
  if [[ ! -f "$run_dir/test-eval/coco-metrics.json" ]]; then
    conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/evaluate_picodet.py \
      --run-dir "$run_dir" --annotations "$TEST_ANNOTATIONS" --output-dir "$run_dir/test-eval" --upstream third_party/PaddleDetection
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/evaluate_coco_metrics.py \
      --ground-truth "$TEST_ANNOTATIONS" --predictions "$run_dir/test-eval/bbox.json" --output "$run_dir/test-eval/coco-metrics.json"
  fi
  if [[ ! -f "$run_dir/picodet-onnx-decoder-check.json" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/compare_picodet_onnx.py \
      --run-dir "$run_dir" --annotations "$TEST_ANNOTATIONS" --images-root "$DATASET/images" --output "$run_dir/picodet-onnx-decoder-check.json"
  fi
}

PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_surface3_stretch_configs.py --output "$NANO_CONFIG_ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/check_surface3_training_readiness.py \
  --nanodet-config "$NANO_CONFIG_ROOT/nanodet-plus-m-320-surface3-stretch-v005.yml" \
  --picodet-matrix configs/picodet-surface3-320-stretch-v005.yml --expected-input-size 320 --geometry direct_stretch
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  python scripts/check_picodet_training_readiness.py --matrix configs/picodet-surface3-320-stretch-v005.yml

run_nanodet "nanodet-plus-m-320-surface3-stretch-v005__data-v001__seed-42__20261120-0000" \
  "$NANO_CONFIG_ROOT/nanodet-plus-m-320-surface3-stretch-v005.yml" \
  artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt data/manifests/nanodet-plus-m_416_checkpoint.json
run_nanodet "nanodet-plus-m15-320-surface3-stretch-v005__data-v001__seed-42__20261120-0000" \
  "$NANO_CONFIG_ROOT/nanodet-plus-m15-320-surface3-stretch-v005.yml" \
  artifacts/pretrained/nanodet-plus-m15-416.ckpt data/manifests/nanodet-plus-m15-416_checkpoint.json

if [[ ! -f "$PICO_ROOT/variant-inventory.json" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_picodet_sweep.py \
    --matrix configs/picodet-surface3-320-stretch-v005.yml --output "$PICO_ROOT/configs"
fi
run_picodet_training
run_picodet_postprocess "$PICO_RUNS/picodet-surface3-320-stretch-v005-picodet-l-320-surface3-stretch-seed-42"
run_picodet_postprocess "$PICO_RUNS/picodet-surface3-320-stretch-v005-picodet-m-320-surface3-stretch-seed-42"

if [[ ! -f "$EXPERIMENT_ROOT/onnx-evaluation/summary.json" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    python scripts/evaluate_onnx_splits.py --jobs configs/surface3-320-direct-stretch-v005-onnx-jobs.json \
    --output-dir "$EXPERIMENT_ROOT/onnx-evaluation" --provider cuda
fi

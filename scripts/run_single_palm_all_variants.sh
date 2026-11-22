#!/usr/bin/env bash
# Train every supported NanoDet and PicoDet variant on the audited one-class task.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPERIMENT_ROOT="$ROOT/artifacts/single-palm-v001"
readonly DATASET="$ROOT/data/portable-hagridv2-mediapipe-hand/coco-single-palm-v001"
readonly DATA_AUDIT="$EXPERIMENT_ROOT/single-palm-data-audit.json"
readonly NANO_MATRIX="$ROOT/configs/single-palm-nanodet-v001.yml"
readonly PICO_LETTERBOX_MATRIX="$ROOT/configs/picodet-single-palm-letterbox-v001.yml"
readonly PICO_STRETCH_MATRIX="$ROOT/configs/picodet-single-palm-stretch-v001.yml"
readonly NANO_RUNS="$EXPERIMENT_ROOT/nanodet/runs"
readonly PICO_ROOT="$EXPERIMENT_ROOT/picodet"
readonly PICO_RUNS="$PICO_ROOT/runs"
readonly STATE="${SINGLE_PALM_STATE_PATH:-$EXPERIMENT_ROOT/training-state.json}"
readonly COMPARISON_PATH="${SINGLE_PALM_COMPARISON_PATH:-$EXPERIMENT_ROOT/interim-comparison.json}"
readonly FINAL_COMPARISON_PATH="${SINGLE_PALM_FINAL_COMPARISON_PATH:-$EXPERIMENT_ROOT/final-comparison.json}"
readonly PICO_SWEEP_STATE_PREFIX="${SINGLE_PALM_PICO_SWEEP_STATE_PREFIX:-$PICO_ROOT/sweep-state}"
readonly TEST_ANNOTATIONS="$DATASET/annotations/instances_test.json"

usage() {
  cat <<'EOF'
Usage: run_single_palm_all_variants.sh [--geometry all|letterbox|direct_stretch] [--family all|nanodet|picodet]

Without arguments, train the complete matrix in the canonical order.  A geometry
or family selection runs that slice only, enabling independent, resource-isolated lanes.
EOF
}

geometry_selector="all"
family_selector="all"
while (($#)); do
  case "$1" in
    --geometry)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      geometry_selector="$2"
      shift 2
      ;;
    --family)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      family_selector="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done
case "$geometry_selector" in
  all|letterbox|direct_stretch) ;;
  *)
    printf 'Unsupported geometry selector: %s\n' "$geometry_selector" >&2
    exit 2
    ;;
esac
case "$family_selector" in
  all|nanodet|picodet) ;;
  *)
    printf 'Unsupported family selector: %s\n' "$family_selector" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$EXPERIMENT_ROOT" "$NANO_RUNS" "$PICO_RUNS"

record_state() {
  local family="$1"
  local geometry="$2"
  local variant="$3"
  local status="$4"
  local run_dir="$5"
  local detail="${6:-}"
  FAMILY="$family" GEOMETRY="$geometry" VARIANT="$variant" STATUS="$status" RUN_DIR="$run_dir" DETAIL="$detail" TRAINING_STATE_PATH="$STATE" \
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["TRAINING_STATE_PATH"])
payload = json.loads(path.read_text()) if path.is_file() else {"schema_version": 1, "tasks": {}}
key = ":".join(os.environ[name] for name in ("FAMILY", "GEOMETRY", "VARIANT"))
payload["tasks"][key] = {
    "family": os.environ["FAMILY"], "geometry": os.environ["GEOMETRY"],
    "variant": os.environ["VARIANT"], "status": os.environ["STATUS"],
    "run_dir": os.environ["RUN_DIR"], "detail": os.environ["DETAIL"],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
tmp = path.with_suffix(".incoming")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

refresh_comparison() {
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/summarize_single_palm_models.py \
    --experiment-root "$EXPERIMENT_ROOT" --output "$COMPARISON_PATH"
}

with_run_lock() {
  local run_dir="$1"
  shift
  local run_lock_fd
  mkdir -p "$run_dir"
  exec {run_lock_fd}>"$run_dir/.run.lock"
  flock -x "$run_lock_fd"
  local status=0
  if "$@"; then
    status=0
  else
    status=$?
  fi
  exec {run_lock_fd}>&-
  return "$status"
}

with_picodet_pipeline_lock() {
  local pico_lock_fd
  exec {pico_lock_fd}>"$EXPERIMENT_ROOT/.picodet-pipeline.lock"
  flock -x "$pico_lock_fd"
  local status=0
  if "$@"; then
    status=0
  else
    status=$?
  fi
  exec {pico_lock_fd}>&-
  return "$status"
}

run_nanodet() {
  local geometry="$1"
  local variant="$2"
  local run_id="single-palm-v001-${geometry}-${variant}__data-v001__seed-42"
  with_run_lock "$NANO_RUNS/$run_id" run_nanodet_locked "$@"
}

run_nanodet_locked() {
  local geometry="$1"
  local variant="$2"
  local config="$3"
  local checkpoint="$4"
  local manifest="$5"
  local run_id="single-palm-v001-${geometry}-${variant}__data-v001__seed-42"
  local run_dir="$NANO_RUNS/$run_id"
  local dimensions
  dimensions="$(PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); print(*c["data"]["val"]["input_size"])' "$config")"
  local width height
  read -r width height <<< "$dimensions"
  if ! PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/check_single_palm_training_readiness.py \
    --config "$config" --geometry "$geometry"; then
    record_state nanodet "$geometry" "$variant" failed "$run_dir" "readiness check failed"
    return 1
  fi
  record_state nanodet "$geometry" "$variant" running "$run_dir"
  if [[ ! -d "$run_dir" ]]; then
    if ! PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
      --base-config "$config" --run-id "$run_id" --runs-root "$NANO_RUNS" \
      --pretrained-checkpoint "$checkpoint" --pretrained-manifest "$manifest" --data-report "$DATA_AUDIT" --seed 42; then
      record_state nanodet "$geometry" "$variant" failed "$run_dir" "run creation failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/training-summary.json" ]]; then
    local resume=()
    [[ -f "$run_dir/model_last.ckpt" ]] && resume+=(--resume)
    if ! conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/train_nanodet_sweep.py "$run_dir/config.yml" --seed 42 "${resume[@]}" |& tee -a "$run_dir/train.log"
    then
      record_state nanodet "$geometry" "$variant" failed "$run_dir" "training failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/onnx-check.json" ]]; then
    if ! ./scripts/export_onnx.sh "$run_dir"; then
      record_state nanodet "$geometry" "$variant" failed "$run_dir" "ONNX export failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/onnx-comparison.json" ]]; then
    if ! conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet" python scripts/compare_onnx.py \
      --config "$run_dir/config.yml" --checkpoint "$run_dir/model_best/model_best.ckpt" \
      --onnx "$run_dir/model-${width}x${height}.onnx" --coco-annotations "$TEST_ANNOTATIONS" \
      --images-root "$DATASET/images" --sample-count 20 --sample-manifest "$run_dir/onnx-comparison-samples.json" \
      --output "$run_dir/onnx-comparison.json"; then
      record_state nanodet "$geometry" "$variant" failed "$run_dir" "ONNX comparison failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/evaluations/test/coco-metrics.json" ]]; then
    if ! NANODET_TEST_ANNOTATIONS="$TEST_ANNOTATIONS" ./scripts/evaluate_nanodet.sh "$run_dir" test; then
      record_state nanodet "$geometry" "$variant" failed "$run_dir" "frozen test evaluation failed"
      return 1
    fi
  fi
  record_state nanodet "$geometry" "$variant" complete "$run_dir"
  refresh_comparison
}

run_picodet_postprocess() {
  local geometry="$1"
  local variant="$2"
  local run_dir="$3"
  record_state picodet "$geometry" "$variant" running "$run_dir"
  if [[ ! -f "$run_dir/picodet-onnx-check.json" ]]; then
    if ! conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/export_picodet_onnx.py \
      --run-dir "$run_dir" --upstream third_party/PaddleDetection --samples 20; then
      record_state picodet "$geometry" "$variant" failed "$run_dir" "ONNX export failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/test-eval/coco-metrics.json" ]]; then
    if ! conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/evaluate_picodet.py \
      --run-dir "$run_dir" --annotations "$TEST_ANNOTATIONS" --output-dir "$run_dir/test-eval" --upstream third_party/PaddleDetection; then
      record_state picodet "$geometry" "$variant" failed "$run_dir" "frozen test inference failed"
      return 1
    fi
    if ! PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/evaluate_coco_metrics.py \
      --ground-truth "$TEST_ANNOTATIONS" --predictions "$run_dir/test-eval/bbox.json" --output "$run_dir/test-eval/coco-metrics.json"; then
      record_state picodet "$geometry" "$variant" failed "$run_dir" "frozen test metric calculation failed"
      return 1
    fi
  fi
  if [[ ! -f "$run_dir/picodet-onnx-decoder-check.json" ]]; then
    if ! PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/compare_picodet_onnx.py \
      --run-dir "$run_dir" --annotations "$TEST_ANNOTATIONS" --images-root "$DATASET/images" \
      --output "$run_dir/picodet-onnx-decoder-check.json"; then
      record_state picodet "$geometry" "$variant" failed "$run_dir" "ONNX decoder comparison failed"
      return 1
    fi
  fi
  record_state picodet "$geometry" "$variant" complete "$run_dir"
  refresh_comparison
}

run_picodet_variant() {
  local geometry="$1"
  local variant="$2"
  local run_dir="$3"
  local inventory="$4"
  local state="$5"
  if ! conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
    env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
    --inventory "$inventory" --state "$state" --runs-root "$PICO_RUNS" \
    --run-prefix "single-palm-v001-${geometry}" --only "$variant" --seed 42; then
    record_state picodet "$geometry" "$variant" failed "$run_dir" "training failed"
    return 1
  fi
  run_picodet_postprocess "$geometry" "$variant" "$run_dir"
}

run_picodet_matrix() {
  local geometry="$1"
  local matrix="$2"
  local config_root="$PICO_ROOT/configs-$geometry"
  local inventory="$PICO_ROOT/variant-inventory-$geometry.json"
  local state="${PICO_SWEEP_STATE_PREFIX}-${geometry}.json"
  if [[ ! -f "$inventory" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_picodet_sweep.py --matrix "$matrix" --output "$config_root"
    mv "$config_root/../variant-inventory.json" "$inventory"
  fi
  while IFS=$'\t' read -r variant size; do
    local run_dir="$PICO_RUNS/single-palm-v001-${geometry}-${variant}-seed-42"
    record_state picodet "$geometry" "$variant" running "$run_dir"
    if ! with_run_lock "$run_dir" with_picodet_pipeline_lock run_picodet_variant "$geometry" "$variant" "$run_dir" "$inventory" "$state"; then
      refresh_comparison
      continue
    fi
  done < <(MATRIX="$matrix" conda run --no-capture-output -n palm-nanodet python - <<'PY'
import os
import yaml
for item in yaml.safe_load(open(os.environ["MATRIX"], encoding="utf-8"))["variants"]:
    print(f'{item["id"]}\t{item["input_size"]}')
PY
)
}

if [[ ! -d "$DATASET" ]]; then
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_single_palm_coco.py
fi
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/check_coco_dataset.py \
  --category-contract single-palm --split train "$DATASET/annotations/instances_train.json" "$DATASET/images" \
  --split dev "$DATASET/annotations/instances_dev.json" "$DATASET/images" \
  --split gate "$DATASET/annotations/instances_gate.json" "$DATASET/images" \
  --split test "$DATASET/annotations/instances_test.json" "$DATASET/images" \
  --output "$DATA_AUDIT"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/acquire_picodet_pretrained.py --matrix "$PICO_LETTERBOX_MATRIX" --output artifacts/pretrained/picodet

if [[ "$geometry_selector" == "all" ]]; then
  geometries=(letterbox direct_stretch)
else
  geometries=("$geometry_selector")
fi

if [[ "$family_selector" != "picodet" ]]; then
  for geometry in "${geometries[@]}"; do
    config_root="$EXPERIMENT_ROOT/nanodet/configs-$geometry"
    inventory="$EXPERIMENT_ROOT/nanodet/variant-inventory-$geometry.json"
    if [[ ! -f "$inventory" ]]; then
      PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_single_palm_nanodet.py \
        --matrix "$NANO_MATRIX" --geometry "$geometry" --output "$config_root"
      mv "$config_root/../variant-inventory.json" "$inventory"
    fi
    while IFS=$'\t' read -r variant config checkpoint manifest; do
      run_nanodet "$geometry" "$variant" "$config" "$checkpoint" "$manifest" || refresh_comparison
    done < <(INVENTORY="$inventory" conda run --no-capture-output -n palm-nanodet python - <<'PY'
import json
import os
for item in json.load(open(os.environ["INVENTORY"], encoding="utf-8"))["variants"]:
    print("\t".join((item["id"], item["config"], item["checkpoint"], item["manifest"])))
PY
)
  done
fi

if [[ "$family_selector" != "nanodet" ]]; then
  for geometry in "${geometries[@]}"; do
    case "$geometry" in
      letterbox)
        run_picodet_matrix letterbox "$PICO_LETTERBOX_MATRIX"
        ;;
      direct_stretch)
        run_picodet_matrix direct_stretch "$PICO_STRETCH_MATRIX"
        ;;
    esac
  done
fi
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/summarize_single_palm_models.py \
  --experiment-root "$EXPERIMENT_ROOT" --output "$FINAL_COMPARISON_PATH"

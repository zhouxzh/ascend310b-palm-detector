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
readonly STATE="$EXPERIMENT_ROOT/training-state.json"
readonly TEST_ANNOTATIONS="$DATASET/annotations/instances_test.json"

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

run_nanodet() {
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
  PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/check_single_palm_training_readiness.py \
    --config "$config" --geometry "$geometry"
  record_state nanodet "$geometry" "$variant" running "$run_dir"
  if [[ ! -d "$run_dir" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/create_run.py \
      --base-config "$config" --run-id "$run_id" --runs-root "$NANO_RUNS" \
      --pretrained-checkpoint "$checkpoint" --pretrained-manifest "$manifest" --data-report "$DATA_AUDIT" --seed 42
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
      --config "$run_dir/config.yml" --checkpoint "$run_dir/model_best/model_best.ckpt" \
      --onnx "$run_dir/model-${width}x${height}.onnx" --coco-annotations "$TEST_ANNOTATIONS" \
      --images-root "$DATASET/images" --sample-count 20 --sample-manifest "$run_dir/onnx-comparison-samples.json" \
      --output "$run_dir/onnx-comparison.json"
  fi
  if [[ ! -f "$run_dir/evaluations/test/coco-metrics.json" ]]; then
    NANODET_TEST_ANNOTATIONS="$TEST_ANNOTATIONS" ./scripts/evaluate_nanodet.sh "$run_dir" test
  fi
  record_state nanodet "$geometry" "$variant" complete "$run_dir"
}

run_picodet_postprocess() {
  local geometry="$1"
  local variant="$2"
  local run_dir="$3"
  record_state picodet "$geometry" "$variant" running "$run_dir"
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
      --run-dir "$run_dir" --annotations "$TEST_ANNOTATIONS" --images-root "$DATASET/images" \
      --output "$run_dir/picodet-onnx-decoder-check.json"
  fi
  record_state picodet "$geometry" "$variant" complete "$run_dir"
}

run_picodet_matrix() {
  local geometry="$1"
  local matrix="$2"
  local config_root="$PICO_ROOT/configs-$geometry"
  local inventory="$PICO_ROOT/variant-inventory-$geometry.json"
  local state="$PICO_ROOT/sweep-state-$geometry.json"
  if [[ ! -f "$inventory" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_picodet_sweep.py --matrix "$matrix" --output "$config_root"
    mv "$config_root/../variant-inventory.json" "$inventory"
  fi
  while IFS=$'\t' read -r variant size; do
    local run_dir="$PICO_RUNS/single-palm-v001-${geometry}-${variant}-seed-42"
    record_state picodet "$geometry" "$variant" queued "$run_dir"
    conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
      env PYTHONPATH="$ROOT/src:$ROOT/third_party/PaddleDetection" python scripts/run_picodet_sweep.py \
      --inventory "$inventory" --state "$state" --runs-root "$PICO_RUNS" \
      --run-prefix "single-palm-v001-${geometry}" --only "$variant" --seed 42
    run_picodet_postprocess "$geometry" "$variant" "$run_dir"
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

for geometry in letterbox direct_stretch; do
  config_root="$EXPERIMENT_ROOT/nanodet/configs-$geometry"
  inventory="$EXPERIMENT_ROOT/nanodet/variant-inventory-$geometry.json"
  if [[ ! -f "$inventory" ]]; then
    PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/prepare_single_palm_nanodet.py \
      --matrix "$NANO_MATRIX" --geometry "$geometry" --output "$config_root"
    mv "$config_root/../variant-inventory.json" "$inventory"
  fi
  while IFS=$'\t' read -r variant config checkpoint manifest; do
    run_nanodet "$geometry" "$variant" "$config" "$checkpoint" "$manifest"
  done < <(INVENTORY="$inventory" conda run --no-capture-output -n palm-nanodet python - <<'PY'
import json
import os
for item in json.load(open(os.environ["INVENTORY"], encoding="utf-8"))["variants"]:
    print("\t".join((item["id"], item["config"], item["checkpoint"], item["manifest"])))
PY
)
done

run_picodet_matrix letterbox "$PICO_LETTERBOX_MATRIX"
run_picodet_matrix direct_stretch "$PICO_STRETCH_MATRIX"
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet python scripts/summarize_single_palm_models.py \
  --experiment-root "$EXPERIMENT_ROOT" --output "$EXPERIMENT_ROOT/final-comparison.json"

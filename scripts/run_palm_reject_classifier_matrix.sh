#!/usr/bin/env bash
set -u

ROOT=/home/zhong/Documents/ascend310b-palm-detector
RUN_ID=20261123-palm-reject-three-backbones
OUT_ROOT="$ROOT/artifacts/$RUN_ID"
LOG_ROOT="$OUT_ROOT/logs"
PY=/home/zhong/anaconda3/envs/torch/bin/python
DATASET="$ROOT/data/palm-reject-roi-v005"
BACKBONE_CHECKPOINT="$ROOT/artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt"

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$LOG_ROOT" "$OUT_ROOT/tensorboard"
printf '{"event":"started","run_id":"%s","timestamp":"%s"}\n' "$RUN_ID" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"

specs=(
  "nanodet_shufflenetv2_x1|nanodet-shufflenetv2-x1"
  "lcnet_100|lcnet-100"
  "efficientnet_b0|efficientnet-b0"
)

for spec in "${specs[@]}"; do
  backbone=${spec%%|*}
  name=${spec#*|}
  output="$OUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  if [[ -e "$output" ]]; then
    printf '{"event":"failed-precondition","backbone":"%s","reason":"output-exists","timestamp":"%s"}\n' "$backbone" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"
    continue
  fi
  printf '{"event":"run-started","backbone":"%s","output":"%s","timestamp":"%s"}\n' "$backbone" "$output" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"
  "$PY" "$ROOT/scripts/train_palm_reject_classifier.py" \
    --dataset "$DATASET" \
    --output "$output" \
    --backbone "$backbone" \
    --backbone-checkpoint "$BACKBONE_CHECKPOINT" \
    --tensorboard-dir "$OUT_ROOT/tensorboard/$name" \
    --epochs 30 \
    --batch-size 64 \
    --workers 8 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --patience 8 \
    --seed 43 > "$log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    printf '{"event":"run-completed","backbone":"%s","status":"completed","timestamp":"%s"}\n' "$backbone" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"
  else
    printf '{"event":"run-completed","backbone":"%s","status":"failed","exit_code":%d,"timestamp":"%s"}\n' "$backbone" "$status" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"
  fi
done

touch "$OUT_ROOT/MATRIX_COMPLETE"
printf '{"event":"finished","run_id":"%s","timestamp":"%s"}\n' "$RUN_ID" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/matrix-events.jsonl"

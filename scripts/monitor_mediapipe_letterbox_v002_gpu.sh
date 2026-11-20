#!/usr/bin/env bash
set -euo pipefail

readonly INTERVAL="${1:-30}"
readonly MAX_TEMP="${2:-82}"
readonly LOG="${3:-artifacts/letterbox-v002/gpu-monitor.csv}"
mkdir -p "$(dirname "$LOG")"
if [[ ! -s "$LOG" ]]; then
  printf 'timestamp,index,name,temp_c,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,sm_clock_mhz,throttle_reasons\n' >> "$LOG"
fi

hot=0
while true; do
  sample="$(nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits)"
  printf '%s\n' "$sample" >> "$LOG"
  temp="$(cut -d, -f4 <<< "$sample" | tr -d ' ' | head -n1)"
  if [[ "$temp" =~ ^[0-9]+$ ]] && (( temp >= MAX_TEMP )); then
    ((hot+=1))
    printf 'WARNING: MediaPipe LetterBox v002 GPU temperature %sC >= %sC (%s/3).\n' "$temp" "$MAX_TEMP" "$hot" >&2
    if (( hot >= 3 )); then
      pgrep -f '[t]rain_nanodet_sweep.py.*mediapipe-letterbox-v002' | xargs -r kill -TERM
      pgrep -f '[t]rain_picodet.py.*picodet-v002' | xargs -r kill -TERM
      pgrep -f '[r]un_mediapipe_letterbox_v002.sh' | xargs -r kill -TERM
      break
    fi
  else
    hot=0
  fi
  sleep "$INTERVAL"
done

#!/usr/bin/env bash
set -euo pipefail

readonly INTERVAL="${1:-30}"
readonly MAX_TEMP_C="${2:-82}"
readonly LOG="${3:-artifacts/sweeps/palm-v001/gpu-monitor.csv}"
if [[ ! "$INTERVAL" =~ ^[1-9][0-9]*$ || ! "$MAX_TEMP_C" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [INTERVAL_SECONDS] [MAX_TEMP_C] [CSV_PATH]" >&2
  exit 2
fi
mkdir -p "$(dirname "$LOG")"
if [[ ! -s "$LOG" ]]; then
  printf 'timestamp,index,name,temp_c,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,sm_clock_mhz,throttle_reasons\n' >> "$LOG"
fi
consecutive_hot=0
while true; do
  sample="$(nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits)"
  printf '%s\n' "$sample" >> "$LOG"
  temperature="$(cut -d, -f4 <<< "$sample" | tr -d ' ' | head -n 1)"
  throttle="$(cut -d, -f10 <<< "$sample" | tr -d ' ' | head -n 1)"
  if [[ "$temperature" =~ ^[0-9]+$ ]] && (( temperature >= MAX_TEMP_C )); then
    ((consecutive_hot+=1))
    printf 'WARNING: GPU temperature %sC is at/above %sC (%s/3 consecutive samples).\n' "$temperature" "$MAX_TEMP_C" "$consecutive_hot" >&2
    if (( consecutive_hot >= 3 )); then
      echo "GPU temperature remained high; stopping the sweep queue for cooling." >&2
      pgrep -f '[p]ython scripts/train_nanodet_sweep.py' | xargs -r kill -TERM
      pgrep -f '[p]ython scripts/run_nanodet_sweep.py' | xargs -r kill -TERM
      break
    fi
  else
    consecutive_hot=0
  fi
  if [[ "$throttle" == *0000000000000004 ]]; then
    printf 'INFO: NVIDIA reports software power-cap clock management (%s); this is not thermal throttling.\n' "$throttle" >&2
  elif [[ "$throttle" != "0x0000000000000000" && -n "$throttle" ]]; then
    printf 'WARNING: NVIDIA reports non-power-cap clock throttle reasons: %s\n' "$throttle" >&2
  fi
  sleep "$INTERVAL"
done

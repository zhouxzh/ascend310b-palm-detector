#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 COMMAND [ARG...]" >&2
  exit 2
fi

readonly LIBRARY_DIR="/usr/lib/x86_64-linux-gnu"
driver_version=""
if [[ -r /proc/driver/nvidia/version ]]; then
  driver_version="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -n 1)"
fi
if [[ -n "$driver_version" && -f "$LIBRARY_DIR/libcuda.so.$driver_version" && -f "$LIBRARY_DIR/libnvidia-ml.so.$driver_version" ]]; then
  preload="$LIBRARY_DIR/libcuda.so.$driver_version:$LIBRARY_DIR/libnvidia-ml.so.$driver_version"
  if [[ -n "${LD_PRELOAD-}" ]]; then
    preload="$preload:$LD_PRELOAD"
  fi
  exec env LD_PRELOAD="$preload" "$@"
fi
exec "$@"

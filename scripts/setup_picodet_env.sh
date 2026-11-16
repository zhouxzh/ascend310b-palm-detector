#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ENVIRONMENT="${1:-palm-picodet}"
readonly WHEEL="$ROOT/artifacts/picodet-v001/wheels/paddlepaddle_gpu-3.5.0.dev20261113+12c390870c2-cp311-cp311-linux_x86_64.whl"
readonly EXPECTED_SHA256="caacee3940aaf719b685a490110630d58778b76849b248a2f17d719f3ac90c91"

if [[ ! -f "$WHEEL" ]]; then
  echo "Missing validated Paddle CUDA wheel: $WHEEL" >&2
  echo "Provide the exact wheel through PICODET_PADDLE_WHEEL and rerun this script." >&2
  exit 1
fi
actual="$(sha256sum "$WHEEL" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  echo "Paddle CUDA wheel checksum mismatch" >&2
  exit 1
fi
conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install "$WHEEL"
conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install -r "$ROOT/requirements.picodet.txt"

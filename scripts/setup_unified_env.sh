#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ENVIRONMENT="${1:-palm-nanodet}"
readonly WHEEL="${PICODET_PADDLE_WHEEL:-$ROOT/artifacts/wheels/paddlepaddle_gpu-3.5.0.dev20261113+12c390870c2-cp311-cp311-linux_x86_64.whl}"
readonly EXPECTED_SHA256="caacee3940aaf719b685a490110630d58778b76849b248a2f17d719f3ac90c91"
readonly TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if ! conda env list | awk '{print $1}' | grep -qx "$ENVIRONMENT"; then
  echo "Conda environment not found: $ENVIRONMENT" >&2
  exit 1
fi
if [[ ! -d "$ROOT/third_party/nanodet/.git" || ! -d "$ROOT/third_party/PaddleDetection/.git" ]]; then
  echo "Fetch both pinned upstream repositories before setting up the environment." >&2
  exit 1
fi
if [[ ! -f "$WHEEL" ]]; then
  echo "Missing validated Paddle CUDA wheel: $WHEEL" >&2
  exit 1
fi
actual="$(sha256sum "$WHEEL" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  echo "Paddle CUDA wheel checksum mismatch" >&2
  exit 1
fi

torch_version="$(conda run --no-capture-output -n "$ENVIRONMENT" python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
torchvision_version="$(conda run --no-capture-output -n "$ENVIRONMENT" python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || true)"
if [[ "$torch_version" != "2.8.0+cu128" || "$torchvision_version" != "0.23.0+cu128" ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install \
    --index-url "$TORCH_INDEX" torch==2.8.0+cu128 torchvision==0.23.0+cu128
fi

paddle_version="$(conda run --no-capture-output -n "$ENVIRONMENT" python -c 'import paddle; print(paddle.__version__)' 2>/dev/null || true)"
if [[ "$paddle_version" == "3.5.0.dev20261113+12c390870c2" ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install --no-deps "$WHEEL"
else
  conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install "$WHEEL"
fi

# PaddleDetection and MediaPipe both require OpenCV. Pin the base and contrib
# wheels to the same compatible release in requirements.txt.
conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install --no-deps --upgrade -r "$ROOT/requirements.txt"
conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install --no-deps -e "$ROOT/third_party/nanodet"
conda run --no-capture-output -n "$ENVIRONMENT" python -m pip install --no-deps -e "$ROOT/third_party/PaddleDetection"
conda run --no-capture-output -n "$ENVIRONMENT" "$ROOT/scripts/run_with_host_nvidia.sh" \
  python "$ROOT/scripts/check_unified_training_readiness.py"

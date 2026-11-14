#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
conda run --no-capture-output -n palm-nanodet "$ROOT/scripts/run_with_host_nvidia.sh" \
  env PYTHONPATH=src python scripts/check_training_readiness.py

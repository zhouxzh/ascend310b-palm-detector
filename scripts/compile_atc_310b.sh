#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ONNX_PATH OUTPUT_DIR SOC_VERSION" >&2
  echo "Example SOC_VERSION values must be confirmed on the target board, such as Ascend310B1." >&2
  exit 2
fi
readonly ONNX_PATH="$1"
readonly OUTPUT_DIR="$2"
readonly SOC_VERSION="$3"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v atc >/dev/null || { echo "atc is not on PATH; source the target CANN environment first." >&2; exit 2; }
[[ -f "$ONNX_PATH" ]] || { echo "ONNX file does not exist: $ONNX_PATH" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
{
  date --utc +%Y-%m-%dT%H:%M:%SZ
  atc --version || true
  sha256sum "$ONNX_PATH"
  printf 'soc_version=%s\n' "$SOC_VERSION"
} > "$OUTPUT_DIR/environment.txt"

compile() {
  local name="$1"
  local precision_arg="$2"
  local target="$OUTPUT_DIR/$name/nanodet-plus-m-320-palm"
  mkdir -p "$(dirname "$target")"
  if [[ -n "$precision_arg" ]]; then
    atc --model="$ONNX_PATH" --framework=5 --output="$target" --input_format=NCHW \
      --input_shape='data:1,3,320,320' --soc_version="$SOC_VERSION" "$precision_arg" \
      |& tee "$OUTPUT_DIR/$name/atc.log"
  else
    atc --model="$ONNX_PATH" --framework=5 --output="$target" --input_format=NCHW \
      --input_shape='data:1,3,320,320' --soc_version="$SOC_VERSION" \
      |& tee "$OUTPUT_DIR/$name/atc.log"
  fi
  sha256sum "$target.om" > "$OUTPUT_DIR/$name/om.sha256"
}

compile default ''
compile allow-fp32-to-fp16 '--precision_mode=allow_fp32_to_fp16'
printf 'ATC conversion complete: %s\n' "$OUTPUT_DIR"

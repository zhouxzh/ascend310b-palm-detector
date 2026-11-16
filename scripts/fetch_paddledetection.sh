#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly URL="https://github.com/PaddlePaddle/PaddleDetection.git"
readonly REVISION="b25522a0f4bde8c80603f3ba5e3472059972e3b5"
readonly TARGET="$ROOT/third_party/PaddleDetection"

if [[ -e "$TARGET" && ! -d "$TARGET/.git" ]]; then
  echo "Refusing to replace non-Git path: $TARGET" >&2
  exit 2
fi
if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone --no-checkout "$URL" "$TARGET"
fi
if [[ "$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || true)" == "$REVISION" ]]; then
  printf 'PaddleDetection ready at %s\n' "$REVISION"
  exit 0
fi
git -C "$TARGET" fetch --depth 1 origin "$REVISION"
git -C "$TARGET" checkout --detach "$REVISION"
actual="$(git -C "$TARGET" rev-parse HEAD)"
[[ "$actual" == "$REVISION" ]] || { echo "PaddleDetection revision mismatch: $actual" >&2; exit 2; }
printf 'PaddleDetection ready at %s\n' "$actual"

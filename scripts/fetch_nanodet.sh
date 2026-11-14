#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly URL="https://github.com/RangiLyu/nanodet.git"
readonly REVISION="d3fb34fa91d6020f273d6d063bf324dcd97bac12"
readonly TARGET="$ROOT/third_party/nanodet"

if [[ -e "$TARGET" && ! -d "$TARGET/.git" ]]; then
  echo "Refusing to replace non-Git path: $TARGET" >&2
  exit 2
fi
if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone "$URL" "$TARGET"
fi
git -C "$TARGET" fetch --tags --force origin
git -C "$TARGET" checkout --detach "$REVISION"
actual="$(git -C "$TARGET" rev-parse HEAD)"
[[ "$actual" == "$REVISION" ]] || { echo "NanoDet revision mismatch: $actual" >&2; exit 2; }
printf 'NanoDet ready at %s\n' "$actual"

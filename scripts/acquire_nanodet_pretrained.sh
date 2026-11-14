#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly URL="https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt"
readonly CHECKPOINT="$ROOT/artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt"
readonly MANIFEST="$ROOT/data/manifests/nanodet-plus-m_416_checkpoint.json"

mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$MANIFEST")"
if [[ ! -f "$CHECKPOINT" ]]; then
  temporary="$(mktemp "$CHECKPOINT.download.XXXXXX")"
  trap 'rm -f "$temporary"' EXIT
  curl --fail --location --retry 3 --output "$temporary" "$URL"
  mv "$temporary" "$CHECKPOINT"
  trap - EXIT
fi
digest="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
bytes="$(stat --format=%s "$CHECKPOINT")"
timestamp="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
cat > "$MANIFEST" <<EOF
{
  "model_name": "nanodet-plus-m_416_checkpoint",
  "upstream_url": "$URL",
  "upstream_revision_or_release": "v1.0.0-alpha-1",
  "license": "Apache-2.0",
  "local_path": "artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt",
  "bytes": $bytes,
  "sha256": "$digest",
  "downloaded_at_utc": "$timestamp"
}
EOF
printf 'checkpoint=%s\nmanifest=%s\nsha256=%s\n' "$CHECKPOINT" "$MANIFEST" "$digest"

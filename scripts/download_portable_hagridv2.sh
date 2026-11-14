#!/usr/bin/env bash
set -euo pipefail

# Downloads the public dataset through hf-mirror. LFS object IDs are SHA-256
# digests and are checked after every complete file. Range chunks make slow
# proxy links resumable without trusting a partly written Parquet file.
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO="zhouxzh/portable-hagridv2-mediapipe-hand"
readonly REVISION="e343025bd4fd37719dbc269201d6b24124f0c432"
readonly MIRROR="https://hf-mirror.com"
readonly TARGET="${1:-$ROOT/data/external/portable-hagridv2-mediapipe-hand}"
readonly WORKERS="${HF_MIRROR_RANGE_WORKERS:-4}"
readonly CHUNK_SIZE="${HF_MIRROR_CHUNK_SIZE:-4194304}"
readonly CDN_IP="${HF_MIRROR_CDN_IP:-}"
readonly HUB_IP="${HF_HUB_IP:-$(getent ahostsv4 huggingface.co | awk 'NR == 1 { print $1 }')}"

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "HF_MIRROR_RANGE_WORKERS must be a positive integer" >&2; exit 2; }
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "HF_MIRROR_CHUNK_SIZE must be a positive integer" >&2; exit 2; }
if [[ -n "$CDN_IP" ]] && [[ ! "$CDN_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "HF_MIRROR_CDN_IP must be an IPv4 address" >&2
  exit 2
fi
if [[ -z "$HUB_IP" || ! "$HUB_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "could not resolve huggingface.co; set HF_HUB_IP to a valid IPv4 address" >&2
  exit 2
fi
mkdir -p "$TARGET/partials"

download_small() {
  local filename="$1"
  if [[ -s "$TARGET/$filename" ]]; then
    return
  fi
  curl --fail --silent --show-error --http1.1 --location --retry 5 --retry-all-errors \
    --output "$TARGET/$filename" "$MIRROR/datasets/$REPO/resolve/$REVISION/$filename"
}

download_large() {
  local filename="$1"
  local size="$2"
  local digest="$3"
  local temporary_dir="$TARGET/partials/$filename.ranges-$CHUNK_SIZE"
  local output="$TARGET/$filename"
  local jobs_file="$temporary_dir/jobs.tsv"
  local chunk_size="$CHUNK_SIZE"
  local index=0
  local start=0
  local lfs_response
  local download_url

  mkdir -p "$temporary_dir"
  if [[ -f "$output" ]] && [[ "$(stat --format=%s "$output")" -eq "$size" ]] && [[ "$(sha256sum "$output" | awk '{print $1}')" == "$digest" ]]; then
    printf 'verified %s\n' "$filename"
    return
  fi
  lfs_response="$temporary_dir/lfs-batch.json"
  curl --fail --silent --show-error --http1.1 --location --connect-timeout 10 --max-time 90 --retry 5 --retry-all-errors \
    --noproxy huggingface.co --resolve "huggingface.co:443:$HUB_IP" \
    -H 'Content-Type: application/vnd.git-lfs+json' -H 'Accept: application/vnd.git-lfs+json' \
    --data "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$digest\",\"size\":$size}]}" \
    "$MIRROR/datasets/$REPO.git/info/lfs/objects/batch" > "$lfs_response"
  download_url="$(python -c "import json; print(json.load(open('$lfs_response'))['objects'][0]['actions']['download']['href'])")"

  : > "$jobs_file"
  while [[ "$start" -lt "$size" ]]; do
    local end=$(( start + chunk_size - 1 ))
    if [[ "$end" -ge "$size" ]]; then end=$(( size - 1 )); fi
    printf '%06d\t%s\t%s\n' "$index" "$start" "$end" >> "$jobs_file"
    start=$(( end + 1 ))
    index=$(( index + 1 ))
  done

  export MIRROR REPO REVISION TARGET filename temporary_dir
  local active=0
  while IFS=$'\t' read -r range_index range_start range_end; do
    (
      range_file="$temporary_dir/$range_index.part"
      expected=$(( range_end - range_start + 1 ))
      received=0
      if [[ -f "$range_file" ]]; then
        received="$(stat --format=%s "$range_file")"
      fi
      if [[ "$received" -eq "$expected" ]]; then
        exit 0
      fi
      if [[ "$received" -gt "$expected" ]]; then
        echo "range size exceeds expected size for $filename:$range_index" >&2
        exit 2
      fi
      # A cancelled or flaky HTTP transfer can leave an undersized partial file.
      # Restart the requested range rather than appending a response that may no
      # longer honor the original byte offset.
      if [[ "$received" -gt 0 && "$received" -lt "$expected" ]]; then
        : > "$range_file"
        received=0
      fi
      resume_start=$(( range_start + received ))
      curl_args=(--fail --silent --show-error --http1.1 --location --retry 5 --retry-all-errors \
        --range "$resume_start-$range_end" --output "$range_file")
      if [[ -n "$CDN_IP" ]]; then
        curl_args+=(--connect-timeout 10 --noproxy us.aws.cdn.hf.co --resolve "us.aws.cdn.hf.co:443:$CDN_IP")
      fi
      curl "${curl_args[@]}" "$download_url"
      [[ "$(stat --format=%s "$range_file")" -eq "$expected" ]] || { echo "range size mismatch for $filename:$range_index" >&2; exit 2; }
    ) &
    active=$(( active + 1 ))
    if [[ "$active" -ge "$WORKERS" ]]; then
      wait -n
      active=$(( active - 1 ))
    fi
  done < "$jobs_file"
  while [[ "$active" -gt 0 ]]; do
    wait -n
    active=$(( active - 1 ))
  done

  output_tmp="$output.incomplete"
  : > "$output_tmp"
  while IFS=$'\t' read -r range_index _; do
    cat "$temporary_dir/$range_index.part" >> "$output_tmp"
  done < "$jobs_file"
  [[ "$(stat --format=%s "$output_tmp")" -eq "$size" ]] || { echo "assembled size mismatch for $filename" >&2; exit 2; }
  [[ "$(sha256sum "$output_tmp" | awk '{print $1}')" == "$digest" ]] || { echo "SHA-256 mismatch for $filename" >&2; exit 2; }
  mv "$output_tmp" "$output"
  printf 'verified %s\n' "$filename"
}

for metadata in README.md dataset.json keypoints.json summary.json; do
  download_small "$metadata"
done
download_large train-00000.parquet 600037696 96552cba373ebd92b0d2c1d724202af41447b1c8ed1923f337fb49c1d51bac46
download_large train-00001.parquet 273337158 e142de183819881845954b0a03192dd2352baea7698058bcab58e8c40c93c041
download_large valid-00000.parquet 101291365 72233f3b1316afbd0ee713cc2721d0565d472fc33f6890004abe7ca6b47fdfe7
download_large test-00000.parquet 200644661 5a59200308c92aea9103e43374d35668c510346e338c7fc0ce9c8f0f47ff30f2

#!/usr/bin/env python3
"""Fetch portable HaGRIDv2 label columns from hf-mirror with HTTP ranges.

The public Parquet shards embed JPEG bytes. On a host that already has the
matching HaGRIDv2 512px source images, downloading those bytes again is wasteful.
This tool fetches only the pinned Parquet footer and the columns needed for
training (`source_image`, `width`, `height`, `instances`) and verifies every
source image exists locally before emitting JSONL records.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from palm_detector.dataset_contract import sha256_file

REPO = "zhouxzh/portable-hagridv2-mediapipe-hand"
REVISION = "e343025bd4fd37719dbc269201d6b24124f0c432"
MIRROR = "https://hf-mirror.com"
FOOTER_WINDOW = 64 * 1024
RANGE_CHUNK_BYTES = 1024 * 1024
SPLITS = {
    "train": (
        ("train-00000.parquet", 600037696, "96552cba373ebd92b0d2c1d724202af41447b1c8ed1923f337fb49c1d51bac46", 5000),
        ("train-00001.parquet", 273337158, "e142de183819881845954b0a03192dd2352baea7698058bcab58e8c40c93c041", 2246),
    ),
    "val": (("valid-00000.parquet", 101291365, "72233f3b1316afbd0ee713cc2721d0565d472fc33f6890004abe7ca6b47fdfe7", 845),),
    "test": (("test-00000.parquet", 200644661, "5a59200308c92aea9103e43374d35668c510346e338c7fc0ce9c8f0f47ff30f2", 1663),),
}
COLUMNS = ("source_image", "width", "height", "instances")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=tuple(SPLITS))
    parser.add_argument("--keep-sparse", action="store_true")
    parser.add_argument("--allow-local-source-images", action="store_true")
    parser.add_argument("--cdn-ip", default="", help="Optional measured us.aws.cdn.hf.co IPv4 address for direct signed-range requests.")
    parser.add_argument("--range-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume only previously incomplete range-cache downloads in output-dir.")
    return parser.parse_args()


def _download_url(filename: str, lfs_sha256: str, size: int) -> str:
    response = subprocess.run(
        [
            "curl", "--fail", "--silent", "--show-error", "--location", "--retry", "5", "--retry-all-errors",
            "-H", "Content-Type: application/vnd.git-lfs+json",
            "-H", "Accept: application/vnd.git-lfs+json",
            "--data", json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": lfs_sha256, "size": size}]}),
            f"{MIRROR}/datasets/{REPO}.git/info/lfs/objects/batch",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(response.stdout)
    return str(payload["objects"][0]["actions"]["download"]["href"])


def _download_range(url: str, start: int, end: int, destination: Path, cdn_ip: str, range_workers: int) -> None:
    expected = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected:
        return
    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    if range_workers <= 0:
        raise ValueError("range workers must be positive")
    ranges: list[tuple[Path, int, int]] = []
    position = start
    while position <= end:
        stop = min(end, position + RANGE_CHUNK_BYTES - 1)
        ranges.append((parts_dir / f"{len(ranges):06d}-{position}-{stop}.bin", position, stop))
        position = stop + 1

    def fetch(part: Path, part_start: int, part_end: int) -> Path:
        expected_part = part_end - part_start + 1
        if part.is_file() and part.stat().st_size == expected_part:
            return part
        temporary_part = part.with_suffix(".incoming")
        temporary_part.unlink(missing_ok=True)
        command = [
            "curl", "--fail", "--silent", "--show-error", "--http1.1", "--location",
            "--retry", "8", "--retry-all-errors", "--range", f"{part_start}-{part_end}",
            "--output", str(temporary_part),
        ]
        if cdn_ip:
            command.extend(("--connect-timeout", "10", "--noproxy", "us.aws.cdn.hf.co", "--resolve", f"us.aws.cdn.hf.co:443:{cdn_ip}"))
        command.append(url)
        subprocess.run(command, check=True)
        if temporary_part.stat().st_size != expected_part:
            raise ValueError(f"range size mismatch for [{part_start}, {part_end}]")
        os.replace(temporary_part, part)
        return part

    with ThreadPoolExecutor(max_workers=range_workers) as executor:
        parts = list(executor.map(lambda item: fetch(*item), ranges))
    temporary = destination.with_suffix(destination.suffix + ".incoming")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as handle:
        for part in parts:
            handle.write(part.read_bytes())
    if temporary.stat().st_size != expected:
        raise ValueError(f"assembled range size mismatch for {url} [{start}, {end}]")
    os.replace(temporary, destination)


def _footer_metadata(url: str, size: int, cache_dir: Path, stem: str, cdn_ip: str, range_workers: int) -> tuple[pq.FileMetaData, Path]:
    footer_start = max(0, size - FOOTER_WINDOW)
    footer = cache_dir / f"{stem}.footer.bin"
    _download_range(url, footer_start, size - 1, footer, cdn_ip, range_workers)
    data = footer.read_bytes()
    if data[-4:] != b"PAR1":
        raise ValueError(f"{stem}: Parquet footer magic is missing")
    metadata_size = struct.unpack("<I", data[-8:-4])[0]
    metadata_start = size - metadata_size - 8
    if metadata_start < footer_start:
        raise ValueError(f"{stem}: footer window is too small")
    sparse = cache_dir / f"{stem}.sparse.parquet"
    with sparse.open("wb") as handle:
        handle.truncate(size)
        handle.seek(0)
        handle.write(b"PAR1")
        handle.seek(metadata_start)
        handle.write(data[metadata_start - footer_start :])
    return pq.read_metadata(sparse), sparse


def _column_start(column) -> int:
    return column.dictionary_page_offset if column.dictionary_page_offset is not None else column.data_page_offset


def _write_chunk(sparse: Path, start: int, source: Path) -> None:
    with source.open("rb") as reader, sparse.open("r+b") as writer:
        writer.seek(start)
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)


def _fetch_shard(
    *,
    filename: str,
    size: int,
    lfs_sha256: str,
    expected_rows: int,
    split: str,
    source_root: Path,
    output_dir: Path,
    keep_sparse: bool,
    allow_local_source_images: bool,
    cdn_ip: str,
    range_workers: int,
) -> dict[str, object]:
    cache_dir = output_dir / "range-cache"
    stem = filename.removesuffix(".parquet")
    url = _download_url(filename, lfs_sha256, size)
    metadata, sparse = _footer_metadata(url, size, cache_dir, stem, cdn_ip, range_workers)
    if metadata.num_rows != expected_rows:
        raise ValueError(f"{filename}: expected {expected_rows} rows, got {metadata.num_rows}")
    arrow_names = set(metadata.schema.to_arrow_schema().names)
    if not set(COLUMNS).issubset(arrow_names):
        raise ValueError(f"{filename}: missing columns {sorted(set(COLUMNS) - arrow_names)}")
    ranges: list[tuple[int, int, str]] = []
    group = metadata.row_group(0)
    for index in range(group.num_columns):
        column = group.column(index)
        root_name = column.path_in_schema.split(".", 1)[0]
        if root_name in COLUMNS:
            start = _column_start(column)
            ranges.append((start, start + column.total_compressed_size - 1, root_name))
    for start, end, name in ranges:
        chunk = cache_dir / f"{stem}.{name}.{start}-{end}.bin"
        _download_range(url, start, end, chunk, cdn_ip, range_workers)
        _write_chunk(sparse, start, chunk)
    table = pq.read_table(sparse, columns=list(COLUMNS))
    rows = table.to_pylist()
    records = []
    missing_source_images = []
    for row in rows:
        source_image = str(row["source_image"])
        source_path = source_root / source_image
        if not source_path.is_file():
            missing_source_images.append(source_image)
            continue
        records.append(
            {
                "split": split,
                "source_image": source_image,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "instances": json.loads(row["instances"]),
                "image_source": "local_hagridv2_512px" if allow_local_source_images else "embedded_portable_parquet_jpeg",
            }
        )
    if missing_source_images:
        raise FileNotFoundError(f"{filename}: {len(missing_source_images)} source images are absent; first: {missing_source_images[0]}")
    output_path = output_dir / "records" / f"{stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    if not keep_sparse:
        sparse.unlink(missing_ok=True)
    return {
        "filename": filename,
        "source_size": size,
        "source_lfs_sha256_declared": lfs_sha256,
        "full_lfs_sha256_verified": False,
        "rows": len(records),
        "records": str(output_path),
        "records_sha256": sha256_file(output_path),
        "downloaded_column_bytes": sum(end - start + 1 for start, end, _ in ranges),
        "local_source_images": allow_local_source_images,
    }


def main() -> int:
    args = parse_args()
    if not args.source_image_root.is_dir():
        raise NotADirectoryError(args.source_image_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if not args.allow_local_source_images:
        raise ValueError("this metadata-only workflow requires --allow-local-source-images")
    manifest = {
        "repo_id": REPO,
        "mirror": MIRROR,
        "revision": REVISION,
        "source_image_root": str(args.source_image_root),
        "image_provenance": "local HaGRIDv2 512px images matched by source_image path and dimensions; portable Parquet JPEG bytes are not used",
        "cdn_ip": args.cdn_ip or None,
        "range_workers": args.range_workers,
        "shards": [],
    }
    for split in args.splits:
        shards = SPLITS[split]
        for filename, size, digest, rows in shards:
            manifest["shards"].append(
                _fetch_shard(
                    filename=filename,
                    size=size,
                    lfs_sha256=digest,
                    expected_rows=rows,
                    split=split,
                    source_root=args.source_image_root,
                    output_dir=args.output_dir,
                    keep_sparse=args.keep_sparse,
                    allow_local_source_images=args.allow_local_source_images,
                    cdn_ip=args.cdn_ip,
                    range_workers=args.range_workers,
                )
            )
    manifest_path = args.output_dir / "metadata-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

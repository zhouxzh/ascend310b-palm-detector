#!/usr/bin/env python3
"""Resumably fetch pinned NanoDet Model Zoo checkpoints and record hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/nanodet-sweep-weights-v001.yml"
UPSTREAM_COMMIT = "d3fb34fa91d6020f273d6d063bf324dcd97bac12"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fetch_one(variant: str, entry: dict) -> tuple[str, str]:
    checkpoint = ROOT / entry["checkpoint"]
    manifest_path = ROOT / entry["manifest"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = None
    if checkpoint.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") == digest(checkpoint) and manifest.get("bytes") == checkpoint.stat().st_size:
            return variant, "verified-existing"

    source_url = entry.get("source_url")
    if not source_url:
        return variant, "paused-no-pinned-official-checkpoint"
    partial = checkpoint.with_suffix(checkpoint.suffix + ".partial")
    if not checkpoint.is_file():
        match = re.search(r"/d/([A-Za-z0-9_-]+)", source_url)
        if match:
            url = f"https://drive.usercontent.google.com/download?id={match.group(1)}&export=download&confirm=t"
        else:
            url = source_url
        with partial.with_suffix(partial.suffix + ".log").open("wb") as log:
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "20", "--retry-all-errors",
                "--retry-delay", "5", "--connect-timeout", "20", "--max-time", "7200",
                "--continue-at", "-", "--output", str(partial), url,
            ], check=True, stdout=log, stderr=subprocess.STDOUT)
        os.replace(partial, checkpoint)

    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
        checkpoint.unlink(missing_ok=True)
        raise ValueError(f"{variant}: downloaded object is not a NanoDet state_dict checkpoint")
    record = {
        "model_name": variant,
        "upstream_url": source_url,
        "upstream_revision_or_release": f"NanoDet v1.0.0 model zoo; source config at {UPSTREAM_COMMIT}",
        "license": "Apache-2.0",
        "local_path": str(checkpoint.relative_to(ROOT)),
        "bytes": checkpoint.stat().st_size,
        "sha256": digest(checkpoint),
        "verified_format": "PyTorch checkpoint with non-empty state_dict",
    }
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return variant, "downloaded-and-checksummed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    entries = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["variants"]
    if args.only is not None:
        unknown = set(args.only) - set(entries)
        if unknown:
            parser.error(f"unknown variants: {sorted(unknown)}")
        entries = {key: value for key, value in entries.items() if key in args.only}
    pending = {
        key: value for key, value in entries.items()
        if value.get("source_url") and value.get("checkpoint") and value.get("manifest")
    }
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, name, entry): name for name, entry in pending.items()}
        for future in as_completed(futures):
            try:
                name, status = future.result()
                print(f"{name}: {status}", flush=True)
            except Exception as exc:
                failures.append((futures[future], str(exc)))
                print(f"{futures[future]}: FAILED: {exc}", flush=True)
    for name, entry in entries.items():
        if name not in pending:
            print(f"{name}: paused; {entry.get('source_url') or 'no pinned official checkpoint URL'}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} pretrained checkpoints failed; see messages above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

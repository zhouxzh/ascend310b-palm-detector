#!/usr/bin/env python3
"""Download and checksum the exact LCNet weights used by the PicoDet sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/picodet-sweep-v001.yml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/picodet-v001/pretrained")
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    weights = {}
    for weight_id, item in matrix["pretrained"].items():
        target = args.output / item["filename"]
        if not target.is_file() or target.stat().st_size == 0:
            temporary = target.with_suffix(target.suffix + ".incoming")
            with urllib.request.urlopen(item["url"], timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            os.replace(temporary, target)
        weights[weight_id] = {"url": item["url"], "path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)}
    report = args.output / "weights.json"
    report.write_text(json.dumps({"schema_version": 1, "weights": weights}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download the pinned palm/reject ROI classification dataset for local training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from datasets import DatasetDict, Image, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "zhouxzh/portable-hagridv2-mediapipe-hand"
CONFIG_NAME = "palm-reject-roi"
REVISION = "fbe64737c7894abf5ab94df7dc8967144713f96f"
EXPECTED_SPLITS = {"train": 6747, "valid": 581, "test": 1556}
EXPECTED_LABELS = {0: "palm", 1: "reject"}
MIRROR = "https://hf-mirror.com"
PARQUET_FILES = {
    "train": (
        ("train-00000.parquet", "028b4650be77f81bbf06db1626aceba0e3c5de7fe82bedb51ebcb9a7ff099194"),
        ("train-00001.parquet", "d39d191793eee75c6d29acc267c5d985ac2f8a91c713627ea62b7e9534798e15"),
    ),
    "valid": (("valid-00000.parquet", "08b9cce2cc29d96d9801dc69ce28ef40468e2cb76e19975b5641362828b20644"),),
    "test": (("test-00000.parquet", "30efe8f66d41573e9c8f521c3e3593ced9b21bd3835b294461867fb4283a68ec"),),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(dataset: DatasetDict) -> dict[str, dict[str, int]]:
    if set(dataset) != set(EXPECTED_SPLITS):
        raise ValueError(f"unexpected splits: {sorted(dataset)}")
    split_counts: dict[str, dict[str, int]] = {}
    for split, expected_rows in EXPECTED_SPLITS.items():
        table = dataset[split]
        if len(table) != expected_rows:
            raise ValueError(f"{split}: expected {expected_rows} rows, found {len(table)}")
        required = {"image", "sample_id", "label", "label_id", "subject_id", "session_id"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{split}: missing fields {sorted(missing)}")
        label_ids = table["label_id"]
        labels = table["label"]
        if set(label_ids) - set(EXPECTED_LABELS):
            raise ValueError(f"{split}: unexpected label IDs {sorted(set(label_ids))}")
        for label_id, label in zip(label_ids, labels, strict=True):
            if EXPECTED_LABELS[label_id] != label:
                raise ValueError(f"{split}: label {label!r} does not match ID {label_id}")
        split_counts[split] = {
            name: int(sum(label_id == index for label_id in label_ids))
            for index, name in EXPECTED_LABELS.items()
        }
    return split_counts


def _download_file(*, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and _file_sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incoming")
    temporary.unlink(missing_ok=True)
    relative_path = f"palm-reject-roi/{destination.name}"
    url = f"{MIRROR}/datasets/{DATASET_ID}/resolve/{REVISION}/{relative_path}"
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors", "--output", str(temporary), url],
        check=True,
    )
    actual_sha256 = _file_sha256(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {destination.name}: {actual_sha256}")
    os.replace(temporary, destination)


def _download_snapshot(cache_dir: Path) -> dict[str, list[str]]:
    snapshot = cache_dir / "palm-reject-roi" / REVISION
    data_files: dict[str, list[str]] = {}
    for split, files in PARQUET_FILES.items():
        data_files[split] = []
        for filename, expected_sha256 in files:
            destination = snapshot / filename
            _download_file(destination=destination, expected_sha256=expected_sha256)
            data_files[split].append(str(destination))
    return data_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/palm-reject-roi-v005")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf-cache")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing local dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    data_files = _download_snapshot(args.cache_dir)
    dataset = load_dataset("parquet", data_files=data_files, cache_dir=str(args.cache_dir / "arrow-cache"))
    dataset = dataset.cast_column("image", Image())
    class_counts = _validate(dataset)
    with tempfile.TemporaryDirectory(prefix=".palm-reject-roi-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        dataset.save_to_disk(str(staging))
        provenance = {
            "dataset_id": DATASET_ID,
            "config": CONFIG_NAME,
            "revision": REVISION,
            "class_labels": EXPECTED_LABELS,
            "split_rows": EXPECTED_SPLITS,
            "class_counts": class_counts,
            "format": "datasets.save_to_disk",
            "download_mirror": MIRROR,
            "parquet_files": PARQUET_FILES,
        }
        provenance_path = staging / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance["provenance_sha256"] = _file_sha256(provenance_path)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

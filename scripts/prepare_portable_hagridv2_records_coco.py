#!/usr/bin/env python3
"""Create single-palm COCO from pinned portable HaGRIDv2 JSONL label records.

Records originate from the `zhouxzh/portable-hagridv2-mediapipe-hand` pinned
revision. Images are resolved from a local HaGRIDv2 512px tree through each
record's `source_image` field, because the public Parquet embeds a second JPEG
encoding of the same source images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import cv2

from palm_detector.dataset_contract import sha256_file

SPLIT_SHARDS = {
    "train": ("train-00000.jsonl", "train-00001.jsonl"),
    "val": ("valid-00000.jsonl",),
    "test": ("test-00000.jsonl",),
}
EXPECTED_ROWS = {"train": 7246, "val": 845, "test": 1663}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(source_image: str) -> str:
    source = Path(source_image)
    return "_".join(source.with_suffix("").parts) + source.suffix.lower()


def _iter_records(paths: tuple[str, ...], records_dir: Path) -> Iterator[dict[str, Any]]:
    for filename in paths:
        path = records_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                yield record


def _bboxes(record: dict[str, Any], width: int, height: int) -> list[list[float]]:
    bboxes = []
    instances = record.get("instances")
    if not isinstance(instances, list):
        raise ValueError("instances must be a list")
    for instance in instances:
        mediapipe = instance.get("mediapipe") if isinstance(instance, dict) else None
        box = mediapipe.get("palm_bbox_xyxy") if isinstance(mediapipe, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(value, (int, float)) for value in box)):
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            continue
        x, y = x1 * width, y1 * height
        box_width, box_height = (x2 - x1) * width, (y2 - y1) * height
        bboxes.append([x, y, box_width, box_height])
    return bboxes


def main() -> int:
    args = parse_args()
    if not args.metadata_manifest.is_file() or not args.records_dir.is_dir() or not args.source_image_root.is_dir():
        raise FileNotFoundError("metadata manifest, records directory, and source image root must exist")
    metadata = json.loads(args.metadata_manifest.read_text(encoding="utf-8"))
    if metadata.get("image_provenance") != "local HaGRIDv2 512px images matched by source_image path and dimensions; portable Parquet JPEG bytes are not used":
        raise ValueError("metadata manifest is not approved for local-source-image conversion")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory exists and is non-empty: {args.output_dir}")
    image_dir = args.output_dir / "images"
    annotation_dir = args.output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    image_id = 1
    annotation_id = 1
    image_hashes: dict[str, tuple[str, str]] = {}
    source_names: dict[str, str] = {}
    conversion_manifest: dict[str, Any] = {
        "metadata_manifest": {"path": str(args.metadata_manifest), "sha256": sha256_file(args.metadata_manifest)},
        "source_image_root": str(args.source_image_root),
        "image_provenance": "local HaGRIDv2 512px source images; path and decoded dimensions verified against pinned portable records",
        "splits": {},
    }
    for split, shards in SPLIT_SHARDS.items():
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        for record in _iter_records(shards, args.records_dir):
            counters["source_rows"] += 1
            if record.get("split") != split:
                raise ValueError(f"{split}: record declares split {record.get('split')!r}")
            source_image = str(record.get("source_image", ""))
            source_path = args.source_image_root / source_image
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            image = cv2.imread(str(source_path))
            if image is None:
                raise ValueError(f"cannot decode {source_path}")
            height, width = image.shape[:2]
            if (width, height) != (int(record["width"]), int(record["height"])):
                raise ValueError(f"dimension mismatch for {source_image}: local {(width, height)}, record {(record['width'], record['height'])}")
            bboxes = _bboxes(record, width, height)
            if not bboxes:
                counters["rows_without_valid_palm"] += 1
                continue
            digest = sha256_file(source_path)
            if digest in image_hashes:
                counters["duplicate_content_dropped"] += 1
                continue
            if source_image in source_names:
                counters["duplicate_source_path_dropped"] += 1
                continue
            image_hashes[digest] = (split, source_image)
            source_names[source_image] = split
            file_name = f"{split}/{_safe_name(source_image)}"
            destination = image_dir / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.symlink(source_path, destination)
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "session_id": f"portable-hagridv2:{split}:{source_image}",
                    "sha256": digest,
                    "source_image": source_image,
                    "annotation_provenance": "zhouxzh/portable-hagridv2-mediapipe-hand",
                    "image_provenance": "local_hagridv2_512px",
                }
            )
            for bbox in bboxes:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "annotation_provenance": "mediapipe_0_10_14_legacy_graph_palm_detections",
                    }
                )
                annotation_id += 1
            image_id += 1
            counters["exported_images"] += 1
            counters["exported_instances"] += len(bboxes)
        if counters["source_rows"] != EXPECTED_ROWS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, got {counters['source_rows']}")
        payload = {
            "info": {
                "description": "Portable HaGRIDv2 single-class MediaPipe palm detection labels with verified local source images",
                "subject_id_available": False,
                "subject_leakage_checked": False,
                "session_leakage_checked": True,
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "palm"}],
        }
        annotation_path = annotation_dir / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        conversion_manifest["splits"][split] = {**counters, "annotations": str(annotation_path), "annotations_sha256": sha256_file(annotation_path)}
    manifest_path = args.output_dir / "conversion-manifest.json"
    manifest_path.write_text(json.dumps(conversion_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

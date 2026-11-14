#!/usr/bin/env python3
"""Convert the portable HaGRIDv2 MediaPipe Parquet release to single-palm COCO.

The public portable release has no ``user_id``. It preserves upstream split,
source image, and image-content identifiers, allowing hash/session leakage
checks, but it cannot prove subject-disjointness after conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import cv2
import pyarrow.parquet as pq

from palm_detector.dataset_contract import sha256_file

REVISION = "e343025bd4fd37719dbc269201d6b24124f0c432"
SPLIT_SHARDS = {
    "train": ("train-00000.parquet", "train-00001.parquet"),
    "val": ("valid-00000.parquet",),
    "test": ("test-00000.parquet",),
}
EXPECTED_IMAGES = {"train": 7246, "val": 845, "test": 1663}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _image_bytes(image: Any) -> bytes:
    if isinstance(image, dict) and isinstance(image.get("bytes"), bytes):
        return image["bytes"]
    if isinstance(image, bytes):
        return image
    raise ValueError(f"unsupported embedded image value: {type(image)!r}")


def _instance_bbox(instance: dict[str, Any]) -> list[float] | None:
    mediapipe = instance.get("mediapipe")
    candidates = [
        mediapipe.get("palm_bbox_xyxy") if isinstance(mediapipe, dict) else None,
        instance.get("palm_bbox_xyxy"),
    ]
    for bbox in candidates:
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                return [x1, y1, x2, y2]
    return None


def _iter_rows(shard: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    required = {"image", "image_id", "instances", "width", "height"}
    parquet = pq.ParquetFile(shard)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{shard}: missing required columns {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def _safe_id(value: Any) -> str:
    text = str(value)
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory exists and is non-empty: {args.output_dir}")
    if not args.dataset_dir.is_dir():
        raise NotADirectoryError(args.dataset_dir)
    image_dir = args.output_dir / "images"
    annotation_dir = args.output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    image_id = 1
    annotation_id = 1
    all_file_names: set[str] = set()
    image_hashes: set[str] = set()
    source_images: set[str] = set()
    manifest: dict[str, Any] = {
        "source": {
            "repo_id": "zhouxzh/portable-hagridv2-mediapipe-hand",
            "mirror": "https://hf-mirror.com",
            "revision": REVISION,
            "license": "hagridv2-public-license-with-attribution-and-conditions-reserved",
            "label_semantics": "single MediaPipe palm detection class; no handedness labels",
            "subject_id_available": False,
        },
        "shards": {},
        "splits": {},
    }
    for split, shard_names in SPLIT_SHARDS.items():
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        for shard_name in shard_names:
            shard_path = args.dataset_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            manifest["shards"][shard_name] = {"bytes": shard_path.stat().st_size, "sha256": sha256_file(shard_path)}
            for row in _iter_rows(shard_path, args.batch_size):
                counters["source_rows"] += 1
                raw_image = _image_bytes(row["image"])
                decoded = cv2.imdecode(__import__("numpy").frombuffer(raw_image, dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"{shard_name}: cannot decode image {row['image_id']!r}")
                height, width = decoded.shape[:2]
                if int(row["width"]) != width or int(row["height"]) != height:
                    raise ValueError(f"{shard_name}: dimensions mismatch for image {row['image_id']!r}")
                instances = json.loads(row["instances"])
                if not isinstance(instances, list):
                    raise ValueError(f"{shard_name}: instances must be a JSON list")
                bboxes = [bbox for instance in instances if isinstance(instance, dict) for bbox in [_instance_bbox(instance)] if bbox is not None]
                if not bboxes:
                    counters["rows_without_valid_palm"] += 1
                    continue
                source_id = _safe_id(row["image_id"])
                content_hash = hashlib.sha256(raw_image).hexdigest()
                source_split = str(row.get("source_split") or row.get("split") or split)
                source_image = str(row.get("source_image") or row.get("file_name") or source_id)
                # Keep the first occurrence in train/val/test order, even across shards.
                if content_hash in image_hashes:
                    counters["duplicate_content_dropped"] += 1
                    continue
                if source_image in source_images:
                    counters["duplicate_source_path_dropped"] += 1
                    continue
                file_name = f"{split}/{source_id}.jpg"
                if file_name in all_file_names:
                    raise ValueError(f"duplicate output filename {file_name}")
                all_file_names.add(file_name)
                image_hashes.add(content_hash)
                source_images.add(source_image)
                target = image_dir / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw_image)
                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "session_id": f"portable-hagridv2:{source_split}:{source_image}",
                        "sha256": content_hash,
                        "source_image_id": str(row["image_id"]),
                        "source_split": source_split,
                        "source_gesture": row.get("source_gesture"),
                    }
                )
                for x1, y1, x2, y2 in bboxes:
                    x = x1 * width
                    y = y1 * height
                    box_width = (x2 - x1) * width
                    box_height = (y2 - y1) * height
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x, y, box_width, box_height],
                            "area": box_width * box_height,
                            "iscrowd": 0,
                            "annotation_provenance": "mediapipe_legacy_palm_detection",
                        }
                    )
                    annotation_id += 1
                image_id += 1
                counters["exported_images"] += 1
                counters["exported_instances"] += len(bboxes)
        if counters["source_rows"] != EXPECTED_IMAGES[split]:
            raise ValueError(f"{split}: expected {EXPECTED_IMAGES[split]} source rows, got {counters['source_rows']}")
        payload = {
            "info": {
                "description": "Portable HaGRIDv2 MediaPipe single-class palm detection",
                "source_revision": REVISION,
                "subject_id_available": False,
                "subject_leakage_checked": False,
                "session_leakage_checked": True,
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "palm"}],
        }
        output_path = annotation_dir / f"instances_{split}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["splits"][split] = {**counters, "annotations": str(output_path), "annotations_sha256": sha256_file(output_path)}
    manifest_path = args.output_dir / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

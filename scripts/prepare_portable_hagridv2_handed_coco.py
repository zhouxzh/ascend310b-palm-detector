#!/usr/bin/env python3
"""Build two-class COCO labels using MediaPipe Tasks handedness.

The portable release contains reviewed MediaPipe palm boxes and landmarks but
omits handedness. Match a fresh MediaPipe Tasks result to each saved 21-point
instance; send images with any unmatched hand to a review queue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from palm_detector.dataset_contract import sha256_file
from palm_detector.labels import COCO_CATEGORY_IDS
from prepare_portable_hagridv2_records_coco import (
    EXPECTED_ROWS,
    SPLIT_SHARDS,
    _iter_records,
    _safe_name,
)


def match_tasks_handedness(
    instances: list[dict[str, Any]],
    result: Any,
    *,
    min_score: float,
    max_landmark_distance: float,
) -> list[tuple[int, float, float]] | None:
    landmarks = result.hand_landmarks or []
    handedness = result.handedness or []
    if len(landmarks) != len(handedness) or len(landmarks) < len(instances):
        return None
    distances = []
    for index, instance in enumerate(instances):
        mediapipe = instance.get("mediapipe") if isinstance(instance, dict) else None
        reference = (
            mediapipe.get("full21_keypoints") if isinstance(mediapipe, dict) else None
        )
        if (
            not isinstance(reference, list)
            or len(reference) != 21
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in point
                )
                for point in reference
            )
        ):
            return None
        for detected_index, detected in enumerate(landmarks):
            if len(detected) != 21:
                continue
            distance = (
                sum(
                    math.hypot(point.x - saved[0], point.y - saved[1])
                    for point, saved in zip(detected, reference)
                )
                / 21
            )
            distances.append((distance, index, detected_index))
    assignments: dict[int, tuple[int, float, float]] = {}
    used_detections: set[int] = set()
    for distance, index, detected_index in sorted(distances):
        if (
            index in assignments
            or detected_index in used_detections
            or distance > max_landmark_distance
        ):
            continue
        categories = handedness[detected_index]
        if (
            not categories
            or categories[0].category_name not in ("Left", "Right")
            or categories[0].score < min_score
        ):
            continue
        category = 1 if categories[0].category_name == "Left" else 2
        assignments[index] = (category, float(categories[0].score), distance)
        used_detections.add(detected_index)
    if len(assignments) != len(instances):
        return None
    return [assignments[index] for index in range(len(instances))]


def _palm_box(instance: dict[str, Any], width: int, height: int) -> list[float] | None:
    mediapipe = instance.get("mediapipe")
    box = mediapipe.get("palm_bbox_xyxy") if isinstance(mediapipe, dict) else None
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in box
        )
    ):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]


def _source_ids(
    records: dict[str, list[dict[str, Any]]], annotations_root: Path
) -> dict[tuple[str, str], str | None]:
    needed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, rows in records.items():
        for row in rows:
            source = Path(row["source_image"])
            needed[(split, source.parent.as_posix())].add(source.stem)

    subjects: dict[tuple[str, str], str | None] = {}
    for (split, gesture), identifiers in sorted(needed.items()):
        source = annotations_root / split / f"{gesture}.json"
        annotations = json.loads(source.read_text(encoding="utf-8"))
        for identifier in identifiers:
            item = annotations.get(identifier)
            user_id = item.get("user_id") if isinstance(item, dict) else None
            if item is None:
                raise ValueError(
                    f"source annotation not found: {split}/{gesture}/{identifier}"
                )
            subjects[(split, f"{gesture}/{identifier}")] = (
                f"hagridv2:{user_id}" if isinstance(user_id, str) and user_id else None
            )
    return subjects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--source-annotations-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-handedness-score", type=float, default=0.8)
    parser.add_argument("--max-landmark-distance", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.min_handedness_score <= 1 or not 0 < args.max_landmark_distance < 1:
        parser.error(
            "handedness score and landmark distance thresholds must be in (0, 1]"
        )
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.model_path.is_file():
        parser.error(f"MediaPipe Tasks model is missing: {args.model_path}")
    calibration = json.loads(args.calibration_record.read_text(encoding="utf-8"))
    if (
        calibration.get("mapping")
        != "mediapipe_tasks_Left=anatomical_left;Right=anatomical_right"
    ):
        raise ValueError(
            "calibration does not approve the anatomical left/right mapping"
        )

    metadata = json.loads(args.metadata_manifest.read_text(encoding="utf-8"))
    if (
        metadata.get("image_provenance")
        != "local HaGRIDv2 512px images matched by source_image path and dimensions; portable Parquet JPEG bytes are not used"
    ):
        raise ValueError(
            "metadata manifest does not describe verified local source images"
        )
    expected_shards = {
        item["filename"].replace(".parquet", ".jsonl"): item
        for item in metadata["shards"]
    }
    for shard_names in SPLIT_SHARDS.values():
        for filename in shard_names:
            if (
                filename not in expected_shards
                or sha256_file(args.records_dir / filename)
                != expected_shards[filename]["records_sha256"]
            ):
                raise ValueError(
                    f"record SHA-256 does not match metadata manifest: {filename}"
                )

    records = {
        split: list(_iter_records(shards, args.records_dir))
        for split, shards in SPLIT_SHARDS.items()
    }
    for split, rows in records.items():
        if len(rows) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_ROWS[split]} rows, got {len(rows)}"
            )
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"{split}: record declares another split")
    subjects = _source_ids(records, args.source_annotations_root)

    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    images_root = args.output_dir / "images"
    annotations_root = args.output_dir / "annotations"
    images_root.mkdir(parents=True)
    annotations_root.mkdir()
    seen_hashes: set[str] = set()
    seen_sources: set[str] = set()
    review: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "label_source": "MediaPipe Tasks handedness matched to saved MediaPipe legacy full21_keypoints",
        "class_mapping_calibration": "mediapipe_tasks_Left=anatomical_left;Right=anatomical_right",
        "mediapipe_tasks_model": {
            "path": str(args.model_path),
            "sha256": sha256_file(args.model_path),
        },
        "calibration_record": {
            "path": str(args.calibration_record),
            "sha256": sha256_file(args.calibration_record),
        },
        "min_handedness_score": args.min_handedness_score,
        "max_landmark_distance": args.max_landmark_distance,
        "metadata_manifest": {
            "path": str(args.metadata_manifest),
            "sha256": sha256_file(args.metadata_manifest),
        },
        "source_image_root": str(args.source_image_root),
        "source_annotations_root": str(args.source_annotations_root),
        "splits": {},
    }
    image_id = annotation_id = 1
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for split, rows in records.items():
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            counts: Counter[str] = Counter(source_rows=len(rows))
            for row in rows:
                source_name = str(row["source_image"])
                source = Path(source_name)
                if (
                    source.is_absolute()
                    or ".." in source.parts
                    or len(source.parts) != 2
                ):
                    raise ValueError(f"invalid source image path: {source_name}")
                subject_id = subjects[(split, source.with_suffix("").as_posix())]
                if subject_id is None:
                    counts["images_without_source_subject_id"] += 1
                    review.append(
                        {
                            "split": split,
                            "source_image": source_name,
                            "reason": "source user_id is missing",
                        }
                    )
                    continue
                image_path = args.source_image_root / source
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"cannot decode {image_path}")
                height, width = image.shape[:2]
                if (width, height) != (int(row["width"]), int(row["height"])):
                    raise ValueError(f"dimension mismatch for {source_name}")
                instances = row.get("instances")
                if not isinstance(instances, list) or not instances:
                    counts["images_without_instances"] += 1
                    continue
                boxes = [
                    (
                        _palm_box(instance, width, height)
                        if isinstance(instance, dict)
                        else None
                    )
                    for instance in instances
                ]
                if any(box is None for box in boxes):
                    counts["images_with_invalid_palm_box"] += 1
                    review.append(
                        {
                            "split": split,
                            "source_image": source_name,
                            "reason": "invalid MediaPipe palm box",
                        }
                    )
                    continue
                digest = sha256_file(image_path)
                if digest in seen_hashes:
                    counts["duplicate_content_dropped"] += 1
                    continue
                if source_name in seen_sources:
                    counts["duplicate_source_path_dropped"] += 1
                    continue
                result = landmarker.detect(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    )
                )
                labels = match_tasks_handedness(
                    instances,
                    result,
                    min_score=args.min_handedness_score,
                    max_landmark_distance=args.max_landmark_distance,
                )
                if labels is None:
                    counts["images_needing_handedness_review"] += 1
                    review.append(
                        {
                            "split": split,
                            "source_image": source_name,
                            "reason": "MediaPipe Tasks handedness could not be matched to every saved hand",
                        }
                    )
                    continue
                seen_hashes.add(digest)
                seen_sources.add(source_name)
                file_name = f"{split}/{_safe_name(source_name)}"
                destination = images_root / file_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(image_path.resolve(), destination)
                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "subject_id": subject_id,
                        "session_id": f"hagridv2:{source.stem}",
                        "sha256": digest,
                        "source_image": source_name,
                    }
                )
                for (category, score, distance), box in zip(labels, boxes):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category,
                            "bbox": box,
                            "area": box[2] * box[3],
                            "iscrowd": 0,
                            "annotation_provenance": "mediapipe_legacy_palm_bbox_and_tasks_handedness",
                            "handedness_score": score,
                            "landmark_match_distance": distance,
                        }
                    )
                    counts[COCO_CATEGORY_IDS[category]] += 1
                    annotation_id += 1
                image_id += 1
                counts["exported_images"] += 1
            output = annotations_root / f"instances_{split}.json"
            output.write_text(
                json.dumps(
                    {
                        "info": {
                            "description": "Portable HaGRIDv2 MediaPipe palm boxes with matched Tasks anatomical handedness"
                        },
                        "licenses": [],
                        "images": images,
                        "annotations": annotations,
                        "categories": [
                            {"id": category_id, "name": name}
                            for category_id, name in COCO_CATEGORY_IDS.items()
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest["splits"][split] = {
                **counts,
                "annotation_path": str(output),
                "annotation_sha256": sha256_file(output),
            }
    (args.output_dir / "handedness-review.json").write_text(
        json.dumps(review, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "conversion-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

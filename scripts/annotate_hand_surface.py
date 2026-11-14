#!/usr/bin/env python3
"""Add provisional palm/back/side labels to a two-class COCO release.

The source detection annotations stay unchanged. Surface labels are inferred
from MediaPipe Tasks world landmarks and require visual review before training
a surface classifier or using them as palmprint admission ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from palm_detector.dataset_contract import sha256_file
from palm_detector.labels import validate_categories
from prepare_portable_hagridv2_records_coco import SPLIT_SHARDS, _iter_records


def surface_from_world_landmarks(
    points: list[Any], category_id: int, face_threshold: float, side_threshold: float
) -> tuple[str, float | None]:
    if len(points) != 21 or category_id not in (1, 2):
        return "unknown", None
    wrist, index, pinky = points[0], points[5], points[17]
    a = (index.x - wrist.x, index.y - wrist.y, index.z - wrist.z)
    b = (pinky.x - wrist.x, pinky.y - wrist.y, pinky.z - wrist.z)
    normal = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    length = math.sqrt(sum(value * value for value in normal))
    if length < 1e-10 or not math.isfinite(length):
        return "unknown", None
    # Positive faces the camera for anatomical left, negative for right.
    facing = normal[2] / length * (1 if category_id == 1 else -1)
    if facing >= face_threshold:
        return "palm", facing
    if facing <= -face_threshold:
        return "back", facing
    if abs(facing) <= side_threshold:
        return "side", facing
    return "unknown", facing


def _match_instances(
    annotations: list[dict[str, Any]], instances: list[dict[str, Any]],
    image: dict[str, Any], result: Any, max_landmark_distance: float,
) -> list[int | None]:
    if len(result.hand_landmarks or []) != len(result.hand_world_landmarks or []):
        return [None] * len(annotations)
    width, height = image["width"], image["height"]
    candidates = []
    for annotation_index, annotation in enumerate(annotations):
        for instance_index, instance in enumerate(instances):
            mediapipe = instance.get("mediapipe") if isinstance(instance, dict) else None
            box = mediapipe.get("palm_bbox_xyxy") if isinstance(mediapipe, dict) else None
            points = mediapipe.get("full21_keypoints") if isinstance(mediapipe, dict) else None
            if not isinstance(box, list) or len(box) != 4 or not isinstance(points, list) or len(points) != 21:
                continue
            saved_box = [box[0] * width, box[1] * height, (box[2] - box[0]) * width, (box[3] - box[1]) * height]
            if max(abs(left - right) for left, right in zip(annotation["bbox"], saved_box)) > 0.01:
                continue
            for detected_index, detected in enumerate(result.hand_landmarks or []):
                if len(detected) != 21:
                    continue
                distance = sum(
                    math.hypot(point.x - saved[0], point.y - saved[1])
                    for point, saved in zip(detected, points)
                ) / 21
                candidates.append((distance, annotation_index, instance_index, detected_index))
    matches: list[int | None] = [None] * len(annotations)
    used_instances: set[int] = set()
    used_detections: set[int] = set()
    for distance, annotation_index, instance_index, detected_index in sorted(candidates):
        if distance > max_landmark_distance:
            break
        if matches[annotation_index] is not None or instance_index in used_instances or detected_index in used_detections:
            continue
        categories = result.handedness[detected_index]
        category_id = 1 if categories and categories[0].category_name == "Left" else 2 if categories and categories[0].category_name == "Right" else None
        if category_id != annotations[annotation_index]["category_id"]:
            continue
        matches[annotation_index] = detected_index
        used_instances.add(instance_index)
        used_detections.add(detected_index)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face-threshold", type=float, default=0.7)
    parser.add_argument("--side-threshold", type=float, default=0.2)
    parser.add_argument("--max-landmark-distance", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 <= args.side_threshold < args.face_threshold <= 1:
        parser.error("require 0 <= side-threshold < face-threshold <= 1")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.model_path.is_file():
        parser.error(f"MediaPipe Tasks model is missing: {args.model_path}")

    records = {}
    for split, filenames in SPLIT_SHARDS.items():
        records[split] = {row["source_image"]: row for row in _iter_records(filenames, args.records_dir)}

    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    annotation_root = args.output_dir / "annotations"
    annotation_root.mkdir(parents=True)
    (args.output_dir / "images").symlink_to((args.source_dir / "images").resolve(), target_is_directory=True)
    review = []
    manifest = {
        "status": "provisional_auto_labels_require_visual_review",
        "source_dir": str(args.source_dir),
        "model": {"path": str(args.model_path), "sha256": sha256_file(args.model_path)},
        "method": "MediaPipe Tasks world-landmark palm normal, signed by anatomical handedness",
        "face_threshold": args.face_threshold,
        "side_threshold": args.side_threshold,
        "max_landmark_distance": args.max_landmark_distance,
        "splits": {},
    }
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for split in SPLIT_SHARDS:
            source = args.source_dir / "annotations" / f"instances_{split}.json"
            coco = json.loads(source.read_text(encoding="utf-8"))
            errors = validate_categories(coco["categories"])
            if errors:
                raise ValueError("; ".join(errors))
            annotations_by_image: dict[int, list[dict[str, Any]]] = {}
            for annotation in coco["annotations"]:
                annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
            counts: Counter[str] = Counter()
            for image in coco["images"]:
                source_name = image["source_image"]
                record = records[split].get(source_name)
                if record is None:
                    raise ValueError(f"missing portable record: {split}/{source_name}")
                path = args.source_dir / "images" / image["file_name"]
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError(f"cannot decode {path}")
                result = detector.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                ))
                annotations = annotations_by_image.get(image["id"], [])
                matches = _match_instances(annotations, record["instances"], image, result, args.max_landmark_distance)
                for annotation, detected_index in zip(annotations, matches):
                    if detected_index is None:
                        candidate, facing = "unknown", None
                        reason = "world landmarks could not be matched to this hand"
                    else:
                        candidate, facing = surface_from_world_landmarks(
                            result.hand_world_landmarks[detected_index],
                            annotation["category_id"], args.face_threshold, args.side_threshold,
                        )
                        reason = (
                            "edge-on candidate requires visual confirmation" if candidate == "side"
                            else "angle is between side and face thresholds" if candidate == "unknown"
                            else None
                        )
                    view = "unknown" if candidate == "side" else candidate
                    annotation["surface_view"] = view
                    annotation["surface_candidate_view"] = candidate
                    annotation["surface_facing"] = facing
                    annotation["surface_provenance"] = "mediapipe_tasks_world_landmark_candidate"
                    annotation["surface_reviewed"] = False
                    annotation["palmprint_usable"] = None
                    counts[view] += 1
                    if view == "unknown":
                        review.append({
                            "split": split, "image_id": image["id"],
                            "annotation_id": annotation["id"], "file_name": image["file_name"],
                            "suggested_view": candidate, "facing": facing, "reason": reason,
                        })
            output = annotation_root / f"instances_{split}.json"
            output.write_text(json.dumps(coco, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            manifest["splits"][split] = {
                "images": len(coco["images"]), "annotations": len(coco["annotations"]),
                "surface_counts": dict(counts),
                "source_annotation_sha256": sha256_file(source),
                "output_annotation_sha256": sha256_file(output),
            }
            print(f"{split}: {dict(counts)}", flush=True)
    (args.output_dir / "surface-review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["review_items"] = len(review)
    (args.output_dir / "surface-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_dir / "surface-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

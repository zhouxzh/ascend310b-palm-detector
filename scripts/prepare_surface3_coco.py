#!/usr/bin/env python3
"""Materialize the user-approved palm/back/side training release.

The source surface annotations remain immutable. ``unknown`` hands are
excluded from both supervision and metrics; images containing one are omitted
to prevent treating a visible but ambiguous hand as detector background.
``instances_train_balanced.json`` intentionally repeats side-image records for
training only. The unique files are used for audit, development, gate, and
test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palm_detector.dataset_contract import sha256_file
from palm_detector.labels import SURFACE_CATEGORY_IDS

ROOT = Path(__file__).resolve().parents[1]


def _surface_category(annotation: dict[str, Any]) -> int | None:
    view = annotation.get("surface_view")
    candidate = annotation.get("surface_candidate_view")
    if view == "palm":
        return 1
    if view == "back":
        return 2
    if candidate == "side":
        return 3
    return None


def _convert_split(source: dict[str, Any], *, allowed_images: set[int] | None, side_repeat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)

    unique_images: list[dict[str, Any]] = []
    unique_annotations: list[dict[str, Any]] = []
    accepted: dict[int, list[dict[str, Any]]] = {}
    skipped_unknown_images = 0
    skipped_non_target_images = 0
    for image in source["images"]:
        image_id = int(image["id"])
        if allowed_images is not None and image_id not in allowed_images:
            skipped_non_target_images += 1
            continue
        annotations = grouped.get(image_id, [])
        classes = [_surface_category(annotation) for annotation in annotations]
        if not annotations or any(category is None for category in classes):
            skipped_unknown_images += 1
            continue
        accepted[image_id] = []
        unique_images.append(copy.deepcopy(image))
        for annotation, category in zip(annotations, classes):
            record = copy.deepcopy(annotation)
            record["source_annotation_id"] = record["id"]
            record["category_id"] = category
            record["surface_training_class"] = SURFACE_CATEGORY_IDS[category]
            accepted[image_id].append(record)
            unique_annotations.append(record)

    unique = {
        "info": {
            "description": "User-approved palm/back/side surface labels; unknown hands excluded from supervision and metrics",
            "source_surface_contract": "surface_view palm/back; surface_candidate_view side",
        },
        "licenses": source.get("licenses", []),
        "images": unique_images,
        "annotations": unique_annotations,
        "categories": [{"id": key, "name": value} for key, value in SURFACE_CATEGORY_IDS.items()],
    }
    if side_repeat == 1:
        return unique, {
            "source_images": len(source["images"]),
            "unique_images": len(unique_images),
            "unique_annotations": len(unique_annotations),
            "unknown_images_excluded": skipped_unknown_images,
            "outside_requested_split_excluded": skipped_non_target_images,
            "class_counts": dict(Counter(SURFACE_CATEGORY_IDS[annotation["category_id"]] for annotation in unique_annotations)),
        }

    balanced_images: list[dict[str, Any]] = []
    balanced_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for image in unique_images:
        annotations = accepted[int(image["id"])]
        copies = side_repeat if any(item["category_id"] == 3 for item in annotations) else 1
        for repeat_index in range(copies):
            image_copy = copy.deepcopy(image)
            image_copy["source_image_id"] = image["id"]
            image_copy["sampling_repeat"] = repeat_index + 1
            image_copy["id"] = next_image_id
            balanced_images.append(image_copy)
            for annotation in annotations:
                annotation_copy = copy.deepcopy(annotation)
                annotation_copy["id"] = next_annotation_id
                annotation_copy["image_id"] = next_image_id
                annotation_copy["sampling_repeat"] = repeat_index + 1
                balanced_annotations.append(annotation_copy)
                next_annotation_id += 1
            next_image_id += 1
    balanced = {**unique, "images": balanced_images, "annotations": balanced_annotations}
    report = {
        "source_images": len(source["images"]),
        "unique_images": len(unique_images),
        "unique_annotations": len(unique_annotations),
        "unknown_images_excluded": skipped_unknown_images,
        "outside_requested_split_excluded": skipped_non_target_images,
        "class_counts": dict(Counter(SURFACE_CATEGORY_IDS[annotation["category_id"]] for annotation in unique_annotations)),
        "side_repeat": side_repeat,
        "balanced_images": len(balanced_images),
        "balanced_annotations": len(balanced_annotations),
        "balanced_class_counts": dict(Counter(SURFACE_CATEGORY_IDS[annotation["category_id"]] for annotation in balanced_annotations)),
    }
    return balanced, report


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-surface-v001")
    parser.add_argument("--dev-source", type=Path, default=ROOT / "artifacts/sweeps/palm-v001/validation-audited/instances_dev.json")
    parser.add_argument("--gate-source", type=Path, default=ROOT / "artifacts/sweeps/palm-v001/validation-audited/instances_gate.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/portable-hagridv2-mediapipe-hand/coco-surface3-v001")
    parser.add_argument("--side-repeat", type=int, default=32)
    args = parser.parse_args()
    if args.side_repeat < 1:
        parser.error("side-repeat must be at least 1")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    source_root = args.source_dir.resolve()
    source = {split: _load(source_root / "annotations" / f"instances_{split}.json") for split in ("train", "val", "test")}
    dev_ids = {int(item["id"]) for item in _load(args.dev_source)["images"]}
    gate_ids = {int(item["id"]) for item in _load(args.gate_source)["images"]}
    if dev_ids & gate_ids:
        raise ValueError("development and gate image IDs overlap")

    args.output_dir.mkdir(parents=True)
    os.symlink((source_root / "images").resolve(), args.output_dir / "images", target_is_directory=True)
    annotation_root = args.output_dir / "annotations"
    annotation_root.mkdir()
    train_unique, train_report = _convert_split(source["train"], allowed_images=None, side_repeat=1)
    train_balanced, balanced_report = _convert_split(source["train"], allowed_images=None, side_repeat=args.side_repeat)
    dev, dev_report = _convert_split(source["val"], allowed_images=dev_ids, side_repeat=1)
    gate, gate_report = _convert_split(source["val"], allowed_images=gate_ids, side_repeat=1)
    test, test_report = _convert_split(source["test"], allowed_images=None, side_repeat=1)
    outputs = {
        "train_unique": train_unique,
        "train_balanced": train_balanced,
        "dev": dev,
        "gate": gate,
        "test": test,
    }
    for split, payload in outputs.items():
        _write(annotation_root / f"instances_{split}.json", payload)
    manifest = {
        "schema_version": 1,
        "label_status": "user-confirmed human-validated source surface annotations",
        "classes": SURFACE_CATEGORY_IDS,
        "unknown_policy": "exclude any image containing an unknown surface from training and evaluation",
        "side_policy": "source_candidate_side is supervised as side after user confirmation",
        "side_repeat": args.side_repeat,
        "source": {
            "dir": str(source_root),
            "surface_manifest_sha256": sha256_file(source_root / "surface-manifest.json"),
            "train_sha256": sha256_file(source_root / "annotations/instances_train.json"),
            "val_sha256": sha256_file(source_root / "annotations/instances_val.json"),
            "test_sha256": sha256_file(source_root / "annotations/instances_test.json"),
            "dev_source_sha256": sha256_file(args.dev_source),
            "gate_source_sha256": sha256_file(args.gate_source),
        },
        "splits": {
            "train_unique": train_report,
            "train_balanced": balanced_report,
            "dev": dev_report,
            "gate": gate_report,
            "test": test_report,
        },
        "output_sha256": {name: sha256_file(annotation_root / f"instances_{name}.json") for name in outputs},
    }
    _write(args.output_dir / "surface3-manifest.json", manifest)
    print(args.output_dir / "surface3-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

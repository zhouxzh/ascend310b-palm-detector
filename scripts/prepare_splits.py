#!/usr/bin/env python3
"""Create disjoint train/val/test COCO annotations from one audited source."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palm_detector.dataset_contract import (
    REQUIRED_IMAGE_FIELDS,
    DatasetContractError,
    SplitSpec,
    connected_components,
    load_coco,
    sha256_file,
    validate_splits,
)
from palm_detector.labels import COCO_CATEGORY_IDS, validate_categories


def _component_counts(component: list[dict[str, Any]], annotations_by_image: dict[int, list[dict[str, Any]]]) -> Counter[int]:
    return Counter(annotation["category_id"] for image in component for annotation in annotations_by_image[image["id"]])


def _choose_split(
    component: list[dict[str, Any]],
    component_counts: Counter[int],
    allocated_images: dict[str, int],
    allocated_classes: dict[str, Counter[int]],
    target_images: dict[str, float],
    target_classes: dict[str, Counter[int]],
    names: list[str],
) -> str:
    best_name: str | None = None
    best_score: float | None = None
    for name in names:
        # Other splits are unchanged, so this delta is the global error change.
        image_target = max(target_images[name], 1.0)
        image_error = allocated_images[name] - target_images[name]
        image_score = ((image_error + len(component)) / image_target) ** 2 - (image_error / image_target) ** 2
        class_score = 0.0
        for category_id in COCO_CATEGORY_IDS:
            target = max(float(target_classes[name][category_id]), 1.0)
            class_error = allocated_classes[name][category_id] - target_classes[name][category_id]
            class_score += ((class_error + component_counts[category_id]) / target) ** 2 - (class_error / target) ** 2
        score = image_score + class_score
        if best_score is None or score < best_score:
            best_name, best_score = name, score
    assert best_name is not None
    return best_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-image-hashes", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.ratios) or abs(sum(args.ratios) - 1.0) > 1e-8:
        parser.error("--ratios must be positive and sum to 1")
    try:
        coco = load_coco(args.source_annotations)
    except DatasetContractError as exc:
        parser.error(str(exc))
    errors = validate_categories(coco["categories"])
    image_ids = set()
    for image in coco["images"]:
        missing = [field for field in REQUIRED_IMAGE_FIELDS if field not in image or image[field] in (None, "")]
        if missing:
            errors.append(f"image {image.get('id')}: missing {missing}")
            continue
        if image["id"] in image_ids:
            errors.append(f"duplicate image id {image['id']}")
        image_ids.add(image["id"])
        path = args.images_root / image["file_name"]
        if not path.is_file():
            errors.append(f"image {image['id']}: missing {path}")
        elif args.verify_image_hashes and sha256_file(path) != image["sha256"]:
            errors.append(f"image {image['id']}: sha256 does not match content")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("image_id") not in image_ids:
            errors.append(f"annotation {annotation.get('id')}: unknown image")
        if annotation.get("category_id") not in COCO_CATEGORY_IDS:
            errors.append(f"annotation {annotation.get('id')}: invalid category")
        annotations_by_image[annotation.get("image_id")].append(annotation)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    components = connected_components(coco["images"])
    if len(components) < 3:
        print("ERROR: fewer than three independent subject/session components; cannot create train/val/test")
        return 2
    names = ["train", "val", "test"]
    total_classes = Counter(annotation["category_id"] for annotation in coco["annotations"])
    target_images = {name: len(coco["images"]) * ratio for name, ratio in zip(names, args.ratios)}
    target_classes = {name: Counter({category_id: total_classes[category_id] * ratio for category_id in COCO_CATEGORY_IDS}) for name, ratio in zip(names, args.ratios)}
    random.Random(args.seed).shuffle(components)
    components.sort(key=lambda group: len(group), reverse=True)
    allocated: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in names}
    allocated_images = {name: 0 for name in names}
    allocated_classes = {name: Counter() for name in names}
    for component in components:
        counts = _component_counts(component, annotations_by_image)
        target = _choose_split(component, counts, allocated_images, allocated_classes, target_images, target_classes, names)
        allocated[target].append(component)
        allocated_images[target] += len(component)
        allocated_classes[target].update(counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_id = {annotation["image_id"]: [] for annotation in coco["annotations"]}
    for annotation in coco["annotations"]:
        annotations_by_id.setdefault(annotation["image_id"], []).append(annotation)
    outputs: list[SplitSpec] = []
    for name in names:
        images = [image for group in allocated[name] for image in group]
        selected_ids = {image["id"] for image in images}
        payload = {key: coco[key] for key in ("info", "licenses") if key in coco}
        payload.update({"images": images, "annotations": [annotation for annotation in coco["annotations"] if annotation["image_id"] in selected_ids], "categories": coco["categories"]})
        output_path = args.output_dir / f"instances_{name}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(SplitSpec(name, output_path, args.images_root))
    report = validate_splits(outputs, verify_image_hashes=args.verify_image_hashes)
    report.update({"source_annotations": str(args.source_annotations), "source_sha256": sha256_file(args.source_annotations), "seed": args.seed, "ratios": dict(zip(names, args.ratios)), "output_sha256": {spec.name: sha256_file(spec.annotation_path) for spec in outputs}})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir} and {args.report}: {'PASS' if report['valid'] else 'FAIL'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

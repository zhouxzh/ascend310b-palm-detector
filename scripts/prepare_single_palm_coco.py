#!/usr/bin/env python3
"""Materialize the audited single-palm COCO task from surface3 annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from palm_detector.labels import SINGLE_PALM_CATEGORY_IDS, SURFACE_CATEGORY_IDS, validate_categories


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-surface3-v001"
DEFAULT_OUTPUT = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-single-palm-v001"
SPLITS = {
    "train": "instances_train_unique.json",
    "dev": "instances_dev.json",
    "gate": "instances_gate.json",
    "test": "instances_test.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standard_image(image: dict) -> dict:
    required = ("id", "file_name", "height", "width", "sha256", "subject_id", "session_id", "source_image")
    missing = [key for key in required if key not in image]
    if missing:
        raise ValueError(f"image {image.get('id')} is missing required fields: {missing}")
    return {key: image[key] for key in required}


def _standard_annotation(annotation: dict) -> dict:
    required = ("id", "image_id", "bbox", "area", "iscrowd")
    missing = [key for key in required if key not in annotation]
    if missing:
        raise ValueError(f"annotation {annotation.get('id')} is missing required fields: {missing}")
    return {**{key: annotation[key] for key in required}, "category_id": 1}


def collapse_split(source_path: Path) -> tuple[dict, dict[str, int]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    errors = validate_categories(source.get("categories", []), SURFACE_CATEGORY_IDS)
    if errors:
        raise ValueError("; ".join(errors))
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in source.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    selected_images = []
    selected_annotations = []
    counts = Counter()
    for image in source.get("images", []):
        image_id = int(image["id"])
        annotations = annotations_by_image.get(image_id, [])
        category_ids = {int(item["category_id"]) for item in annotations}
        if 3 in category_ids:
            counts["images_excluded_with_side"] += 1
            continue
        retained = [item for item in annotations if int(item["category_id"]) in (1, 2)]
        if not retained:
            counts["images_excluded_without_palm_or_back"] += 1
            continue
        selected_images.append(_standard_image(image))
        selected_annotations.extend(_standard_annotation(item) for item in retained)
        counts["source_palm_annotations"] += sum(int(item["category_id"]) == 1 for item in retained)
        counts["source_back_annotations"] += sum(int(item["category_id"]) == 2 for item in retained)
    output = {
        "info": {
            "description": "Human-confirmed palm/back annotations collapsed into one palm_surface category; side-containing images excluded",
            "version": "single-palm-v001",
        },
        "licenses": source.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": [{"id": category_id, "name": name} for category_id, name in SINGLE_PALM_CATEGORY_IDS.items()],
    }
    counts["images"] = len(selected_images)
    counts["annotations"] = len(selected_annotations)
    return output, dict(counts)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".single-palm-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "images").is_dir():
        raise FileNotFoundError(f"source images are missing: {source / 'images'}")
    if output.exists():
        raise FileExistsError(f"single-palm output already exists: {output}")

    output.mkdir(parents=True)
    try:
        os.symlink("../images", output / "images")
        manifest = {
            "schema_version": 1,
            "label_status": "derived only from user-confirmed human-validated surface annotations",
            "classes": SINGLE_PALM_CATEGORY_IDS,
            "source_policy": "collapse palm and back to palm_surface; exclude every image containing side",
            "image_root": "../images",
            "splits": {},
        }
        for split, filename in SPLITS.items():
            source_path = source / "annotations" / filename
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            payload, counts = collapse_split(source_path)
            output_path = output / "annotations" / f"instances_{split}.json"
            _atomic_json(output_path, payload)
            manifest["splits"][split] = {
                "source_annotation_sha256": sha256_file(source_path),
                "output_annotation_sha256": sha256_file(output_path),
                **counts,
            }
        _atomic_json(output / "single-palm-manifest.json", manifest)
    except Exception:
        # The caller can safely remove an incomplete output directory and retry.
        raise
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

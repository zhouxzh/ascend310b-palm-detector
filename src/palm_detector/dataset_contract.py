"""Validation and leakage checks for the versioned COCO data contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .labels import SURFACE_CATEGORY_IDS, validate_categories

REQUIRED_IMAGE_FIELDS = ("id", "file_name", "width", "height", "subject_id", "session_id", "sha256")


class DatasetContractError(ValueError):
    """Raised when a dataset cannot be safely used for this task."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    annotation_path: Path
    images_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coco(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"Cannot parse COCO JSON {path}: {exc}") from exc
    for field in ("images", "annotations", "categories"):
        if not isinstance(coco.get(field), list):
            raise DatasetContractError(f"{path}: missing list field '{field}'")
    return coco


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 3)
    return {"min": at(0), "p50": at(0.5), "p95": at(0.95), "max": at(1.0)}


def validate_split(
    spec: SplitSpec,
    *,
    verify_image_hashes: bool = False,
    decode_images: bool = True,
    category_ids: dict[int, str] = SURFACE_CATEGORY_IDS,
    require_subject_id: bool = True,
    require_session_id: bool = True,
    require_all_classes: bool = True,
) -> tuple[dict[str, Any], list[str], dict[str, set[str]]]:
    """Return report, violations, and IDs used by cross-split checks."""
    coco = load_coco(spec.annotation_path)
    errors = validate_categories(coco["categories"], category_ids)
    image_ids: set[int] = set()
    image_hashes: set[str] = set()
    subject_ids: set[str] = set()
    session_ids: set[str] = set()
    images_by_id: dict[int, dict[str, Any]] = {}
    duplicate_hashes: Counter[str] = Counter()

    required_fields = ["id", "file_name", "width", "height", "sha256"]
    if require_subject_id:
        required_fields.append("subject_id")
    if require_session_id:
        required_fields.append("session_id")
    for image in coco["images"]:
        missing = [key for key in required_fields if key not in image or image[key] in (None, "")]
        if missing:
            errors.append(f"image {image.get('id', '<missing>')}: missing required fields {missing}")
            continue
        image_id = image["id"]
        if not isinstance(image_id, int) or image_id in image_ids:
            errors.append(f"duplicate or non-integer image id: {image_id!r}")
            continue
        image_ids.add(image_id)
        images_by_id[image_id] = image
        if require_subject_id:
            subject_ids.add(str(image["subject_id"]))
        if require_session_id:
            session_ids.add(str(image["session_id"]))
        declared_hash = str(image["sha256"]).lower()
        if len(declared_hash) != 64 or any(char not in "0123456789abcdef" for char in declared_hash):
            errors.append(f"image {image_id}: sha256 must be a lowercase 64-character digest")
        duplicate_hashes[declared_hash] += 1
        image_path = spec.images_root / image["file_name"]
        if not image_path.is_file() or image_path.stat().st_size == 0:
            errors.append(f"image {image_id}: missing or empty file {image_path}")
            continue
        if verify_image_hashes and sha256_file(image_path) != declared_hash:
            errors.append(f"image {image_id}: sha256 does not match file content")
        if decode_images:
            try:
                import cv2
                decoded = cv2.imread(str(image_path))
            except ImportError:
                decoded = None
                errors.append("OpenCV is required for corrupt-image checks")
            if decoded is None:
                errors.append(f"image {image_id}: OpenCV cannot decode {image_path}")
            elif decoded.shape[1] != image["width"] or decoded.shape[0] != image["height"]:
                errors.append(f"image {image_id}: annotation dimensions do not match decoded image")

    for digest, count in duplicate_hashes.items():
        if count > 1:
            errors.append(f"duplicate image sha256 in split: {digest}")

    annotation_ids: set[int] = set()
    class_counts: Counter[str] = Counter()
    boxes_by_image: Counter[int] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    for annotation in coco["annotations"]:
        annotation_id = annotation.get("id")
        if not isinstance(annotation_id, int) or annotation_id in annotation_ids:
            errors.append(f"duplicate or non-integer annotation id: {annotation_id!r}")
            continue
        annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if image_id not in images_by_id:
            errors.append(f"annotation {annotation_id}: image_id {image_id!r} does not exist")
            continue
        category_id = annotation.get("category_id")
        if category_id not in category_ids:
            errors.append(f"annotation {annotation_id}: unknown category_id {category_id!r}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            errors.append(f"annotation {annotation_id}: bbox must be numeric COCO xywh")
            continue
        x, y, width, height = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            errors.append(f"annotation {annotation_id}: bbox values must be finite")
            continue
        image = images_by_id[image_id]
        if width <= 0 or height <= 0:
            errors.append(f"annotation {annotation_id}: bbox width and height must be positive")
        if x < 0 or y < 0 or x + width > image["width"] + 1e-4 or y + height > image["height"] + 1e-4:
            errors.append(f"annotation {annotation_id}: bbox is outside image bounds")
        declared_area = annotation.get("area")
        if not isinstance(declared_area, (int, float)) or not math.isfinite(declared_area) or abs(float(declared_area) - width * height) > max(1.0, width * height * 1e-3):
            errors.append(f"annotation {annotation_id}: area must match bbox width * height")
        if annotation.get("iscrowd", 0) not in (0, False):
            errors.append(f"annotation {annotation_id}: iscrowd must be 0 for this task")
        if category_id in category_ids:
            class_counts[category_ids[category_id]] += 1
        boxes_by_image[image_id] += 1
        widths.append(width)
        heights.append(height)
        areas.append(width * height)

    if require_all_classes:
        for class_name in category_ids.values():
            if class_counts[class_name] == 0:
                errors.append(f"split has no {class_name} annotations")

    report = {
        "split": spec.name,
        "annotation_path": str(spec.annotation_path),
        "images_root": str(spec.images_root),
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "class_counts": {name: class_counts[name] for name in category_ids.values()},
        "objects_per_image": _quantiles([float(boxes_by_image[image_id]) for image_id in images_by_id]),
        "bbox_width_px": _quantiles(widths),
        "bbox_height_px": _quantiles(heights),
        "bbox_area_px2": _quantiles(areas),
        "subjects": len(subject_ids) if require_subject_id else None,
        "sessions": len(session_ids) if require_session_id else None,
        "subject_leakage_checked": require_subject_id,
        "session_leakage_checked": require_session_id,
        "errors": errors,
    }
    identities = {"image_ids": {str(value) for value in image_ids}, "hashes": image_hashes | set(duplicate_hashes), "subjects": subject_ids, "sessions": session_ids}
    return report, errors, identities


def validate_splits(
    specs: Iterable[SplitSpec],
    *,
    category_ids: dict[int, str] = SURFACE_CATEGORY_IDS,
    require_subject_id: bool = True,
    require_session_id: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    identities_by_split: dict[str, dict[str, set[str]]] = {}
    all_errors: list[str] = []
    for spec in specs:
        report, errors, identities = validate_split(
            spec,
            category_ids=category_ids,
            require_subject_id=require_subject_id,
            require_session_id=require_session_id,
            **kwargs,
        )
        reports.append(report)
        identities_by_split[spec.name] = identities
        all_errors.extend(f"{spec.name}: {error}" for error in errors)
    names = list(identities_by_split)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            checks = [("image_ids", "image IDs"), ("hashes", "image hashes")]
            if require_subject_id:
                checks.append(("subjects", "subject IDs"))
            if require_session_id:
                checks.append(("sessions", "session IDs"))
            for key, label in checks:
                overlap = identities_by_split[left_name][key] & identities_by_split[right_name][key]
                if overlap:
                    all_errors.append(f"{left_name}/{right_name}: cross-split {label} leakage ({len(overlap)} values)")
    return {"valid": not all_errors, "splits": reports, "errors": all_errors}


def connected_components(images: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group images so a subject or a session can never cross a split."""
    images = list(images)
    parent = list(range(len(images)))
    owner: dict[tuple[str, str], int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, image in enumerate(images):
        for field in ("subject_id", "session_id"):
            key = (field, str(image[field]))
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, image in enumerate(images):
        groups[find(index)].append(image)
    return list(groups.values())

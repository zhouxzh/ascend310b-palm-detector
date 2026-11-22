#!/usr/bin/env python3
"""Render ground truth and final detections for controlled visual review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from palm_detector.labels import SURFACE_CATEGORY_IDS, validate_categories

COLORS = {1: (72, 180, 72), 2: (0, 190, 255), 3: (255, 140, 0)}


def _draw(image, bbox, text, color, thickness):
    x, y, width, height = bbox
    cv2.rectangle(image, (round(x), round(y)), (round(x + width), round(y + height)), color, thickness)
    cv2.putText(image, text, (round(x), max(16, round(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, thickness, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    args = parser.parse_args()
    coco = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    errors = validate_categories(coco.get("categories", []), SURFACE_CATEGORY_IDS)
    if errors:
        raise ValueError("; ".join(errors))
    annotations = defaultdict(list)
    for item in coco["annotations"]:
        annotations[item["image_id"]].append(item)
    predictions = defaultdict(list)
    for item in json.loads(args.predictions.read_text(encoding="utf-8")):
        if item.get("score", 0.0) >= args.score_threshold:
            predictions[item["image_id"]].append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for image_info in coco["images"]:
        image = cv2.imread(str(args.images_root / image_info["file_name"]))
        if image is None:
            raise FileNotFoundError(image_info["file_name"])
        for item in annotations[image_info["id"]]:
            _draw(image, item["bbox"], f"GT {SURFACE_CATEGORY_IDS[item['category_id']]}", COLORS[item["category_id"]], 3)
        for item in predictions[image_info["id"]]:
            _draw(image, item["bbox"], f"P {SURFACE_CATEGORY_IDS[item['category_id']]} {item['score']:.2f}", COLORS[item["category_id"]], 1)
        output = args.output_dir / image_info["file_name"]
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"could not write {output}")
        rendered += 1
    print(f"wrote {rendered} prediction overlays to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

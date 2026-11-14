#!/usr/bin/env python3
"""Render deterministic 320x320 COCO annotation overlays for human review."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

from palm_detector.labels import COCO_CATEGORY_IDS, validate_categories
from palm_detector.postprocess import INPUT_HEIGHT, INPUT_WIDTH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    errors = validate_categories(coco.get("categories", []))
    if errors:
        raise ValueError("; ".join(errors))
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    images = list(coco["images"])
    random.Random(args.seed).shuffle(images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    palette = {1: (255, 140, 0), 2: (0, 190, 255)}
    written = 0
    for image_info in images:
        if written >= args.limit:
            break
        image = cv2.imread(str(args.images_root / image_info["file_name"]))
        if image is None:
            raise FileNotFoundError(image_info["file_name"])
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        for annotation in annotations_by_image[image_info["id"]]:
            category_id = annotation["category_id"]
            x, y, width, height = annotation["bbox"]
            x1, y1 = round(x * INPUT_WIDTH / original_width), round(y * INPUT_HEIGHT / original_height)
            x2, y2 = round((x + width) * INPUT_WIDTH / original_width), round((y + height) * INPUT_HEIGHT / original_height)
            color = palette[category_id]
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized, COCO_CATEGORY_IDS[category_id], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        output = args.output_dir / f"{written:04d}_image-{image_info['id']}.jpg"
        if not cv2.imwrite(str(output), resized):
            raise RuntimeError(f"failed to write {output}")
        written += 1
    print(f"wrote {written} overlays to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

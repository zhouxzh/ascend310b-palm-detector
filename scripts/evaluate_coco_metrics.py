#!/usr/bin/env python3
"""Persist standard COCO and per-class AP metrics for a prediction JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from palm_detector.dataset_contract import sha256_file
from palm_detector.labels import validate_categories

METRIC_NAMES = ("mAP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ground_truth = COCO(str(args.ground_truth))
    categories = ground_truth.dataset.get("categories", [])
    category_ids = {category["id"]: category["name"] for category in categories}
    errors = validate_categories(categories, category_ids)
    if errors:
        raise ValueError("; ".join(errors))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    detections = ground_truth.loadRes(predictions)
    evaluation = COCOeval(ground_truth, detections, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    precisions = evaluation.eval["precision"]
    per_class = {}
    for index, category_id in enumerate(sorted(category_ids)):
        ap50 = precisions[0, :, index, 0, -1]
        map_values = precisions[:, :, index, 0, -1]
        valid_ap50 = ap50[ap50 > -1]
        valid_map = map_values[map_values > -1]
        per_class[category_ids[category_id]] = {
            "AP50": float(np.mean(valid_ap50)) if valid_ap50.size else None,
            "mAP50_95": float(np.mean(valid_map)) if valid_map.size else None,
        }
    report = {
        "ground_truth": {"path": str(args.ground_truth), "sha256": sha256_file(args.ground_truth)},
        "predictions": {"path": str(args.predictions), "sha256": sha256_file(args.predictions)},
        "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, evaluation.stats[:6])},
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

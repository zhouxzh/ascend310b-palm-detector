#!/usr/bin/env python3
"""Report left/right accuracy, misses, and duplicate detections from COCO JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palm_detector.dataset_contract import load_coco, sha256_file
from palm_detector.labels import validate_categories
from palm_detector.task_metrics import task_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="COCO detection result list")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ground_truth = load_coco(args.ground_truth)
    errors = validate_categories(ground_truth["categories"])
    if errors:
        raise ValueError("; ".join(errors))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("COCO predictions must be a JSON list")
    report = task_metrics(ground_truth, predictions, score_threshold=args.score_threshold, iou_threshold=args.iou_threshold)
    report["ground_truth"] = {"path": str(args.ground_truth), "sha256": sha256_file(args.ground_truth)}
    report["predictions"] = {"path": str(args.predictions), "sha256": sha256_file(args.predictions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

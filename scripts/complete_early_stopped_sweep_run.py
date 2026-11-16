#!/usr/bin/env python3
"""Finalize a plateaued sweep run while preserving its best checkpoint evidence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


METRIC_PATTERN = re.compile(r"Val_metrics: .*?'mAP': np\.float64\(([^)]+)\)")
EPOCH_PATTERN = re.compile(r"Train\|Epoch(\d+)/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-delta", type=float, default=0.001)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    train_log = run_dir / "train.log"
    best_checkpoint = run_dir / "model_best" / "model_best.ckpt"
    best_weights = run_dir / "model_best" / "nanodet_model_best.pth"
    if not train_log.is_file() or not best_checkpoint.is_file() or not best_weights.is_file():
        raise FileNotFoundError("run log and both best model artifacts are required")
    text = train_log.read_text(encoding="utf-8", errors="replace")
    metrics = [float(match.group(1)) for match in METRIC_PATTERN.finditer(text)]
    if not metrics:
        raise ValueError("no validation mAP values found in the training log")
    best_index, best_map = max(enumerate(metrics), key=lambda item: item[1])
    trailing_checks = len(metrics) - best_index - 1
    if trailing_checks < 1:
        raise ValueError("the latest validation result is still the best result")
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    epochs = [int(match.group(1)) for match in EPOCH_PATTERN.finditer(text)]
    summary = {
        "schema_version": 1,
        "status": "early_stopped",
        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "validation_mAP_plateau",
        "min_delta": args.min_delta,
        "best_dev_map": best_map,
        "best_validation_index": best_index + 1,
        "validation_checks": len(metrics),
        "trailing_non_improving_checks": trailing_checks,
        "last_observed_epoch": max(epochs),
        "best_checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "best_checkpoint": str(best_checkpoint),
        "best_weights": str(best_weights),
    }
    (run_dir / "training-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir / "training-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

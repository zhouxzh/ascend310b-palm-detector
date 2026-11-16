#!/usr/bin/env python3
"""Generate the four controlled tuning recipes for each top-three baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    ("lr-075", "LearningRate.base_lr", 0.75),
    ("lr-125", "LearningRate.base_lr", 1.25),
    ("wd-050", "OptimizerBuilder.regularizer.factor", 0.5),
    ("wd-200", "OptimizerBuilder.regularizer.factor", 2.0),
)


def scale_path(payload: dict, dotted_path: str, scale: float) -> None:
    keys = dotted_path.split(".")
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = round(float(target[keys[-1]]) * scale, 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, default=ROOT / "artifacts/picodet-v001/baseline-ranking.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/picodet-v001/tuning-configs")
    args = parser.parse_args()
    finalists = json.loads(args.ranking.read_text(encoding="utf-8"))["ranking"][:3]
    if len(finalists) != 3:
        raise ValueError("ranking must contain at least three completed baselines")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for finalist in finalists:
        base = Path(finalist["run_dir"]) / "config.yml"
        config = yaml.safe_load(base.read_text(encoding="utf-8"))
        for suffix, field, scale in EXPERIMENTS:
            trial = yaml.safe_load(yaml.safe_dump(config))
            scale_path(trial, field, scale)
            trial_id = f"{finalist['variant']}--{suffix}"
            path = args.output / f"{trial_id}.yml"
            path.write_text(yaml.safe_dump(trial, sort_keys=False), encoding="utf-8")
            inventory.append({"id": trial_id, "baseline": finalist["variant"], "field": field, "scale": scale, "config": str(path)})
    report = args.output / "tuning-inventory.json"
    report.write_text(json.dumps({"schema_version": 1, "finalists": finalists, "trials": inventory}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

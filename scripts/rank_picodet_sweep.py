#!/usr/bin/env python3
"""Rank completed PicoDet baseline runs strictly by audited dev mAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/picodet-v001/sweep-state.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/picodet-v001/baseline-ranking.json")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))["variants"]
    ranking = []
    for variant, record in state.items():
        if record.get("status") != "trained":
            raise ValueError(f"baseline is not complete: {variant}")
        run_dir = Path(record["run_dir"])
        summary = json.loads((run_dir / "training-summary.json").read_text(encoding="utf-8"))
        if summary.get("status") not in {"early_stopped", "max_epochs"}:
            raise ValueError(f"invalid training summary: {variant}")
        progress_path = run_dir / "training-progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        observed = progress.get("best_observed_dev_map")
        if observed is None and progress.get("history"):
            observed = max(float(item["mAP"]) for item in progress["history"])
        ranking.append({
            "variant": variant,
            "run_dir": str(run_dir),
            "best_dev_map": float(observed if observed is not None else summary["best_dev_map"]),
            "early_stop_best_dev_map": float(summary["best_dev_map"]),
        })
    ranking.sort(key=lambda item: (-item["best_dev_map"], item["variant"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "ranking": ranking}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize completed one-class NanoDet and PicoDet runs by frozen test mAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nanodet_result(run_dir: Path) -> dict | None:
    summary = run_dir / "training-summary.json"
    report = run_dir / "onnx-check.json"
    metrics = run_dir / "evaluations/test/coco-metrics.json"
    if not all(path.is_file() for path in (summary, report, metrics)):
        return None
    training = read_json(summary)
    onnx = read_json(report)
    test = read_json(metrics)
    geometry = "letterbox" if "-letterbox-" in run_dir.name else "direct_stretch"
    return {
        "framework": "NanoDet",
        "variant": run_dir.name.split("__", 1)[0].removeprefix(f"single-palm-v001-{geometry}-"),
        "geometry": geometry,
        "run_dir": str(run_dir),
        "best_dev_map": training.get("best_dev_map"),
        "test": test["metrics"],
        "onnx_valid": onnx.get("valid"),
        "onnx_sha256": onnx.get("sha256"),
    }


def picodet_result(run_dir: Path) -> dict | None:
    summary = run_dir / "training-summary.json"
    report = run_dir / "picodet-onnx-check.json"
    metrics = run_dir / "test-eval/coco-metrics.json"
    if not all(path.is_file() for path in (summary, report, metrics)):
        return None
    training = read_json(summary)
    onnx = read_json(report)
    test = read_json(metrics)
    geometry = "letterbox" if "-letterbox-" in run_dir.name else "direct_stretch"
    return {
        "framework": "PicoDet",
        "variant": run_dir.name.removeprefix(f"single-palm-v001-{geometry}-").removesuffix("-seed-42"),
        "geometry": geometry,
        "run_dir": str(run_dir),
        "best_dev_map": training.get("best_dev_map"),
        "test": test["metrics"],
        "onnx_valid": onnx.get("valid"),
        "onnx_sha256": onnx.get("sha256"),
    }


def markdown(report: dict) -> str:
    lines = [
        "# Single-Palm Full Variant Comparison",
        "",
        "| Framework | Variant | Geometry | Dev mAP | Test mAP50-95 | Test AP50 | ONNX |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in report["completed"]:
        lines.append(
            "| {framework} | {variant} | {geometry} | {dev:.4f} | {map:.4f} | {ap50:.4f} | {onnx} |".format(
                framework=item["framework"], variant=item["variant"], geometry=item["geometry"],
                dev=float(item["best_dev_map"] or 0.0), map=float(item["test"]["mAP50_95"]),
                ap50=float(item["test"]["AP50"]), onnx="PASS" if item["onnx_valid"] else "FAIL",
            )
        )
    if not report["completed"]:
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend(["", f"Completed: {len(report['completed'])}; pending or failed: {len(report['pending_or_failed'])}.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    state_path = root / "training-state.json"
    state = read_json(state_path) if state_path.is_file() else {"tasks": {}}
    completed = [
        result for result in
        [*(nanodet_result(path) for path in sorted((root / "nanodet/runs").glob("*"))),
         *(picodet_result(path) for path in sorted((root / "picodet/runs").glob("*")))]
        if result is not None
    ]
    completed.sort(key=lambda item: float(item["test"]["mAP50_95"]), reverse=True)
    completion_keys = {(item["framework"].lower(), item["geometry"], item["variant"]) for item in completed}
    pending_or_failed = [
        item for item in state.get("tasks", {}).values()
        if (item["family"], item["geometry"], item["variant"]) not in completion_keys
    ]
    report = {"schema_version": 1, "completed": completed, "pending_or_failed": pending_or_failed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

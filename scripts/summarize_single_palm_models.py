#!/usr/bin/env python3
"""Summarize completed LetterBox one-class NanoDet and PicoDet runs by mAP."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_key(item: dict) -> tuple[str, str, str]:
    return item["framework"].lower(), item["geometry"], item["variant"]


def expected_tasks(root: Path, scope: Path | None) -> set[tuple[str, str, str]]:
    if scope is not None:
        payload = read_json(scope)
        tasks = payload.get("tasks", [])
        expected = {
            (str(item["family"]), str(item["geometry"]), str(item["variant"]))
            for item in tasks
        }
        if not expected:
            raise ValueError(f"scope contains no tasks: {scope}")
        return expected

    expected: set[tuple[str, str, str]] = set()
    for family in ("nanodet", "picodet"):
        for geometry in ("letterbox",):
            inventory_path = root / family / f"variant-inventory-{geometry}.json"
            if not inventory_path.is_file():
                continue
            for item in read_json(inventory_path).get("variants", []):
                expected.add((family, geometry, item["id"]))
    return expected


def write_text_atomically(path: Path, text: str) -> None:
    """Publish a report without exposing a partially-written file to another lane."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def median(values: list[float]) -> float | None:
    """Return the median without bringing a numerical dependency into reporting."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def geometry_aggregate(completed: list[dict]) -> list[dict]:
    """Summarize completed results for the fixed LetterBox preprocessing contract."""
    summaries: list[dict] = []
    for framework in ("NanoDet", "PicoDet"):
        for geometry in ("letterbox",):
            results = [
                result for result in completed
                if result["framework"] == framework and result["geometry"] == geometry
            ]
            maps = [float(result["test"]["mAP50_95"]) for result in results]
            best = max(results, key=lambda result: float(result["test"]["mAP50_95"]), default=None)
            summaries.append({
                "framework": framework,
                "geometry": geometry,
                "completed_count": len(results),
                "mean_map50_95": sum(maps) / len(maps) if maps else None,
                "median_map50_95": median(maps),
                "best_variant": best["variant"] if best else None,
                "best_map50_95": float(best["test"]["mAP50_95"]) if best else None,
            })

    return summaries


def nanodet_result(run_dir: Path) -> dict | None:
    if "-letterbox-" not in run_dir.name:
        return None
    summary = run_dir / "training-summary.json"
    report = run_dir / "onnx-check.json"
    comparison = run_dir / "onnx-comparison.json"
    metrics = run_dir / "evaluations/test/coco-metrics.json"
    if not all(path.is_file() for path in (summary, report, comparison, metrics)):
        return None
    training = read_json(summary)
    onnx = read_json(report)
    onnx_comparison = read_json(comparison)
    test = read_json(metrics)
    if not onnx.get("valid") or not onnx_comparison.get("valid"):
        return None
    geometry = "letterbox"
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
    if "-letterbox-" not in run_dir.name:
        return None
    summary = run_dir / "training-summary.json"
    report = run_dir / "picodet-onnx-check.json"
    comparison = run_dir / "picodet-onnx-decoder-check.json"
    metrics = run_dir / "test-eval/coco-metrics.json"
    if not all(path.is_file() for path in (summary, report, comparison, metrics)):
        return None
    training = read_json(summary)
    onnx = read_json(report)
    onnx_comparison = read_json(comparison)
    test = read_json(metrics)
    if not onnx.get("valid") or not onnx_comparison.get("valid"):
        return None
    geometry = "letterbox"
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
    lines.extend([
        "",
        "## LetterBox Aggregate mAP",
        "",
        "| Framework | Geometry | Completed | Mean mAP50-95 | Median mAP50-95 | Best variant | Best mAP50-95 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ])
    for item in report["geometry_aggregate"]:
        mean = item["mean_map50_95"]
        median_map = item["median_map50_95"]
        best_map = item["best_map50_95"]
        lines.append(
            f"| {item['framework']} | {item['geometry']} | {item['completed_count']} | "
            f"{mean:.4f} | {median_map:.4f} | {item['best_variant']} | {best_map:.4f} |"
            if mean is not None else
            f"| {item['framework']} | {item['geometry']} | 0 | pending | pending | pending | pending |"
        )
    lines.extend([
        "",
        f"Completed: {report['complete_task_count']} / {report['expected_task_count']}; "
        f"pending or failed: {len(report['pending_or_failed'])}.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path, help="optional task-scope JSON")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    state_path = root / "training-state.json"
    state = read_json(state_path) if state_path.is_file() else {"tasks": {}}
    expected = expected_tasks(root, args.scope)
    all_completed = [
        result for result in
        [*(nanodet_result(path) for path in sorted((root / "nanodet/runs").glob("*"))),
         *(picodet_result(path) for path in sorted((root / "picodet/runs").glob("*")))]
        if result is not None
    ]
    completed = [item for item in all_completed if result_key(item) in expected]
    completed.sort(key=lambda item: float(item["test"]["mAP50_95"]), reverse=True)
    completion_keys = {result_key(item) for item in completed}
    state_by_key = {
        (item["family"], item["geometry"], item["variant"]): item
        for item in state.get("tasks", {}).values()
    }
    pending_or_failed = [
        state_by_key.get(key, {"family": key[0], "geometry": key[1], "variant": key[2], "status": "pending"})
        for key in sorted(expected - completion_keys)
    ]
    report = {
        "schema_version": 3,
        "expected_task_count": len(expected),
        "complete_task_count": len(completed),
        "completed": completed,
        "pending_or_failed": pending_or_failed,
        "geometry_aggregate": geometry_aggregate(completed),
    }
    write_text_atomically(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_text_atomically(args.output.with_suffix(".md"), markdown(report))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

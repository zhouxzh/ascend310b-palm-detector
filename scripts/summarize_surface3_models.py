#!/usr/bin/env python3
"""Create a reproducible comparison of completed palm-surface model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("runs must use NAME=PATH") from exc
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("runs must use a nonempty NAME and PATH")
    return name, Path(raw_path)


def summarize(name: str, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    training = read_json(run_dir / "training-summary.json")
    pico = (run_dir / "picodet-onnx-check.json").is_file()
    onnx_report = read_json(run_dir / ("picodet-onnx-check.json" if pico else "onnx-check.json"))
    metrics_path = run_dir / ("test-eval/coco-metrics.json" if pico else "evaluations/test/coco-metrics.json")
    metrics = read_json(metrics_path)
    return {
        "name": name,
        "run_dir": str(run_dir),
        "framework": "PicoDet" if pico else "NanoDet",
        "training": {
            "status": training["status"],
            "best_dev_map": training["best_dev_map"],
            "bad_checks": training["early_stop"]["bad_checks"],
        },
        "onnx": {
            "valid": onnx_report["valid"],
            "path": onnx_report["onnx"],
            "sha256": onnx_report["sha256"],
            "input": onnx_report.get("input", onnx_report.get("inputs")),
            "outputs": onnx_report.get("outputs"),
        },
        "test": {
            "metrics": metrics["metrics"],
            "per_class": metrics["per_class"],
            "ground_truth": metrics["ground_truth"],
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# Palm Surface Model Comparison",
        "",
        "| Model | Framework | Dev mAP | Test mAP | Test AP50 | Palm AP50 | Back AP50 | Side AP50 | ONNX |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["models"]:
        test = item["test"]
        per_class = test["per_class"]
        lines.append(
            "| {name} | {framework} | {dev:.4f} | {map:.4f} | {ap50:.4f} | {palm:.4f} | {back:.4f} | {side:.4f} | {onnx} |".format(
                name=item["name"],
                framework=item["framework"],
                dev=item["training"]["best_dev_map"],
                map=test["metrics"]["mAP50_95"],
                ap50=test["metrics"]["AP50"],
                palm=per_class["palm"]["AP50"],
                back=per_class["back"]["AP50"],
                side=per_class["side"]["AP50"],
                onnx="PASS" if item["onnx"]["valid"] else "FAIL",
            )
        )
    lines.extend([
        "",
        "All test metrics are evaluated on the same immutable surface3 test annotation file. Side metrics are high-variance because the test split has only 13 side instances.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=RUN_DIR; may be repeated")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    models = [summarize(name, path) for name, path in args.run]
    report = {"schema_version": 1, "models": models}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

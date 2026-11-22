#!/usr/bin/env python3
"""Package the verified formal NanoDet palm-surface ONNX release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FORMAL_RUN = ROOT / "artifacts/runs/nanodet-plus-m-416-surface3-letterbox-v003__data-v001__seed-42__20261118-1700"
FORMAL_TEST = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json"
FORMAL_OUTPUT = ROOT / "artifacts/releases/nanodet-plus-m-416-surface3-v003"
CLASSES = ("palm", "back", "side")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=FORMAL_RUN)
    parser.add_argument("--test-annotations", type=Path, default=FORMAL_TEST)
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output is not empty: {output}")

    config_path = run_dir / "config.yml"
    onnx_path = run_dir / "model-416x416.onnx"
    onnx_check_path = run_dir / "onnx-check.json"
    metrics_path = run_dir / "evaluations/test/coco-metrics.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    onnx_check = json.loads(onnx_check_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if tuple(config.get("class_names", ())) != CLASSES:
        raise ValueError("formal release must use the palm/back/side class contract")
    if config["data"]["val"]["input_size"] != [416, 416]:
        raise ValueError("formal release must use a 416x416 input")
    if not config["data"]["val"]["pipeline"].get("palm_letterbox", False):
        raise ValueError("formal release must use centered LetterBox preprocessing")
    if not onnx_check.get("valid") or onnx_check.get("sha256") != sha256_file(onnx_path):
        raise ValueError("formal ONNX check is missing, invalid, or does not match the ONNX file")
    if metrics.get("ground_truth", {}).get("sha256") != sha256_file(args.test_annotations):
        raise ValueError("test metrics do not match the cleaned formal test annotations")

    output.mkdir(parents=True, exist_ok=True)
    copied = {
        "model.onnx": onnx_path,
        "config.yml": config_path,
        "onnx-check.json": onnx_check_path,
        "onnx-comparison.json": run_dir / "onnx-comparison.json",
        "onnx-comparison-samples.json": run_dir / "onnx-comparison-samples.json",
        "test-metrics.json": metrics_path,
    }
    for destination_name, source in copied.items():
        _copy(source, output / destination_name)

    metric = metrics["metrics"]
    card = f"""---
license: apache-2.0
library_name: onnxruntime
tags:
- object-detection
- palm-detection
- onnx
---

# NanoDet Palm Surface Detector

This formal release classifies detected palm regions as `palm`, `back`, or
`side`. Only a sufficiently confident `palm` detection is eligible for the
downstream palmprint-recognition stage. The model does not infer anatomical
handedness.

## Input

- Tensor: `data`, `float32`, shape `1x3x416x416`
- Source image: OpenCV BGR
- Geometry: aspect-ratio-preserving centered zero-padded LetterBox to 416x416
- Normalization: BGR mean `[103.53, 116.28, 123.675]`, standard deviation
  `[57.375, 57.12, 58.395]`

## Output

- Tensor: `output`, `float32`, shape `1x3598x35`
- Each prior contains 3 sigmoid class scores followed by four 8-bin DFL box
  distributions. Decode and class-aware NMS are host-side; use
  `src/palm_detector/postprocess.py` from the matching project revision.

## Frozen Test Result

- mAP50-95: `{metric['mAP50_95']:.6f}`
- AP50: `{metric['AP50']:.6f}`
- Palm AP50: `{metrics['per_class']['palm']['AP50']:.6f}`
- Back AP50: `{metrics['per_class']['back']['AP50']:.6f}`
- Side AP50: `{metrics['per_class']['side']['AP50']:.6f}`

The test split is subject-disjoint and uses human-confirmed surface labels.
`onnx-check.json` verifies the static tensor contract. `onnx-comparison.json`
and `onnx-comparison-samples.json` record deterministic framework/ONNX checks.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "classes": list(CLASSES),
        "files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size} for name in (*copied, "README.md")},
        "test_annotation_sha256": sha256_file(args.test_annotations),
    }
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

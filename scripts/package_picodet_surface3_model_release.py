#!/usr/bin/env python3
"""Package the verified formal PicoDet-L palm-surface ONNX release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FORMAL_RUN = ROOT / "artifacts/picodet-surface3-v003/runs/picodet-surface3-v003-picodet-l-416-surface3-seed-42"
FORMAL_TEST = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-surface3-v001/annotations/instances_test.json"
FORMAL_OUTPUT = ROOT / "artifacts/releases/picodet-l-416-surface3-v003"
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

    config = yaml.safe_load((run_dir / "config.yml").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    onnx_path = run_dir / "picodet-416x416.onnx"
    onnx_check_path = run_dir / "picodet-onnx-check.json"
    decoder_check_path = run_dir / "picodet-onnx-decoder-check.json"
    metrics_path = run_dir / "test-eval/coco-metrics.json"
    onnx_check = json.loads(onnx_check_path.read_text(encoding="utf-8"))
    decoder_check = json.loads(decoder_check_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    transforms = config["TestReader"]["sample_transforms"]
    if tuple(run.get("classes", ())) != CLASSES or int(config.get("num_classes", 0)) != len(CLASSES):
        raise ValueError("formal release must use the palm/back/side class contract")
    if config["TestReader"]["inputs_def"]["image_shape"] != [1, 3, 416, 416]:
        raise ValueError("formal release must use a 416x416 input")
    if not any("PalmLetterBox" in item for item in transforms):
        raise ValueError("formal release must use centered LetterBox preprocessing")
    if not onnx_check.get("valid") or onnx_check.get("sha256") != sha256_file(onnx_path):
        raise ValueError("formal ONNX check is missing, invalid, or does not match the ONNX file")
    if not decoder_check.get("valid"):
        raise ValueError("formal PicoDet image-level ONNX decoder check did not pass")
    if metrics.get("ground_truth", {}).get("sha256") != sha256_file(args.test_annotations):
        raise ValueError("test metrics do not match the cleaned formal test annotations")

    output.mkdir(parents=True, exist_ok=True)
    copied = {
        "model.onnx": onnx_path,
        "picodet-onnx-check.json": onnx_check_path,
        "picodet-onnx-decoder-check.json": decoder_check_path,
        "test-metrics.json": metrics_path,
    }
    for destination_name, source in copied.items():
        _copy(source, output / destination_name)
    release_run = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "framework": "PicoDet-L / LCNet x2.0",
        "classes": list(CLASSES),
        "release": True,
    }
    (output / "run.json").write_text(
        json.dumps(release_run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "classes": list(CLASSES),
        "input": onnx_check["input"],
        "preprocessing": onnx_check["preprocessing"],
        "outputs": onnx_check["outputs"],
        "output_contract": onnx_check["output_contract"],
        "reference_inference": "scripts/run_picodet_onnx.py --run-dir <release-dir> --image <image>",
    }
    (output / "deployment-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metric = metrics["metrics"]
    card = f"""---
license: apache-2.0
library_name: onnxruntime
tags:
- object-detection
- palm-detection
- onnx
---

# PicoDet-L Palm Surface Detector

This formal alternative to the NanoDet release classifies detected palm regions
as `palm`, `back`, or `side`. Only a sufficiently confident `palm` detection is
eligible for downstream palmprint recognition. The model does not infer
anatomical handedness.

## Input

- Tensor: `image`, `float32`, shape `1x3x416x416`
- Source image: OpenCV BGR, converted to RGB
- Geometry: aspect-ratio-preserving centered zero-padded LetterBox to 416x416
- Normalization: RGB `/255`, mean `[0.485, 0.456, 0.406]`, standard deviation
  `[0.229, 0.224, 0.225]`, NCHW layout

## Output

The ONNX model emits four class-probability heads and four DFL heads. These are
raw PicoDet heads; use the matching host decoder and class-aware NMS from
`src/palm_detector/picodet_postprocess.py` or the reference command in
`deployment-contract.json`.

## Frozen Test Result

- mAP50-95: `{metric['mAP50_95']:.6f}`
- AP50: `{metric['AP50']:.6f}`
- Palm AP50: `{metrics['per_class']['palm']['AP50']:.6f}`
- Back AP50: `{metrics['per_class']['back']['AP50']:.6f}`
- Side AP50: `{metrics['per_class']['side']['AP50']:.6f}`

The test split is subject-disjoint and uses human-confirmed surface labels.
`picodet-onnx-check.json` records static/raw Paddle-to-ONNX agreement over 20
deterministic tensors. `picodet-onnx-decoder-check.json` records a 20-image
native-Paddle/ONNX decoding comparison.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    manifest_files = (*copied, "run.json", "deployment-contract.json", "README.md")
    manifest = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "classes": list(CLASSES),
        "files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size} for name in manifest_files},
        "test_annotation_sha256": sha256_file(args.test_annotations),
    }
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

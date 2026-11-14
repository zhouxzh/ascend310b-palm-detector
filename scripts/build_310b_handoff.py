#!/usr/bin/env python3
"""Build a self-contained numeric-validation package for an Atlas 310B host."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from palm_detector.dataset_contract import sha256_file
from palm_detector.postprocess import decode_nanodet_plus, detections_to_json, preprocess_bgr


def _sha256_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    args = parser.parse_args()
    onnx_path = args.run_dir / "nanodet-plus-m-320-palm.onnx"
    model_card = args.run_dir / "model-card.md"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory exists and is non-empty: {args.output_dir}")
    if not onnx_path.is_file() or not model_card.is_file() or not args.comparison_report.is_file():
        raise FileNotFoundError("run must contain ONNX, model card, and a comparison report")
    comparison = json.loads(args.comparison_report.read_text(encoding="utf-8"))
    if not comparison.get("valid"):
        raise ValueError("comparison report must pass before building a 310B handoff")
    images = [Path(item["image"]) for item in comparison["images"][: args.cases]]
    if len(images) < args.cases:
        raise ValueError(f"comparison report has only {len(images)} image cases")
    args.output_dir.mkdir(parents=True)
    shutil.copy2(onnx_path, args.output_dir / onnx_path.name)
    shutil.copy2(model_card, args.output_dir / model_card.name)
    inputs_dir = args.output_dir / "inputs"
    outputs_dir = args.output_dir / "onnx-reference-output"
    inputs_dir.mkdir()
    outputs_dir.mkdir()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cases = []
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read {image_path}")
        tensor, _ = preprocess_bgr(image)
        output = session.run(["output"], {"data": tensor})[0]
        input_path = inputs_dir / f"case-{index:03d}.npy"
        input_bin_path = inputs_dir / f"case-{index:03d}.bin"
        output_path = outputs_dir / f"case-{index:03d}.npy"
        np.save(input_path, tensor)
        tensor.astype("<f4", copy=False).tofile(input_bin_path)
        np.save(output_path, output)
        cases.append(
            {
                "id": f"case-{index:03d}",
                "source_image_path": str(image_path),
                "source_image_sha256": sha256_file(image_path),
                "source_image_shape_hwc": list(image.shape),
                "input_file": str(input_path.relative_to(args.output_dir)),
                "input_bin_file": str(input_bin_path.relative_to(args.output_dir)),
                "input_shape": list(tensor.shape),
                "input_dtype": "float32 little-endian",
                "input_bytes": input_bin_path.stat().st_size,
                "input_sha256": _sha256_array(tensor),
                "onnx_output_file": str(output_path.relative_to(args.output_dir)),
                "onnx_output_shape": list(output.shape),
                "onnx_output_sha256": _sha256_array(output),
                "onnx_detections": detections_to_json(decode_nanodet_plus(output, original_size=image.shape[:2])[0]),
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "atlas_310b_atc_acl_numeric_validation",
        "model": {"file": onnx_path.name, "sha256": sha256_file(onnx_path), "input": {"name": "data", "shape": [1, 3, 320, 320]}, "output": {"name": "output", "shape": [1, 2125, 34]}},
        "model_card": {"file": model_card.name, "sha256": sha256_file(model_card)},
        "comparison_report": {"path": str(args.comparison_report), "sha256": sha256_file(args.comparison_report)},
        "cases": cases,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

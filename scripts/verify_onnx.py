#!/usr/bin/env python3
"""Check the exported NanoDet ONNX file has the static deployment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palm_detector.dataset_contract import sha256_file


def _shape(value) -> list[int | str]:
    return [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param for dimension in value.type.tensor_type.shape.dim]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    if args.num_classes <= 0:
        parser.error("--num-classes must be positive")
    import onnx
    model = onnx.load(str(args.onnx))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    inputs = [{"name": value.name, "shape": _shape(value)} for value in inferred.graph.input if value.name == "data"]
    outputs = [{"name": value.name, "shape": _shape(value)} for value in inferred.graph.output]
    errors = []
    if inputs != [{"name": "data", "shape": [1, 3, 320, 320]}]:
        errors.append(f"expected static input data [1, 3, 320, 320], got {inputs}")
    expected_output = [1, 2125, args.num_classes + 32]
    if outputs != [{"name": "output", "shape": expected_output}]:
        errors.append(f"expected raw output output {expected_output}, got {outputs}")
    report = {"valid": not errors, "onnx": str(args.onnx), "sha256": sha256_file(args.onnx), "num_classes": args.num_classes, "inputs": inputs, "outputs": outputs, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if report['valid'] else 'FAIL'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

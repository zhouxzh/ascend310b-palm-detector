#!/usr/bin/env python3
"""Check the exported NanoDet ONNX file has the static deployment contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file


def _shape(value) -> list[int | str]:
    return [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param for dimension in value.type.tensor_type.shape.dim]


def _opset_version(model) -> int | None:
    imports = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    return max(imports, default=None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Training config; determines input, head strides and reg_max")
    parser.add_argument("--num-classes", type=int, default=3, help="Standalone verification defaults to the palm/back/side contract")
    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        width, height = (int(value) for value in config["data"]["val"]["input_size"])
        head = config["model"]["arch"]["head"]
        num_classes = int(head["num_classes"])
        strides = tuple(int(value) for value in head["strides"])
        reg_max = int(head["reg_max"])
        if tuple(config["data"]["train"]["input_size"]) != (width, height):
            raise ValueError("training and validation input sizes differ")
    else:
        width = height = 320
        num_classes = args.num_classes
        strides = (8, 16, 32, 64)
        reg_max = 7
    if width <= 0 or height <= 0 or num_classes <= 0 or reg_max <= 0 or not strides or any(stride <= 0 for stride in strides):
        parser.error("input size, class count, reg_max, and strides must be positive")
    import onnx
    model = onnx.load(str(args.onnx))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    inputs = [{"name": value.name, "shape": _shape(value)} for value in inferred.graph.input if value.name == "data"]
    outputs = [{"name": value.name, "shape": _shape(value)} for value in inferred.graph.output]
    errors = []
    expected_input = [1, 3, height, width]
    if inputs != [{"name": "data", "shape": expected_input}]:
        errors.append(f"expected static input data {expected_input}, got {inputs}")
    num_priors = sum(math.ceil(height / stride) * math.ceil(width / stride) for stride in strides)
    expected_output = [1, num_priors, num_classes + 4 * (reg_max + 1)]
    if outputs != [{"name": "output", "shape": expected_output}]:
        errors.append(f"expected raw output output {expected_output}, got {outputs}")
    report = {"valid": not errors, "onnx": str(args.onnx), "sha256": sha256_file(args.onnx), "opset_version": _opset_version(model), "num_classes": num_classes, "input_size": [width, height], "strides": list(strides), "reg_max": reg_max, "inputs": inputs, "outputs": outputs, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if report['valid'] else 'FAIL'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a fixed NanoDet model without mutating the pinned upstream source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import onnxsim
import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


def _export(model, dummy: torch.Tensor, output: Path, opset_version: int) -> None:
    torch.onnx.export(
        model,
        dummy,
        str(output),
        verbose=False,
        keep_initializers_as_inputs=True,
        opset_version=opset_version,
        input_names=["data"],
        output_names=["output"],
    )


def export_with_compatible_opset(model, dummy: torch.Tensor, output: Path, preferred_opset: int = 11) -> int:
    """Keep the deployment-friendly opset 11 unless a backbone requires 13."""
    if preferred_opset == 13:
        _export(model, dummy, output, opset_version=13)
        return 13
    try:
        _export(model, dummy, output, opset_version=11)
        return 11
    except torch.onnx.errors.UnsupportedOperatorError as exc:
        if "aten::unflatten" not in str(exc):
            raise
        _export(model, dummy, output, opset_version=13)
        return 13


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_config(cfg, str(args.config))
    input_width, input_height = (int(value) for value in cfg.data.val.input_size)
    if input_width <= 0 or input_height <= 0 or cfg.data.train.input_size != cfg.data.val.input_size:
        raise ValueError("training and validation must share a positive static input size")
    model = build_model(cfg.model).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_model_weight(model, checkpoint, Logger(-1, str(args.output.parent), False))
    dummy = torch.randn(1, 3, input_height, input_width)
    # TAN contains a transformer reshape path that materializes aten::unflatten.
    # ONNX only supports it from opset 13 onward.
    fpn_name = str(cfg.model.arch.get("fpn", {}).get("name", ""))
    preferred_opset = 13 if fpn_name == "TAN" else 11
    opset_version = export_with_compatible_opset(model, dummy, args.output, preferred_opset)
    simplified, success = onnxsim.simplify(str(args.output), input_data={"data": dummy.numpy()})
    if not success:
        raise RuntimeError("ONNX simplification failed")
    onnx.save(simplified, str(args.output))
    manifest = args.output.with_suffix(".export.json")
    manifest.write_text(json.dumps({"opset_version": opset_version}, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

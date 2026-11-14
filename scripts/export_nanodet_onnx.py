#!/usr/bin/env python3
"""Export a fixed NanoDet model without mutating the pinned upstream source."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxsim
import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_config(cfg, str(args.config))
    model = build_model(cfg.model).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_model_weight(model, checkpoint, Logger(-1, str(args.output.parent), False))
    dummy = torch.randn(1, 3, 320, 320)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        verbose=False,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    simplified, success = onnxsim.simplify(str(args.output), input_data={"data": dummy.numpy()})
    if not success:
        raise RuntimeError("ONNX simplification failed")
    onnx.save(simplified, str(args.output))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the single runtime used by NanoDet, PicoDet, ONNX, and future YOLO."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/nanodet"))
sys.path.insert(0, str(ROOT / "third_party/PaddleDetection"))


def _require(module: str) -> None:
    importlib.import_module(module)


def main() -> int:
    import paddle
    import torch

    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError(f"unexpected PyTorch binary: {torch.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    torch_sum = float((torch.ones((32, 32), device="cuda") * 2).sum().cpu())
    if torch_sum != 2048.0:
        raise RuntimeError(f"PyTorch GPU arithmetic gate failed: {torch_sum}")
    if paddle.__version__ != "3.5.0.dev20261113+12c390870c2" or not paddle.is_compiled_with_cuda():
        raise RuntimeError(f"unexpected Paddle binary: {paddle.__version__}")
    paddle.set_device("gpu:0")
    paddle_sum = float((paddle.ones([32, 32], dtype="float32") * 2).cpu().numpy().sum())
    if paddle_sum != 2048.0:
        raise RuntimeError(f"Paddle GPU arithmetic gate failed: {paddle_sum}")

    for module in (
        "nanodet.util",
        "ppdet.core.workspace",
        "onnx",
        "onnxruntime",
        "paddle2onnx",
        "tensorboardX",
        "cv2",
        "scipy",
    ):
        _require(module)

    print(json.dumps({
        "valid": True,
        "environment": sys.prefix,
        "torch": torch.__version__,
        "paddle": paddle.__version__,
        "torch_gpu_sum": torch_sum,
        "paddle_gpu_sum": paddle_sum,
        "paddledet": importlib.metadata.version("paddledet"),
        "nanodet": importlib.metadata.version("nanodet"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail closed when the PicoDet environment or audited training inputs drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import paddle
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from palm_detector.paddledetection_extensions import install_paddledetection_extensions

    install_paddledetection_extensions()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/picodet-surface3-320-letterbox-v004.yml")
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
        raise RuntimeError("exactly one usable Paddle CUDA device is required")
    paddle.set_device("gpu:0")
    x = paddle.ones([8, 8], dtype="float32")
    if float((x + x).cpu().numpy().sum()) != 128.0:
        raise RuntimeError("Paddle GPU arithmetic gate failed")
    root = ROOT / matrix["dataset"]
    train_annotations = root / matrix.get("sampling", {}).get("train_annotations", "annotations/instances_train.json")
    if not train_annotations.is_file():
        raise FileNotFoundError("training annotations are missing")
    for split in ("dev", "gate", "test"):
        if not (ROOT / matrix["validation"][split]).is_file():
            raise FileNotFoundError(f"missing {split} annotations")
    weights = json.loads((ROOT / "artifacts/pretrained/picodet/weights.json").read_text(encoding="utf-8"))["weights"]
    for item in weights.values():
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"pretrained weight verification failed: {path}")
    print(json.dumps({"valid": True, "variants": len(matrix["variants"]), "cuda": paddle.get_device()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

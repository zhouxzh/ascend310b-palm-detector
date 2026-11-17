#!/usr/bin/env python3
"""Fail closed before the two-model MediaPipe-style LetterBox retrain."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NANODET_CONFIG = ROOT / "configs/nanodet-plus-m-320-mediapipe-letterbox-v002.yml"
PICODET_MATRIX = ROOT / "configs/picodet-mediapipe-letterbox-v002.yml"


def main() -> int:
    import torch

    nanodet = yaml.safe_load(NANODET_CONFIG.read_text(encoding="utf-8"))
    picodet = yaml.safe_load(PICODET_MATRIX.read_text(encoding="utf-8"))
    blockers: list[str] = []
    for split in ("train", "val"):
        pipeline = nanodet["data"][split]["pipeline"]
        if not pipeline.get("palm_letterbox", False):
            blockers.append(f"NanoDet {split} does not use the fixed LetterBox transform")
    train = nanodet["data"]["train"]["pipeline"]
    if train.get("flip") != 0.5 or not train.get("swap_handedness_on_flip", False):
        blockers.append("NanoDet must exchange anatomical class IDs on 0.5-probability horizontal flip")
    if picodet.get("architecture") != "PicoDet":
        blockers.append("PicoDet v002 must retain the standard exportable architecture name")
    preprocess = picodet.get("preprocessing", {})
    if preprocess.get("mode") != "mediapipe_letterbox" or preprocess.get("handedness_flip_probability") != 0.5:
        blockers.append("PicoDet must use MediaPipe-style LetterBox and 0.5 semantic flip")
    if len(picodet.get("variants", [])) != 1:
        blockers.append("v002 is intentionally limited to one PicoDet comparison model")
    data_root = ROOT / picodet["dataset"]
    for relative in ("annotations/instances_train.json", "annotations/instances_test.json"):
        if not (data_root / relative).is_file():
            blockers.append(f"missing dataset file: {relative}")
    for split in ("dev", "gate"):
        if not (ROOT / picodet["validation"][split]).is_file():
            blockers.append(f"missing validation file: {split}")
    if not torch.cuda.is_available():
        blockers.append("PyTorch CUDA is unavailable")
    print(json.dumps({"ready": not blockers, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "blockers": blockers}, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())

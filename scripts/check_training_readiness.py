#!/usr/bin/env python3
"""Emit a factual readiness report without starting a training run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from palm_detector.labels import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/nanodet-plus-m-320-portable-hagridv2-handed.yml"


def _cuda_status() -> dict[str, object]:
    try:
        import torch
        return {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if tuple(config["class_names"]) != CLASS_NAMES or config["model"]["arch"]["head"]["num_classes"] != 2:
        blockers.append("model category contract is not exactly two anatomical hand classes")
    if config["data"]["train"]["pipeline"].get("flip") != 0.0:
        blockers.append("horizontal flip is enabled without a verified label-exchange transform")
    annotation_root = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations"
    required = [annotation_root / f"instances_{split}.json" for split in ("train", "val", "test")]
    missing_data = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing_data:
        blockers.append("missing audited two-class split annotations: " + ", ".join(missing_data))
    cuda = _cuda_status()
    if not cuda.get("cuda_available"):
        blockers.append("CUDA is unavailable in the active Python environment")
    nanodet = ROOT / "third_party/nanodet/.git"
    if not nanodet.exists():
        blockers.append("fixed NanoDet upstream checkout is missing; run scripts/fetch_nanodet.sh")
    report = {"ready": not blockers, "python": sys.version, "config": str(CONFIG.relative_to(ROOT)), "cuda": cuda, "blockers": blockers}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

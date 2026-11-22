#!/usr/bin/env python3
"""Fail closed before a one-class NanoDet training run begins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from palm_detector.labels import SINGLE_PALM_CATEGORY_IDS, validate_categories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry", choices=("letterbox", "direct_stretch"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    blockers = []
    classes = tuple(config.get("class_names", ()))
    head = config["model"]["arch"]["head"]
    if classes != ("palm_surface",) or int(head["num_classes"]) != 1:
        blockers.append("NanoDet must use exactly one palm_surface class")
    if "aux_head" in config["model"]["arch"] and int(config["model"]["arch"]["aux_head"]["num_classes"]) != 1:
        blockers.append("NanoDet auxiliary head must use exactly one palm_surface class")
    for split in ("train", "val"):
        data = config["data"][split]
        annotations = Path(data["ann_path"])
        images = Path(data["img_path"])
        if not annotations.is_file() or not images.is_dir():
            blockers.append(f"{split} annotations or images are missing")
            continue
        categories = json.loads(annotations.read_text(encoding="utf-8")).get("categories", [])
        blockers.extend(validate_categories(categories, SINGLE_PALM_CATEGORY_IDS))
        if bool(data["pipeline"].get("palm_letterbox", False)) != (args.geometry == "letterbox"):
            blockers.append(f"{split} geometry does not match the requested {args.geometry} policy")
        if data.get("keep_ratio", True):
            blockers.append(f"{split} must set keep_ratio=false")
    train_pipeline = config["data"]["train"]["pipeline"]
    if train_pipeline.get("flip") != 0.5:
        blockers.append("single-palm NanoDet requires label-preserving 0.5 horizontal flip")
    if train_pipeline.get("stretch") != [[1.0, 1.0], [1.0, 1.0]]:
        blockers.append("random anisotropic stretch must remain disabled")
    if config["device"].get("gpu_ids") != [0] or not torch.cuda.is_available():
        blockers.append("exactly one usable PyTorch CUDA device is required")
    print(json.dumps({"ready": not blockers, "config": str(args.config), "geometry": args.geometry, "blockers": blockers}, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())

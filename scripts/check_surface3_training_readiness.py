#!/usr/bin/env python3
"""Validate user-approved palm/back/side inputs before formal training."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file
from palm_detector.labels import SURFACE_CATEGORY_IDS, validate_categories

ROOT = Path(__file__).resolve().parents[1]
NANODET_CONFIG = ROOT / "configs/nanodet-plus-m15-416-surface3-letterbox-v003.yml"
PICODET_MATRIX = ROOT / "configs/picodet-surface3-letterbox-v003.yml"
DATASET = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-surface3-v001"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(payload: dict) -> dict[str, int]:
    result = {name: 0 for name in SURFACE_CATEGORY_IDS.values()}
    for annotation in payload["annotations"]:
        result[SURFACE_CATEGORY_IDS[annotation["category_id"]]] += 1
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanodet-config", type=Path, default=NANODET_CONFIG)
    parser.add_argument("--picodet-matrix", type=Path, default=PICODET_MATRIX)
    parser.add_argument("--expected-input-size", type=int)
    parser.add_argument("--geometry", choices=("letterbox", "direct_stretch"), default="letterbox")
    args = parser.parse_args()
    blockers: list[str] = []
    warnings: list[str] = []
    nanodet = yaml.safe_load(args.nanodet_config.read_text(encoding="utf-8"))
    picodet = yaml.safe_load(args.picodet_matrix.read_text(encoding="utf-8"))
    if tuple(nanodet["class_names"]) != tuple(SURFACE_CATEGORY_IDS.values()):
        blockers.append("NanoDet must use palm/back/side in the fixed class order")
    if tuple(picodet["classes"]) != tuple(SURFACE_CATEGORY_IDS.values()):
        blockers.append("PicoDet must use palm/back/side in the fixed class order")
    train_pipeline = nanodet["data"]["train"]["pipeline"]
    letterbox_enabled = train_pipeline.get("palm_letterbox") is True and nanodet["data"]["val"]["pipeline"].get("palm_letterbox") is True
    if args.geometry == "letterbox" and not letterbox_enabled:
        blockers.append("NanoDet surface model must use LetterBox for both train and validation")
    if args.geometry == "direct_stretch":
        if letterbox_enabled or train_pipeline.get("palm_letterbox") or nanodet["data"]["val"]["pipeline"].get("palm_letterbox"):
            blockers.append("direct-stretch NanoDet must disable LetterBox for both train and validation")
        if nanodet["data"]["train"].get("keep_ratio") or nanodet["data"]["val"].get("keep_ratio"):
            blockers.append("direct-stretch NanoDet must set keep_ratio=false")
        if train_pipeline.get("stretch") != [[1.0, 1.0], [1.0, 1.0]]:
            blockers.append("primary direct-stretch comparison must not add random anisotropic stretch")
        if picodet.get("preprocessing", {}).get("mode") != "direct_stretch":
            blockers.append("direct-stretch PicoDet matrix must declare direct_stretch preprocessing")
        if picodet.get("preprocessing", {}).get("random_anisotropic_stretch") is not False:
            blockers.append("primary direct-stretch PicoDet comparison must not add random anisotropic stretch")
    if train_pipeline.get("swap_handedness_on_flip"):
        blockers.append("surface-class horizontal flip must not exchange class IDs")
    if train_pipeline.get("flip") != 0.5:
        blockers.append("NanoDet surface model must use 0.5 label-preserving flip")
    if args.expected_input_size is not None:
        expected = [args.expected_input_size, args.expected_input_size]
        if nanodet["data"]["train"]["input_size"] != expected or nanodet["data"]["val"]["input_size"] != expected:
            blockers.append(f"NanoDet input must be {expected} for this comparison")
        sizes = [int(item["input_size"]) for item in picodet.get("variants", [])]
        if not sizes or any(size != args.expected_input_size for size in sizes):
            blockers.append(f"all PicoDet variants must use {args.expected_input_size} input for this comparison")
    manifest_path = DATASET / "surface3-manifest.json"
    if not manifest_path.is_file():
        blockers.append("surface3 dataset has not been materialized")
        manifest = {}
    else:
        manifest = _load(manifest_path)
        if manifest.get("label_status") != "user-confirmed human-validated source surface annotations":
            blockers.append("surface3 manifest lacks the user-approved label status")
        if int(manifest.get("side_repeat", 0)) < 2:
            blockers.append("surface3 training must repeat the confirmed side class")
    reports = {}
    for split in ("train_unique", "dev", "gate", "test"):
        path = DATASET / "annotations" / f"instances_{split}.json"
        if not path.is_file():
            blockers.append(f"missing converted split: {split}")
            continue
        payload = _load(path)
        errors = validate_categories(payload["categories"], SURFACE_CATEGORY_IDS)
        if errors:
            blockers.extend(errors)
        reports[split] = _counts(payload)
        if split in {"train_unique", "test"} and reports[split]["side"] == 0:
            blockers.append(f"{split} has no side instances")
        if split in {"dev", "gate"} and reports[split]["side"] == 0:
            warnings.append(f"{split} has no side instances; do not use it to tune the side decision boundary")
    train_balanced = DATASET / "annotations/instances_train_balanced.json"
    if not train_balanced.is_file():
        blockers.append("missing balanced training annotations")
    else:
        reports["train_balanced"] = _counts(_load(train_balanced))
        if reports["train_balanced"]["side"] <= reports["train_unique"]["side"]:
            blockers.append("side oversampling is not reflected in balanced training annotations")
    if reports.get("dev", {}).get("side", 0) < 10:
        warnings.append("development split has fewer than 10 side instances; treat side AP as high-variance")
    import torch

    if not torch.cuda.is_available():
        blockers.append("PyTorch CUDA is unavailable")
    print(json.dumps({
        "ready": not blockers,
        "nanodet_config": str(args.nanodet_config),
        "picodet_matrix": str(args.picodet_matrix),
        "dataset_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "class_counts": reports,
        "warnings": warnings,
        "blockers": blockers,
    }, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())

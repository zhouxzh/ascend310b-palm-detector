#!/usr/bin/env python3
"""Prepare immutable PicoDet sweep configs and provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/picodet-sweep-v001.yml"
OUTPUT = ROOT / "artifacts/picodet-v001/configs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(item: dict, matrix: dict, upstream_root: Path) -> dict:
    source = upstream_root / item["config"]
    if not source.is_file():
        raise FileNotFoundError(source)
    # Keep the upstream config as a base. Runtime overrides are emitted as a
    # separate JSON command manifest so upstream YAML tags remain untouched.
    size = int(item["input_size"])
    pretrained = matrix["pretrained"][item["pretrained"]]
    microbatch = int(item["batch_size"])
    sampling = matrix.get("sampling", {})
    train_annotations = str(sampling.get("train_annotations", "annotations/instances_train.json"))
    preprocessing = matrix.get("preprocessing", {})
    mediapipe_letterbox = preprocessing.get("mode") == "mediapipe_letterbox"
    classes = tuple(matrix["classes"])
    if classes not in {("left_palm", "right_palm"), ("palm", "back", "side")}:
        raise ValueError("PicoDet classes must be anatomical hands or palm/back/side")
    if mediapipe_letterbox:
        train_transforms = [{"Decode": {}}]
        if preprocessing.get("random_crop", True):
            train_transforms.append({"RandomCrop": {}})
        if classes == ("left_palm", "right_palm"):
            train_transforms.append({"PalmHandednessFlip": {"prob": float(preprocessing.get("handedness_flip_probability", 0.5))}})
        else:
            train_transforms.append({"RandomFlip": {"prob": float(preprocessing.get("flip_probability", 0.5))}})
        if preprocessing.get("random_distort", True):
            train_transforms.append({"RandomDistort": {}})
        train_transforms.append({"PalmLetterBox": {"target_size": [size, size], "pad_value": int(preprocessing.get("pad_value", 0))}})
        eval_transforms = [
            {"Decode": {}},
            {"PalmLetterBox": {"target_size": [size, size], "pad_value": int(preprocessing.get("pad_value", 0))}},
        ]
        train_batch_transforms = [
            {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
            {"Permute": {}},
            {"PadGT": {}},
        ]
    else:
        train_transforms = [{"Decode": {}}]
        if preprocessing.get("random_crop", True):
            train_transforms.append({"RandomCrop": {}})
        train_transforms.append({
            "RandomFlip": {"prob": 0.0 if classes == ("left_palm", "right_palm") else float(preprocessing.get("flip_probability", 0.5))}
        })
        if preprocessing.get("random_distort", True):
            train_transforms.append({"RandomDistort": {}})
        eval_transforms = [
            {"Decode": {}},
            {"Resize": {"interp": int(preprocessing.get("resize_interpolation", 2)), "target_size": [size, size], "keep_ratio": False}},
        ]
        train_batch_transforms = [
            {"BatchRandomResize": {"target_size": [size, size], "random_size": False, "random_interp": False, "keep_ratio": False}},
            {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
            {"Permute": {}},
            {"PadGT": {}},
        ]
    training_config = {
        "_BASE_": [str(source.resolve())],
        "num_classes": len(matrix["classes"]),
        **({"architecture": matrix["architecture"]} if matrix.get("architecture") else {}),
        "use_gpu": True,
        "use_vdl": True,
        "epoch": int(item["epochs"]),
        "snapshot_epoch": int(matrix["validation_interval"]),
        "worker_num": 8,
        "TrainDataset": {
            "name": "COCODataSet",
            "dataset_dir": str((ROOT / matrix["dataset"]).resolve()),
            "image_dir": "images",
            "anno_path": train_annotations,
            "data_fields": ["image", "gt_bbox", "gt_class", "is_crowd"],
        },
        "EvalDataset": {
            "name": "COCODataSet",
            "dataset_dir": str((ROOT / matrix["dataset"]).resolve()),
            "image_dir": "images",
            "anno_path": str((ROOT / matrix["validation"]["dev"]).resolve()),
            "allow_empty": True,
        },
        "TrainReader": {
            "sample_transforms": train_transforms,
            "batch_transforms": train_batch_transforms,
            "batch_size": microbatch,
            "shuffle": True,
            "drop_last": True,
        },
        "EvalReader": {
            "sample_transforms": [
                *eval_transforms,
                {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"Permute": {}},
            ],
            "batch_transforms": [{"PadBatch": {"pad_to_stride": 32}}],
            "batch_size": 8,
            "shuffle": False,
        },
        "TestReader": {
            "inputs_def": {"image_shape": [1, 3, size, size]},
            "sample_transforms": [
                *eval_transforms,
                {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"Permute": {}},
            ],
            "batch_size": 1,
        },
        "LearningRate": {"base_lr": round(item["reference_lr"] * matrix["single_gpu_count"] / matrix["reference_gpu_count"], 8)},
        "pretrain_weights": str((ROOT / "artifacts/picodet-v001/pretrained" / pretrained["filename"]).resolve()),
    }
    return {
        "id": item["id"],
        "source_config": str(source),
        "source_sha256": _sha256(source),
        "pretrained": {**pretrained, "id": item["pretrained"]},
        "input_size": item["input_size"],
        "epochs": item["epochs"],
        "batch_size": microbatch,
        "reference_batch": item["batch_size"],
        "reference_lr": item["reference_lr"],
        "reference_gpu_count": matrix["reference_gpu_count"],
        "single_gpu_count": matrix["single_gpu_count"],
        "scaled_lr": training_config["LearningRate"]["base_lr"],
        "classes": matrix["classes"],
        "dataset_root": str((ROOT / matrix["dataset"]).resolve()),
        "dev_annotations": str((ROOT / matrix["validation"]["dev"]).resolve()),
        "gate_annotations": str((ROOT / matrix["validation"]["gate"]).resolve()),
        "test_annotations": str((ROOT / matrix["validation"]["test"]).resolve()),
        "seed": matrix["seed"],
        "validation_interval": matrix["validation_interval"],
        "early_stop": matrix["early_stop"],
        "preprocessing": preprocessing,
        "sampling": sampling,
        "training_config": training_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/PaddleDetection")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    matrix_bytes = args.matrix.read_bytes()
    matrix = yaml.safe_load(matrix_bytes)
    expected_variants = int(matrix.get("expected_variants", 9))
    if len(matrix["variants"]) != expected_variants:
        raise ValueError(f"PicoDet matrix must contain exactly {expected_variants} variants")
    train_annotations = ROOT / matrix["dataset"] / matrix.get("sampling", {}).get("train_annotations", "annotations/instances_train.json")
    if not train_annotations.is_file():
        raise FileNotFoundError("training COCO annotations are missing")
    for split in ("dev", "gate", "test"):
        if not (ROOT / matrix["validation"][split]).is_file():
            raise FileNotFoundError(f"missing {split} annotations")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": 1,
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "upstream_ref": matrix["upstream"]["ref"],
        "variants": [],
    }
    for item in matrix["variants"]:
        config = build_config(item, matrix, args.upstream)
        config_path = args.output / f"{item['id']}.yml"
        config_path.write_text(yaml.safe_dump(config.pop("training_config"), sort_keys=False), encoding="utf-8")
        config["training_config_path"] = str(config_path)
        manifest_path = args.output / f"{item['id']}.json"
        manifest_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inventory["variants"].append({"id": item["id"], "path": str(manifest_path), "sha256": _sha256(manifest_path), "config": str(config_path), "config_sha256": _sha256(config_path)})
    report = args.output.parent / "variant-inventory.json"
    report.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

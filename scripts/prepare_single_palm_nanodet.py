#!/usr/bin/env python3
"""Generate immutable NanoDet configs for the complete single-palm matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/single-palm-nanodet-v001.yml"
DEFAULT_DATASET = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-single-palm-v001"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(source: dict, variant: dict, matrix: dict, dataset: Path, geometry: str) -> dict:
    """Apply only task/data/geometry changes to an upstream NanoDet recipe."""
    config = json.loads(json.dumps(source))
    classes = list(matrix["classes"])
    if classes != ["palm_surface"]:
        raise ValueError("single-palm NanoDet matrix must define exactly palm_surface")
    head = config["model"]["arch"]["head"]
    source_batch = int(config["device"]["batchsize_per_gpu"])
    batch_size = int(variant["batch_size"])
    if source_batch < 1 or batch_size < 1:
        raise ValueError("NanoDet source and configured batch sizes must be positive")
    head["num_classes"] = len(classes)
    if "aux_head" in config["model"]["arch"]:
        config["model"]["arch"]["aux_head"]["num_classes"] = len(classes)
    for split, annotations in (("train", "instances_train.json"), ("val", "instances_dev.json")):
        target = config["data"][split]
        target["img_path"] = str(dataset / "images")
        target["ann_path"] = str(dataset / "annotations" / annotations)
        target["keep_ratio"] = False
        pipeline = target.setdefault("pipeline", {})
        pipeline["palm_letterbox"] = geometry == "letterbox"
        if split == "train":
            augmentation = matrix["augmentation"]
            pipeline.update({
                "perspective": 0.0,
                "scale": augmentation["scale"],
                "stretch": [[1.0, 1.0], [1.0, 1.0]],
                "rotation": augmentation["rotation"],
                "shear": 0.0,
                "translate": augmentation["translate"],
                "flip": augmentation["flip_probability"],
                "brightness": augmentation["brightness"],
                "contrast": augmentation["contrast"],
                "saturation": augmentation["saturation"],
            })
        else:
            pipeline["flip"] = 0.0
    config["test_annotations"] = str(dataset / "annotations/instances_test.json")
    config["device"].update({
        "gpu_ids": [0],
        "workers_per_gpu": min(int(config["device"].get("workers_per_gpu", 8)), 8),
        "batchsize_per_gpu": batch_size,
        "accumulate_grad_batches": 1,
        "precision": 32,
    })
    optimizer = config["schedule"]["optimizer"]
    optimizer["lr"] = round(float(optimizer["lr"]) * batch_size / source_batch, 10)
    config["schedule"]["val_intervals"] = 5
    config["class_names"] = classes
    config["save_dir"] = "artifacts/runs/_requires_run_id"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--geometry", choices=("letterbox", "direct_stretch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix_bytes = args.matrix.read_bytes()
    matrix = yaml.safe_load(matrix_bytes)
    dataset = args.dataset.resolve()
    if not (dataset / "annotations/instances_train.json").is_file():
        raise FileNotFoundError("single-palm training annotations are missing")
    variants = matrix.get("variants", [])
    if len(variants) != 16 or len({item["id"] for item in variants}) != len(variants):
        raise ValueError("single-palm NanoDet v001 must contain exactly 16 unique variants")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": 1,
        "task": matrix["task"],
        "geometry": args.geometry,
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "dataset": str(dataset),
        "variants": [],
    }
    for variant in variants:
        source_path = ROOT / "third_party/nanodet/config" / variant["source"]
        checkpoint = ROOT / variant["checkpoint"]
        manifest = ROOT / variant["manifest"]
        if not source_path.is_file() or not checkpoint.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"missing source, checkpoint, or manifest for {variant['id']}")
        config = configure(yaml.safe_load(source_path.read_text(encoding="utf-8")), variant, matrix, dataset, args.geometry)
        config_path = args.output / f"{variant['id']}-{args.geometry}.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        inventory["variants"].append({
            "id": variant["id"],
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "input_size": config["data"]["val"]["input_size"],
            "checkpoint": str(checkpoint),
            "manifest": str(manifest),
        })
    inventory_path = args.output.parent / "variant-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(inventory_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

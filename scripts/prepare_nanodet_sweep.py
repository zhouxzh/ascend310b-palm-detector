#!/usr/bin/env python3
"""Generate audited, comparable configs for pinned NanoDet variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/nanodet/config"
COMMON = ROOT / "configs/nanodet-plus-m-320-portable-hagridv2-handed.yml"
MATRIX = ROOT / "configs/nanodet-sweep-v001.yml"
OUTPUT = ROOT / "artifacts/sweeps/palm-v001/configs"
MICROBATCH = {320: 32, 416: 32, 512: 16, 640: 8}


def prepare_variant(source: dict, common: dict, *, dataset: Path, validation: Path, effective_batch: int) -> dict:
    result = copy.deepcopy(source)
    model = result["model"]["arch"]
    width, height = result["data"]["train"]["input_size"]
    if width != height or width not in MICROBATCH:
        raise ValueError(f"unsupported model size: {width}x{height}")
    microbatch = MICROBATCH[width]
    if effective_batch % microbatch:
        raise ValueError(f"effective batch {effective_batch} is not divisible by {microbatch}")
    original_batch = result["device"]["batchsize_per_gpu"]
    for head_name in ("head", "aux_head"):
        if head_name in model:
            model[head_name]["num_classes"] = 2
    backbone = model["backbone"]
    if backbone["name"] in ("ShuffleNetV2", "EfficientNetLite"):
        backbone["pretrain"] = False
    if "pretrained" in backbone:
        backbone["pretrained"] = False
    result["class_names"] = ["left_palm", "right_palm"]
    result["data"] = copy.deepcopy(common["data"])
    for split in ("train", "val"):
        result["data"][split]["input_size"] = [width, height]
        result["data"][split]["keep_ratio"] = False
        result["data"][split]["img_path"] = str(dataset / "images")
    result["data"]["train"]["ann_path"] = str(dataset / "annotations/instances_train.json")
    result["data"]["val"]["ann_path"] = str(validation)
    result["device"]["gpu_ids"] = [0]
    result["device"]["workers_per_gpu"] = 8
    result["device"]["batchsize_per_gpu"] = microbatch
    result["device"]["accumulate_grad_batches"] = effective_batch // microbatch
    result["device"]["precision"] = 32
    optimizer = result["schedule"]["optimizer"]
    if optimizer["name"] == "SGD":
        optimizer["lr"] = round(optimizer["lr"] * effective_batch / original_batch, 8)
    elif optimizer["name"] != "AdamW":
        raise ValueError(f"unsupported optimizer: {optimizer['name']}")
    result["schedule"]["val_intervals"] = 5
    result["log"] = {"interval": 50}
    result["grad_clip"] = 35
    result["save_dir"] = "artifacts/runs/_requires_run_id"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    matrix_bytes = args.matrix.read_bytes()
    matrix = yaml.safe_load(matrix_bytes)
    common = yaml.safe_load(COMMON.read_text(encoding="utf-8"))
    if len(matrix["variants"]) != 16:
        raise ValueError("sweep must contain exactly 16 unique variants")
    ids = [item["id"] for item in matrix["variants"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate variant IDs")
    dataset = ROOT / matrix["dataset"]
    validation = ROOT / matrix["validation"]
    if not validation.is_file():
        raise FileNotFoundError(f"prepare the subject-disjoint dev/gate split first: {validation}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {"matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(), "variants": []}
    for item in matrix["variants"]:
        source_path = UPSTREAM / item["source"]
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config = prepare_variant(
            source, common, dataset=dataset, validation=validation,
            effective_batch=matrix["effective_batch"],
        )
        output_path = args.output_dir / f"{item['id']}.yml"
        output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        inventory["variants"].append({
            "id": item["id"], "source": item["source"],
            "source_sha256": sha256_file(source_path),
            "config": str(output_path), "config_sha256": sha256_file(output_path),
            "epochs": config["schedule"]["total_epochs"],
            "input_size": config["data"]["train"]["input_size"],
            "microbatch": config["device"]["batchsize_per_gpu"],
            "accumulate_grad_batches": config["device"]["accumulate_grad_batches"],
            "optimizer": config["schedule"]["optimizer"],
        })
    report = args.output_dir.parent / "variant-inventory.json"
    report.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

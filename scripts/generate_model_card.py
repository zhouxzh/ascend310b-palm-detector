#!/usr/bin/env python3
"""Generate a model card from a completed run and verified ONNX export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file
from palm_detector.postprocess import center_priors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((args.run_dir / "config.yml").read_text(encoding="utf-8"))
    onnx_report = json.loads(args.onnx_report.read_text(encoding="utf-8"))
    if not onnx_report.get("valid"):
        raise ValueError("cannot write a model card for an invalid ONNX export")
    class_names = tuple(config["class_names"])
    num_classes = int(config["model"]["arch"]["head"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError("config class_names and head num_classes disagree")
    class_table = ", ".join(f"{index + 1}={name}" for index, name in enumerate(class_names))
    head = config["model"]["arch"]["head"]
    width, height = config["data"]["val"]["input_size"]
    strides = tuple(head["strides"])
    reg_max = int(head["reg_max"])
    output_channels = num_classes + 4 * (reg_max + 1)
    prior_count = len(center_priors(height, width, strides=strides))
    mean, std = config["data"]["val"]["pipeline"]["normalize"]
    architecture = config["model"]["arch"]["name"]
    status = (
        "This is a pipeline smoke artifact and has no task-quality claim."
        if run.get("purpose") == "pipeline_smoke_not_quality_measurement"
        else "This is a training candidate until independent human-reviewed task quality and ATC/ACL performance evidence is recorded."
    )
    text = f"""# NanoDet palm detector\n\n## Identity\n\n- Run ID: `{run['run_id']}`\n- Architecture: `{architecture}`\n- Classes: `{class_table}`\n- Checkpoint SHA-256: `{sha256_file(args.run_dir / 'model_best/model_best.ckpt')}`\n- ONNX SHA-256: `{sha256_file(args.onnx)}`\n- NanoDet upstream commit: `{run['upstream_nanodet']['commit']}`\n\n## Input\n\n- Tensor: `data`, `float32`, shape `1x3x{height}x{width}`\n- OpenCV BGR input, direct resize to `{width}x{height}`\n- Normalize BGR 0-255 values with mean `{mean}` and standard deviation `{std}`\n\n## Output and host postprocessing\n\n- Tensor: `output`, `float32`, shape `1x{prior_count}x{output_channels}`\n- Layout per prior: {num_classes} sigmoid class score(s) followed by four {reg_max + 1}-bin DFL distributions\n- Feature strides: `{list(strides)}`; prior offset: `{0.0 if architecture == 'NanoDetPlus' else 0.5}`\n- Decode and class-aware NMS remain on the host; reference implementation: `src/palm_detector/postprocess.py`.\n\n## Provenance\n\n- Base config SHA-256: `{run['base_config']['sha256']}`\n- Run config SHA-256: `{run['run_config']['sha256']}`\n- Data audit SHA-256: `{run['data_audit']['sha256']}`\n- Pretrained checkpoint URL: `{run['pretrained']['upstream_url']}`\n- Pretrained checkpoint SHA-256: `{run['pretrained']['verified_sha256']}`\n- ONNX validation report: `{args.onnx_report}`\n\n{status}\n"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

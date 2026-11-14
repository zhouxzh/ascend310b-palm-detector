#!/usr/bin/env python3
"""Generate a model card from a completed run and verified ONNX export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file


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
    output_channels = num_classes + 32
    status = (
        "This is a pipeline smoke artifact and has no task-quality claim."
        if run.get("purpose") == "pipeline_smoke_not_quality_measurement"
        else "This is a training candidate until independent human-reviewed task quality and ATC/ACL performance evidence is recorded."
    )
    text = f"""# NanoDet-Plus-m anatomical hand detector\n\n## Identity\n\n- Run ID: `{run['run_id']}`\n- Classes: `1=left_palm`, `2=right_palm` (person anatomical direction)\n- Checkpoint SHA-256: `{sha256_file(args.run_dir / 'model_best/model_best.ckpt')}`\n- ONNX SHA-256: `{sha256_file(args.onnx)}`\n- NanoDet upstream commit: `{run['upstream_nanodet']['commit']}`\n\n## Input\n\n- Tensor: `data`, `float32`, shape `1x3x320x320`\n- OpenCV BGR input, bilinear resize directly to `320x320`\n- Normalize BGR 0-255 values with mean `[103.53, 116.28, 123.675]` and standard deviation `[57.375, 57.12, 58.395]`\n\n## Output and host postprocessing\n\n- Tensor: `output`, `float32`, shape `1x2125x34`\n- Layout per prior: two sigmoid class scores followed by four 8-bin DFL distributions (`reg_max=7`)\n- Feature strides: `8, 16, 32, 64`; decode and class-aware NMS remain on the host.\n- Reference implementation: `src/palm_detector/postprocess.py`; fixed defaults are score `0.35`, NMS IoU `0.60`.\n\n## Provenance\n\n- Base config SHA-256: `{run['base_config']['sha256']}`\n- Run config SHA-256: `{run['run_config']['sha256']}`\n- Data audit SHA-256: `{run['data_audit']['sha256']}`\n- Pretrained checkpoint URL: `{run['pretrained']['upstream_url']}`\n- Pretrained checkpoint SHA-256: `{run['pretrained']['verified_sha256']}`\n- ONNX validation report: `{args.onnx_report}`\n\n{status}\n"""
    text = text.replace("# NanoDet-Plus-m anatomical hand detector", "# NanoDet-Plus-m hand detector")
    text = text.replace("- Classes: `1=left_palm`, `2=right_palm` (person anatomical direction)", f"- Classes: `{class_table}`")
    text = text.replace("shape `1x2125x34`", f"shape `1x2125x{output_channels}`")
    text = text.replace("two sigmoid class scores", f"{num_classes} sigmoid class score(s)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

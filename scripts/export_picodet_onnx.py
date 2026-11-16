#!/usr/bin/env python3
"""Export one completed PicoDet run to static ONNX and validate raw outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/PaddleDetection")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    config_path = args.run_dir / "config.yml"
    checkpoint = args.run_dir / "paddle-output/best_model"
    if not checkpoint.with_suffix(".pdparams").is_file():
        raise FileNotFoundError("best PicoDet checkpoint is missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    height, width = config["TestReader"]["inputs_def"]["image_shape"][2:]
    export_root = args.run_dir / "onnx-export"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.upstream}:{env.get('PYTHONPATH', '')}"
    env["FLAGS_json_format_model"] = "1"
    command = [
        sys.executable, str(args.upstream / "tools/export_model.py"), "-c", str(config_path),
        "--output_dir", str(export_root), "-o", f"weights={checkpoint}",
        "export.post_process=False", "export.nms=False", "export.benchmark=True",
    ]
    subprocess.run(command, cwd=args.upstream, env=env, check=True)
    candidates = sorted(export_root.rglob("model.json")) + sorted(export_root.rglob("model.pdmodel"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one Paddle inference model, found: {candidates}")
    model_file = candidates[0]
    params_file = model_file.with_name("model.pdiparams")
    onnx_path = args.run_dir / f"picodet-{width}x{height}.onnx"
    subprocess.run([
        "paddle2onnx", "--model_dir", str(model_file.parent), "--model_filename", model_file.name,
        "--params_filename", params_file.name, "--opset_version", "11", "--save_file", str(onnx_path),
    ], cwd=ROOT, env=env, check=True)
    import onnx
    import onnxruntime as ort
    import paddle

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    inputs = [{"name": item.name, "shape": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in model.graph.input]
    outputs = [{"name": item.name, "shape": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in model.graph.output]
    paddle.set_device("gpu:0")
    predictor_config = paddle.inference.Config(str(model_file), str(params_file))
    predictor_config.disable_glog_info()
    predictor_config.disable_gpu()
    predictor = paddle.inference.create_predictor(predictor_config)
    input_names = predictor.get_input_names()
    if input_names != ["image"]:
        raise ValueError(f"expected postprocess-free image-only export, got {input_names}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    samples = []
    for index in range(args.samples):
        tensor = rng.normal(size=(1, 3, height, width)).astype(np.float32)
        handle = predictor.get_input_handle("image")
        handle.copy_from_cpu(tensor)
        predictor.run()
        paddle_outputs = [predictor.get_output_handle(name).copy_to_cpu() for name in predictor.get_output_names()]
        ort_outputs = session.run(None, {"image": tensor})
        if len(paddle_outputs) != len(ort_outputs):
            raise ValueError("Paddle and ONNX output count differs")
        differences = [np.abs(left - right) for left, right in zip(paddle_outputs, ort_outputs)]
        samples.append({"sample": index, "max_abs_error": max(float(item.max()) for item in differences), "mean_abs_error": max(float(item.mean()) for item in differences)})
    report = {
        "schema_version": 1, "valid": all(item["max_abs_error"] <= 1e-4 for item in samples),
        "onnx": str(onnx_path), "sha256": sha256(onnx_path), "input": [1, 3, height, width],
        "paddle_model": str(model_file), "inputs": inputs, "outputs": outputs, "samples": samples,
    }
    report_path = args.run_dir / "picodet-onnx-check.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card = args.run_dir / "picodet-model-card.md"
    card.write_text(
        f"# PicoDet Palm Detector\n\n- Run: `{args.run_dir.name}`\n- Classes: `left_palm`, `right_palm`\n- Input: `image`, float32, `1x3x{height}x{width}`\n- ONNX: `{onnx_path.name}`\n- ONNX SHA-256: `{report['sha256']}`\n- Raw Paddle/ONNX report: `{report_path.name}`\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

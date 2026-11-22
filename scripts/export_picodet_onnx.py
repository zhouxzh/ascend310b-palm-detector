#!/usr/bin/env python3
"""Export one completed PicoDet run to static ONNX and validate raw outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from palm_detector.labels import is_supported_class_contract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_contract(config: dict) -> dict:
    """Extract the deployment preprocessing contract from the generated config."""
    transforms = config["TestReader"]["sample_transforms"]
    letterbox = next((item["PalmLetterBox"] for item in transforms if "PalmLetterBox" in item), None)
    resize = next((item["Resize"] for item in transforms if "Resize" in item), None)
    normalize = next((item["NormalizeImage"] for item in transforms if "NormalizeImage" in item), None)
    if normalize is None or not any("Permute" in item for item in transforms):
        raise ValueError("PicoDet ONNX export requires NormalizeImage and Permute test transforms")
    if letterbox is not None and resize is not None:
        raise ValueError("PicoDet test transforms must not combine LetterBox and Resize")
    if letterbox is None and resize is None:
        raise ValueError("PicoDet ONNX export requires LetterBox or Resize test transforms")
    if letterbox is not None:
        target_size = [int(value) for value in letterbox["target_size"]]
        if len(target_size) != 2 or min(target_size) <= 0:
            raise ValueError("PalmLetterBox target size must contain two positive dimensions")
        geometry = {
            "geometry": "centered_zero_padded_letterbox",
            "target_size": target_size,
            "pad_value": int(letterbox.get("pad_value", 0)),
            "keep_ratio": True,
        }
    else:
        target_size = [int(value) for value in resize["target_size"]]
        if len(target_size) != 2 or min(target_size) <= 0:
            raise ValueError("Resize target size must contain two positive dimensions")
        if resize.get("keep_ratio", False):
            raise ValueError("PicoDet stretch export requires keep_ratio=false")
        geometry = {
            "geometry": "direct_stretch",
            "target_size": target_size,
            "interp": int(resize.get("interp", 2)),
            "keep_ratio": False,
        }
    return {
        "decode_color": "RGB (OpenCV BGR decode converted by PaddleDetection Decode)",
        **geometry,
        "normalize": {
            "scale": 1.0 / 255.0 if normalize.get("is_scale", False) else 1.0,
            "mean": [float(value) for value in normalize["mean"]],
            "std": [float(value) for value in normalize["std"]],
        },
        "layout": "NCHW",
    }


def prepare_pir_for_paddle2onnx(model_file: Path, conversion_dir: Path) -> tuple[Path, int]:
    """Make a converter-only copy with PicoDet resize scales in float32.

    Paddle 3.5 emits the constant ``scale=[2.0, 2.0]`` for nearest-neighbor
    upsampling as float64. Paddle2ONNX 2.1.0's PIR parser only accepts float32
    arrays for that attribute. The scale is exactly representable in float32;
    keep the original export intact and patch only the converter input copy.
    """
    conversion_dir.mkdir(parents=True, exist_ok=True)
    converted_model = conversion_dir / model_file.name
    shutil.copy2(model_file, converted_model)
    params_file = model_file.with_name("model.pdiparams")
    shutil.copy2(params_file, conversion_dir / params_file.name)
    payload = json.loads(converted_model.read_text(encoding="utf-8"))
    patched = 0
    for op in payload["program"]["regions"][0]["blocks"][0]["ops"]:
        if op.get("#") != "1.nearest_interp":
            continue
        for attribute in op.get("A", []):
            if not isinstance(attribute, dict) or attribute.get("N") != "scale":
                continue
            values = attribute.get("AT", {}).get("D", [])
            for value in values:
                if value.get("#") == "0.a_f64" and value.get("D") == 2.0:
                    value["#"] = "0.a_f32"
                    patched += 1
    if patched != 4:
        raise ValueError(f"expected four PicoDet float64 resize scales, patched {patched}")
    converted_model.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return converted_model, patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/PaddleDetection")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.upstream = args.upstream.resolve()
    config_path = args.run_dir / "config.yml"
    checkpoint = args.run_dir / "paddle-output/best_model"
    if not checkpoint.with_suffix(".pdparams").is_file():
        raise FileNotFoundError("best PicoDet checkpoint is missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preprocess = preprocessing_contract(config)
    run_metadata = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    classes = tuple(run_metadata.get("classes", ()))
    if not is_supported_class_contract(classes) or int(config["num_classes"]) != len(classes):
        raise ValueError("PicoDet export requires a supported class contract")
    height, width = config["TestReader"]["inputs_def"]["image_shape"][2:]
    export_root = args.run_dir / "onnx-export"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{args.upstream}:{env.get('PYTHONPATH', '')}"
    env["FLAGS_json_format_model"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable, str(ROOT / "scripts/export_picodet_model.py"), "-c", str(config_path.resolve()),
        "--output_dir", str(export_root), "-o", f"weights={checkpoint}",
        "export.post_process=False", "export.nms=False", "export.benchmark=True",
        "export_with_pir=True", "use_gpu=false",
    ]
    subprocess.run(command, cwd=args.upstream, env=env, check=True)
    candidates = sorted(export_root.rglob("model.json")) + sorted(export_root.rglob("model.pdmodel"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one Paddle inference model, found: {candidates}")
    model_file = candidates[0]
    params_file = model_file.with_name("model.pdiparams")
    conversion_model, conversion_patch_count = prepare_pir_for_paddle2onnx(
        model_file, args.run_dir / "onnx-conversion")
    onnx_path = args.run_dir / f"picodet-{width}x{height}.onnx"
    paddle2onnx = Path(sys.executable).with_name("paddle2onnx")
    if not paddle2onnx.is_file():
        raise FileNotFoundError(f"paddle2onnx is missing next to {sys.executable}")
    subprocess.run([
        str(paddle2onnx), "--model_dir", str(conversion_model.parent), "--model_filename", conversion_model.name,
        "--params_filename", params_file.name, "--opset_version", "11", "--save_file", str(onnx_path),
    ], cwd=ROOT, env=env, check=True)
    import onnx
    import onnxruntime as ort
    import paddle

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    inputs = [{"name": item.name, "shape": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in model.graph.input]
    outputs = [{"name": item.name, "shape": [dim.dim_value or dim.dim_param for dim in item.type.tensor_type.shape.dim]} for item in model.graph.output]
    paddle.set_device("cpu")
    predictor_config = paddle.inference.Config(str(model_file.resolve()), str(params_file.resolve()))
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
        "paddle2onnx_compatibility_patch_count": conversion_patch_count,
        "preprocessing": preprocess,
        "output_contract": "raw PicoDet per-level sigmoid class probabilities followed by DFL box distributions; deployment must decode and apply class-aware NMS",
    }
    report_path = args.run_dir / "picodet-onnx-check.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card = args.run_dir / "picodet-model-card.md"
    geometry = str(preprocess["geometry"])
    card.write_text(
        "\n".join([
            "# PicoDet Palm Surface Detector" if len(classes) == 3 else "# PicoDet Single-Palm Detector",
            "",
            f"- Run: `{args.run_dir.name}`",
            "- Classes: " + ", ".join(f"`{name}`" for name in classes),
            f"- Input: `image`, float32, `1x3x{height}x{width}`",
            f"- Preprocess: RGB decode, `{geometry}`, `/255`, ImageNet mean/std, NCHW. Exact values: `picodet-onnx-check.json`.",
            "- Output: raw per-level sigmoid class probabilities and DFL box distributions; `scripts/run_picodet_onnx.py` supplies the matching decoder and class-aware NMS.",
            f"- ONNX: `{onnx_path.name}`",
            f"- ONNX SHA-256: `{report['sha256']}`",
            f"- Raw Paddle/ONNX report: `{report_path.name}` ({len(samples)} deterministic random tensors)",
            "",
        ]),
        encoding="utf-8",
    )
    print(report_path)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

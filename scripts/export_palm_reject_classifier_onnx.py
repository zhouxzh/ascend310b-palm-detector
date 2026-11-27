#!/usr/bin/env python3
"""Export a trained palm/reject classifier checkpoint to a verified ONNX model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import load_from_disk


ROOT = Path("/home/zhong/Documents/ascend310b-palm-detector")
TRAINER_PATH = ROOT / "scripts/train_palm_reject_classifier.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("refusing to overwrite existing ONNX or metadata")
    spec = importlib.util.spec_from_file_location("classifier_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import classifier trainer")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone = str(checkpoint["backbone"])
    model, model_name, initialization = trainer.build_model(backbone, ROOT / "artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    example = torch.randn(1, 3, trainer.IMAGE_SIZE, trainer.IMAGE_SIZE, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=None,
            dynamo=False,
        )
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["valid"]["threshold"])
    table = load_from_disk(str(args.dataset))["test"]
    transform = trainer.NanoDetTransform(train=False)
    comparisons = []
    with torch.inference_mode():
        for index in range(min(32, len(table))):
            row = table[index]
            image = row["image"]
            if not hasattr(image, "convert"):
                raise TypeError(f"test image is not decoded: {type(image)}")
            tensor = transform(image.convert("RGB")).unsqueeze(0)
            torch_logits = model(tensor).cpu().numpy()
            onnx_logits = session.run(["logits"], {"images": tensor.numpy()})[0]
            absolute = np.abs(torch_logits - onnx_logits)
            torch_probability = float(torch.softmax(torch.from_numpy(torch_logits), dim=1)[0, 0])
            onnx_probability = float(np.exp(onnx_logits[0, 0] - np.max(onnx_logits[0])) / np.exp(onnx_logits[0] - np.max(onnx_logits[0])).sum())
            comparisons.append({
                "sample_id": row["sample_id"],
                "label": row["label"],
                "max_abs_error": float(np.max(absolute)),
                "mean_abs_error": float(np.mean(absolute)),
                "argmax_equal": bool(np.argmax(torch_logits) == np.argmax(onnx_logits)),
                "threshold_decision_equal": bool((torch_probability >= threshold) == (onnx_probability >= threshold)),
                "torch_palm_probability": torch_probability,
                "onnx_palm_probability": onnx_probability,
            })
    max_abs_error = max(item["max_abs_error"] for item in comparisons)
    mean_abs_error = float(np.mean([item["mean_abs_error"] for item in comparisons]))
    allclose = bool(
        max_abs_error <= 1e-3
        and all(item["argmax_equal"] and item["threshold_decision_equal"] for item in comparisons)
    )
    metadata = {
        "schema_version": 1,
        "task": "palm_reject_roi_classification",
        "model": model_name,
        "backbone": backbone,
        "initialization": initialization,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(args.output),
        "onnx_bytes": args.output.stat().st_size,
        "opset": args.opset,
        "input": {"name": "images", "shape": [1, 3, trainer.IMAGE_SIZE, trainer.IMAGE_SIZE], "dtype": "float32", "layout": "NCHW"},
        "output": {"name": "logits", "shape": [1, 2], "dtype": "float32", "labels": ["palm", "reject"]},
        "preprocessing": "RGB ROI converted to BGR, centered zero-padded LetterBox to 416x416, then normalized with NanoDet-Plus-m BGR mean/std",
        "palm_acceptance": {"probability_source": "softmax(logits)[0]", "threshold": threshold},
        "validation": metrics["valid"],
        "test": metrics["test"],
        "numerical_check": {
            "provider": "CPUExecutionProvider",
            "samples": len(comparisons),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_abs_error_tolerance": 1e-3,
            "argmax_equal_all": all(item["argmax_equal"] for item in comparisons),
            "threshold_decision_equal_all": all(item["threshold_decision_equal"] for item in comparisons),
            "allclose": allclose,
            "sample_results": comparisons,
        },
    }
    if not allclose:
        raise ValueError(f"ONNX numerical check failed: max_abs_error={max_abs_error}")
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

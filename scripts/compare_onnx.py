#!/usr/bin/env python3
"""Compare static ONNX and PyTorch raw outputs plus decoded detections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from palm_detector.postprocess import decode_nanodet_plus, detections_to_json, preprocess_bgr


def _export_equivalent_output(pytorch_output: np.ndarray) -> np.ndarray:
    """Mirror NanoDetPlusHead._forward_onnx for raw tensor comparison.

    Normal PyTorch forward returns class logits; the static ONNX branch applies
    sigmoid to the first two channels before concatenating unmodified DFL bins.
    """
    converted = np.array(pytorch_output, copy=True)
    logits = converted[:, :, :2]
    converted[:, :, :2] = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    return converted


def _iou(left: dict[str, object], right: dict[str, object]) -> float:
    left_box = np.asarray(left["bbox_xyxy"], dtype=np.float32)
    right_box = np.asarray(right["bbox_xyxy"], dtype=np.float32)
    top_left = np.maximum(left_box[:2], right_box[:2])
    bottom_right = np.minimum(left_box[2:], right_box[2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod()
    left_area = np.maximum(left_box[2] - left_box[0], 0) * np.maximum(left_box[3] - left_box[1], 0)
    right_area = np.maximum(right_box[2] - right_box[0], 0) * np.maximum(right_box[3] - right_box[1], 0)
    return float(intersection / max(left_area + right_area - intersection, 1e-8))


def _compare_detections(reference: list[dict[str, object]], candidate: list[dict[str, object]], minimum_iou: float) -> tuple[bool, float]:
    if len(reference) != len(candidate):
        return False, 0.0
    remaining = set(range(len(candidate)))
    ious: list[float] = []
    for item in reference:
        matches = [(index, _iou(item, candidate[index])) for index in remaining if candidate[index]["category_id"] == item["category_id"]]
        if not matches:
            return False, 0.0
        index, iou = max(matches, key=lambda pair: pair[1])
        if iou < minimum_iou:
            return False, iou
        remaining.remove(index)
        ious.append(iou)
    return True, min(ious, default=1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    parser.add_argument("--minimum-decoded-iou", type=float, default=0.999)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "third_party/nanodet"))
    from nanodet.model.arch import build_model
    from nanodet.util import Logger, cfg, load_config, load_model_weight
    import onnxruntime as ort

    load_config(cfg, str(args.config))
    model = build_model(cfg.model).eval()
    load_model_weight(model, torch.load(args.checkpoint, map_location="cpu"), Logger(-1, str(args.output.parent), False))
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    report_images = []
    all_ok = True
    with torch.no_grad():
        for image_path in args.images:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot read image {image_path}")
            tensor, _ = preprocess_bgr(image)
            pytorch_logits = model(torch.from_numpy(tensor)).numpy()
            pytorch_output = _export_equivalent_output(pytorch_logits)
            onnx_output = session.run(["output"], {"data": tensor})[0]
            difference = np.abs(pytorch_output - onnx_output)
            max_error = float(difference.max())
            decoded_torch = detections_to_json(decode_nanodet_plus(pytorch_logits, scores_are_logits=True, original_size=image.shape[:2])[0])
            decoded_onnx = detections_to_json(decode_nanodet_plus(onnx_output, original_size=image.shape[:2])[0])
            decoded_match, minimum_iou = _compare_detections(decoded_torch, decoded_onnx, args.minimum_decoded_iou)
            passed = max_error <= args.max_abs_error and decoded_match
            all_ok = all_ok and passed
            report_images.append({"image": str(image_path), "raw_max_abs_error": max_error, "raw_mean_abs_error": float(difference.mean()), "decoded_class_and_iou_match": decoded_match, "decoded_minimum_iou": minimum_iou, "pytorch_detections": decoded_torch, "onnx_detections": decoded_onnx, "passed": passed})
    report = {"valid": all_ok, "raw_tensor_contract": "PyTorch class logits sigmoid-transformed to match NanoDet ONNX export; DFL bins unchanged", "max_abs_error_limit": args.max_abs_error, "decoded_iou_limit": args.minimum_decoded_iou, "images": report_images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

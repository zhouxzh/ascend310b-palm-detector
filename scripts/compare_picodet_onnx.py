#!/usr/bin/env python3
"""Compare PicoDet ONNX raw heads and decoded detections with Paddle inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from palm_detector.picodet_postprocess import decode_picodet, preprocess_picodet_bgr_letterbox


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod()
    left_area = np.maximum(left[2] - left[0], 0) * np.maximum(left[3] - left[1], 0)
    right_area = np.maximum(right[2] - right[0], 0) * np.maximum(right[3] - right[1], 0)
    return float(intersection / max(left_area + right_area - intersection, 1e-8))


def compare_image(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]], *, coordinate_tolerance: float, score_tolerance: float
) -> dict[str, Any]:
    """Greedily pair class-matched detections by IoU and summarize differences."""
    unmatched = set(range(len(actual)))
    matches: list[dict[str, float | int]] = []
    for item in sorted(expected, key=lambda value: float(value["score"]), reverse=True):
        expected_box = xywh_to_xyxy(item["bbox"])
        candidates = [index for index in unmatched if actual[index]["category_id"] == item["category_id"]]
        if not candidates:
            continue
        selected = max(candidates, key=lambda index: iou(expected_box, np.asarray(actual[index]["bbox_xyxy"], dtype=np.float32)))
        actual_box = np.asarray(actual[selected]["bbox_xyxy"], dtype=np.float32)
        matches.append({
            "category_id": int(item["category_id"]),
            "iou": iou(expected_box, actual_box),
            "max_coordinate_error": float(np.abs(expected_box - actual_box).max()),
            "score_error": abs(float(item["score"]) - float(actual[selected]["score"])),
        })
        unmatched.remove(selected)
    maximum_coordinate_error = max((float(item["max_coordinate_error"]) for item in matches), default=0.0)
    maximum_score_error = max((float(item["score_error"]) for item in matches), default=0.0)
    minimum_iou = min((float(item["iou"]) for item in matches), default=1.0)
    valid = (
        len(expected) == len(actual) == len(matches)
        and maximum_coordinate_error <= coordinate_tolerance
        and maximum_score_error <= score_tolerance
    )
    return {
        "expected_count": len(expected), "actual_count": len(actual), "matched_count": len(matches),
        "max_coordinate_error": maximum_coordinate_error, "max_score_error": maximum_score_error,
        "min_iou": minimum_iou, "valid": valid,
    }


def deterministic_sample(images: list[dict[str, Any]], annotations: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    categories_by_image: dict[int, set[int]] = defaultdict(set)
    for annotation in annotations:
        categories_by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))
    by_id = {int(image["id"]): image for image in images}
    selected: list[int] = []
    # Rare classes first, then fill by stable COCO image ID.
    for category_id in (3, 2, 1):
        for image_id in sorted(image_id for image_id, categories in categories_by_image.items() if category_id in categories):
            if image_id not in selected:
                selected.append(image_id)
            if len(selected) >= sample_count:
                return [by_id[image_id] for image_id in selected]
    for image_id in sorted(by_id):
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= sample_count:
            break
    return [by_id[image_id] for image_id in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--coordinate-tolerance", type=float, default=0.15)
    parser.add_argument("--score-tolerance", type=float, default=0.005)
    # Large LCNet exports can differ by roughly 3e-4 in low-level DFL values
    # while their decoded boxes and scores remain bit-close. Keep this well
    # below a single DFL bin and verify decoded detections separately.
    parser.add_argument("--raw-tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    if args.sample_count <= 0 or args.coordinate_tolerance < 0 or args.score_tolerance < 0 or args.raw_tolerance < 0:
        raise ValueError("sample count and tolerances must be non-negative, with sample count positive")
    run_dir = args.run_dir.resolve()
    check = json.loads((run_dir / "picodet-onnx-check.json").read_text(encoding="utf-8"))
    if not check.get("valid"):
        raise ValueError("PicoDet ONNX raw-output validation did not pass")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    classes = tuple(metadata["classes"])
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    target_size = tuple(int(value) for value in check["preprocessing"]["target_size"])
    model_path = run_dir / f"picodet-{target_size[1]}x{target_size[0]}.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    import paddle

    paddle_model = Path(check["paddle_model"])
    paddle_config = paddle.inference.Config(str(paddle_model), str(paddle_model.with_name("model.pdiparams")))
    paddle_config.disable_glog_info()
    paddle_config.disable_gpu()
    predictor = paddle.inference.create_predictor(paddle_config)
    if predictor.get_input_names() != ["image"]:
        raise ValueError("PicoDet export does not have the expected image-only input")
    samples: list[dict[str, Any]] = []
    for image_metadata in deterministic_sample(annotation["images"], annotation["annotations"], args.sample_count):
        image_path = args.images_root / image_metadata["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        tensor, letterbox = preprocess_picodet_bgr_letterbox(image, input_size=target_size)
        input_handle = predictor.get_input_handle("image")
        input_handle.copy_from_cpu(tensor)
        predictor.run()
        paddle_outputs = [predictor.get_output_handle(name).copy_to_cpu() for name in predictor.get_output_names()]
        onnx_outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        if len(paddle_outputs) != len(onnx_outputs):
            raise ValueError("Paddle and ONNX output counts differ")
        raw_max_abs_error = max(float(np.abs(left - right).max()) for left, right in zip(paddle_outputs, onnx_outputs))
        paddle_detections = decode_picodet(
            paddle_outputs, class_names=classes, input_size=target_size, score_threshold=args.score_threshold,
            nms_iou_threshold=0.6, keep_top_k=100, letterbox=letterbox,
        )[0]
        onnx_detections = decode_picodet(
            onnx_outputs, class_names=classes, input_size=target_size, score_threshold=args.score_threshold,
            nms_iou_threshold=0.6, keep_top_k=100, letterbox=letterbox,
        )[0]
        expected = [{"category_id": item.label + 1, "score": item.score, "bbox": [
            item.xyxy[0], item.xyxy[1], item.xyxy[2] - item.xyxy[0], item.xyxy[3] - item.xyxy[1]
        ]} for item in paddle_detections]
        actual = [{"category_id": item.label + 1, "score": item.score, "bbox_xyxy": list(item.xyxy)} for item in onnx_detections]
        comparison = compare_image(
            expected, actual,
            coordinate_tolerance=args.coordinate_tolerance, score_tolerance=args.score_tolerance,
        )
        samples.append({
            "image_id": int(image_metadata["id"]), "file_name": image_metadata["file_name"],
            **comparison,
            "raw_max_abs_error": raw_max_abs_error,
            "valid": comparison["valid"] and raw_max_abs_error <= args.raw_tolerance,
        })
    report = {
        "schema_version": 1, "valid": all(item["valid"] for item in samples), "sample_count": len(samples),
        "score_threshold": args.score_threshold, "coordinate_tolerance": args.coordinate_tolerance,
        "score_tolerance": args.score_tolerance, "raw_tolerance": args.raw_tolerance, "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

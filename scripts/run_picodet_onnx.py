#!/usr/bin/env python3
"""Run a completed PicoDet ONNX export on one BGR image and emit detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import onnxruntime as ort

from palm_detector.picodet_postprocess import decode_picodet, preprocess_picodet_bgr, preprocess_picodet_bgr_letterbox
from palm_detector.postprocess import Detection, detections_to_json


def _unscale_direct_detections(detections: list[Detection], image_shape: tuple[int, int], input_size: tuple[int, int]) -> list[Detection]:
    original_height, original_width = image_shape
    input_height, input_width = input_size
    scale_x = original_width / input_width
    scale_y = original_height / input_height
    return [
        Detection(
            (item.xyxy[0] * scale_x, item.xyxy[1] * scale_y, item.xyxy[2] * scale_x, item.xyxy[3] * scale_y),
            item.score, item.label, item.class_names,
        )
        for item in detections
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--keep-top-k", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "picodet-onnx-check.json").read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise ValueError("PicoDet ONNX validation did not pass")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    classes = tuple(metadata["classes"])
    target_size = tuple(int(value) for value in report["preprocessing"]["target_size"])
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"cannot decode image: {args.image}")
    geometry = report["preprocessing"]["geometry"]
    if geometry == "centered_zero_padded_letterbox":
        tensor, letterbox = preprocess_picodet_bgr_letterbox(image, input_size=target_size)
        direct_stretch = False
    elif geometry == "direct_stretch":
        tensor = preprocess_picodet_bgr(
            image, input_size=target_size,
            mean_rgb=report["preprocessing"]["normalize"]["mean"],
            std_rgb=report["preprocessing"]["normalize"]["std"],
            interpolation=int(report["preprocessing"].get("interp", 2)),
        )
        letterbox = None
        direct_stretch = True
    else:
        raise ValueError(f"unsupported PicoDet preprocessing geometry: {geometry}")
    session = ort.InferenceSession(str(run_dir / f"picodet-{target_size[1]}x{target_size[0]}.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    detections = decode_picodet(
        outputs,
        class_names=classes,
        input_size=target_size,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        keep_top_k=args.keep_top_k,
        letterbox=letterbox,
    )[0]
    if direct_stretch:
        detections = _unscale_direct_detections(detections, image.shape[:2], target_size)
    payload = {
        "image": str(args.image.resolve()),
        "model": str(run_dir / f"picodet-{target_size[1]}x{target_size[0]}.onnx"),
        "classes": list(classes),
        "detections": detections_to_json(detections),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

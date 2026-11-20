#!/usr/bin/env python3
"""Evaluate exported palm-detector ONNX models on an explicit COCO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def coco_report(annotations_path: Path, predictions: list[dict[str, Any]], expected: dict[int, str]) -> dict[str, Any]:
    from palm_detector.labels import validate_categories

    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    errors = validate_categories(payload.get("categories", []), expected)
    if errors:
        raise ValueError("; ".join(errors))
    ground_truth = COCO(str(annotations_path))
    detections = ground_truth.loadRes(predictions)
    evaluation = COCOeval(ground_truth, detections, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    names = ("mAP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    metrics = {name: float(value) for name, value in zip(names, evaluation.stats[:6])}
    precision = evaluation.eval["precision"]
    per_class = {}
    for index, category_id in enumerate(sorted(expected)):
        ap50 = precision[0, :, index, 0, -1]
        all_ap = precision[:, :, index, 0, -1]
        valid_ap50 = ap50[ap50 > -1]
        valid_all = all_ap[all_ap > -1]
        per_class[expected[category_id]] = {
            "AP50": float(np.mean(valid_ap50)) if valid_ap50.size else None,
            "mAP50_95": float(np.mean(valid_all)) if valid_all.size else None,
        }
    return {"metrics": metrics, "per_class": per_class, "prediction_count": len(predictions)}


def detections_to_coco(image_id: int, detections) -> list[dict[str, Any]]:
    values = []
    for item in detections:
        left, top, right, bottom = item.xyxy
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        values.append(
            {
                "image_id": int(image_id),
                "category_id": int(item.label + 1),
                "bbox": [float(left), float(top), float(width), float(height)],
                "score": float(item.score),
            }
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path("/home/zhong/Documents/ascend310b-palm-detector")
    sys.path.insert(0, str(root / "src"))
    from palm_detector.labels import COCO_CATEGORY_IDS, SINGLE_PALM_CATEGORY_IDS, SURFACE_CATEGORY_IDS
    from palm_detector.picodet_postprocess import decode_picodet, preprocess_picodet_bgr, preprocess_picodet_bgr_letterbox
    from palm_detector.postprocess import decode_nanodet_plus, preprocess_bgr, preprocess_bgr_letterbox

    contracts = {
        "left_right": COCO_CATEGORY_IDS,
        "single_palm": SINGLE_PALM_CATEGORY_IDS,
        "surface3": SURFACE_CATEGORY_IDS,
    }
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.only)
    summary = {"schema_version": 1, "kind": "onnx_split_map", "started_at": now(), "jobs": []}
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if args.provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
    for job in jobs["jobs"]:
        name = job["name"]
        if requested and name not in requested:
            continue
        job_dir = output_dir / name
        job_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"name": name, "status": "running", "job": job}
        try:
            expected = contracts[job["category_contract"]]
            class_names = tuple(expected[key] for key in sorted(expected))
            onnx_path = Path(job["onnx"])
            annotations = Path(job["annotations"])
            images_root = Path(job["images_root"])
            payload = json.loads(annotations.read_text(encoding="utf-8"))
            options = ort.SessionOptions()
            session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
            active = session.get_providers()
            if args.provider == "cuda" and (not active or active[0] != "CUDAExecutionProvider"):
                raise RuntimeError(f"CUDA not primary: {active}")
            input_name = session.get_inputs()[0].name
            input_size = tuple(int(value) for value in job["input_size"])
            family = job["family"]
            geometry = job["geometry"]
            score_threshold = float(job.get("score_threshold", 0.05 if family == "nanodet" else 0.025))
            nms_iou = float(job.get("nms_iou", 0.6))
            keep_top_k = int(job.get("keep_top_k", 100))
            records = sorted(payload["images"], key=lambda item: int(item["id"]))
            if job.get("expected_images") and len(records) != int(job["expected_images"]):
                raise ValueError(f"image count {len(records)} != {job['expected_images']}")
            first = cv2.imread(str(images_root / records[0]["file_name"]), cv2.IMREAD_COLOR)
            if first is None:
                raise ValueError(f"cannot read {records[0]['file_name']}")
            def preprocess(image):
                if family == "nanodet" and geometry == "direct_stretch":
                    tensor, _ = preprocess_bgr(image, input_size=input_size)
                    return tensor, None
                if family == "nanodet" and geometry == "letterbox":
                    return preprocess_bgr_letterbox(image, input_size=input_size)
                if family == "picodet" and geometry == "direct_stretch":
                    return preprocess_picodet_bgr(image, input_size=input_size), None
                if family == "picodet" and geometry == "letterbox":
                    return preprocess_picodet_bgr_letterbox(image, input_size=input_size)
                raise ValueError(f"unsupported family/geometry {family}/{geometry}")
            warm, _ = preprocess(first)
            for _ in range(args.warmup):
                session.run(None, {input_name: warm})
            predictions = []
            timings = []
            for index, item in enumerate(records, start=1):
                image = cv2.imread(str(images_root / item["file_name"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot decode {item['file_name']}")
                tensor, letterbox = preprocess(image)
                started = time.perf_counter_ns()
                outputs = session.run(None, {input_name: tensor})
                timings.append((time.perf_counter_ns() - started) / 1e6)
                if family == "nanodet":
                    detections = decode_nanodet_plus(
                        outputs[0],
                        class_names=class_names,
                        input_size=input_size,
                        strides=tuple(job.get("strides", [8, 16, 32, 64])),
                        reg_max=int(job.get("reg_max", 7)),
                        center_offset=float(job.get("center_offset", 0.0)),
                        score_threshold=score_threshold,
                        nms_iou_threshold=nms_iou,
                        original_size=None if letterbox is not None else (int(item["height"]), int(item["width"])),
                        letterbox=letterbox,
                    )[0]
                else:
                    detections = decode_picodet(
                        outputs,
                        class_names=class_names,
                        input_size=input_size,
                        score_threshold=score_threshold,
                        nms_iou_threshold=nms_iou,
                        keep_top_k=keep_top_k,
                        letterbox=letterbox,
                    )[0]
                    if letterbox is None:
                        scale_x = float(item["width"]) / input_size[1]
                        scale_y = float(item["height"]) / input_size[0]
                        scaled = []
                        for det in detections:
                            left, top, right, bottom = det.xyxy
                            scaled.append(type(det)((left * scale_x, top * scale_y, right * scale_x, bottom * scale_y), det.score, det.label, det.class_names))
                        detections = scaled
                predictions.extend(detections_to_coco(int(item["id"]), detections))
                if index % 100 == 0:
                    print(f"{name}: {index}/{len(records)}", flush=True)
            pred_path = job_dir / "predictions.coco.json"
            write_json(pred_path, predictions)
            metrics = coco_report(annotations, predictions, expected)
            ordered = sorted(timings)
            report.update(
                {
                    "status": "completed",
                    "onnx_sha256": sha256_file(onnx_path),
                    "annotations_sha256": sha256_file(annotations),
                    "active_providers": active,
                    "predictions": {"path": str(pred_path), "sha256": sha256_file(pred_path), "count": len(predictions)},
                    "metrics": metrics,
                    "performance": {
                        "p50_ms": ordered[len(ordered) // 2] if ordered else None,
                        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None,
                        "count": len(timings),
                    },
                }
            )
        except BaseException as exc:  # noqa: BLE001
            report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        summary["jobs"].append(report)
        write_json(output_dir / "summary.json", summary)
        print(report["status"], name, flush=True)
    summary["finished_at"] = now()
    write_json(output_dir / "summary.json", summary)
    failures = [item["name"] for item in summary["jobs"] if item["status"] != "completed"]
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

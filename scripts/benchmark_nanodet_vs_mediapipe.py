#!/usr/bin/env python3
"""Benchmark NanoDet ONNX CUDA inference against a MediaPipe GPU task.

NanoDet is a palm detector only. MediaPipe's Hand Landmarker task includes palm
detection, landmark inference, and task-level postprocessing, so the latency
figures are reported separately and are not a like-for-like speed ranking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from palm_detector.dataset_contract import sha256_file

NANODET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
NANODET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanodet-onnx", type=Path, required=True)
    parser.add_argument("--mediapipe-task", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "iterations": len(values),
        "mean_ms": statistics.fmean(values) * 1_000,
        "median_ms": statistics.median(values) * 1_000,
        "p90_ms": percentile(values, 0.9) * 1_000,
        "throughput_fps_from_mean": 1 / statistics.fmean(values),
    }


def task_payload(task_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(task_path) as archive:
        members = {member.filename: member.file_size for member in archive.infolist()}
    return {
        "path": str(task_path),
        "sha256": sha256_file(task_path),
        "file_bytes": task_path.stat().st_size,
        "embedded_models": members,
    }


def nanodet_input(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {image_path}")
    resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - NANODET_MEAN) / NANODET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    for path in (args.nanodet_onnx, args.mediapipe_task, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)

    report: dict[str, Any] = {
        "status": "blocked",
        "purpose": "GPU deployment-cost comparison, not detector-quality comparison",
        "image": {"path": str(args.image), "sha256": sha256_file(args.image)},
        "models": {
            "nanodet": {
                "path": str(args.nanodet_onnx),
                "sha256": sha256_file(args.nanodet_onnx),
                "file_bytes": args.nanodet_onnx.stat().st_size,
                "input_contract": "float32 NCHW [1,3,320,320], BGR direct resize and NanoDet normalization",
                "scope": "palm detector forward graph only",
            },
            "mediapipe": {
                **task_payload(args.mediapipe_task),
                "scope": "Hand Landmarker task: palm detection plus landmark inference and task postprocessing",
                "teacher_equivalence": "not established; portable dataset labels use a MediaPipe 0.10.14 legacy graph",
            },
        },
        "benchmark": {"warmup": args.warmup, "iterations": args.iterations},
        "warnings": [
            "MediaPipe task latency includes hand landmark inference and is not directly comparable to detector-only NanoDet latency.",
            "The two models use different static input sizes: NanoDet 320x320 and embedded MediaPipe hand_detector 192x192.",
        ],
        "errors": [],
    }
    try:
        import onnxruntime as ort
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        providers = ort.get_available_providers()
        session = ort.InferenceSession(
            str(args.nanodet_onnx),
            providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        )
        active_providers = session.get_providers()
        report["nanodet_runtime"] = {
            "available_providers": providers,
            "active_providers": active_providers,
        }
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError("ONNX Runtime could not activate CUDAExecutionProvider")

        task_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(args.mediapipe_task),
                delegate=python.BaseOptions.Delegate.GPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        media_image = mp.Image.create_from_file(str(args.image))
        detector = vision.HandLandmarker.create_from_options(task_options)
        try:
            model_input = nanodet_input(args.image)
            for _ in range(args.warmup):
                session.run(None, {"data": model_input})
                detector.detect(media_image)

            nanodet_latencies = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                session.run(None, {"data": model_input})
                nanodet_latencies.append(time.perf_counter() - started)

            mediapipe_latencies = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                detector.detect(media_image)
                mediapipe_latencies.append(time.perf_counter() - started)
        finally:
            detector.close()

        report["results"] = {
            "nanodet_onnx_cuda_forward": latency_summary(nanodet_latencies),
            "mediapipe_gpu_hand_landmarker": latency_summary(mediapipe_latencies),
        }
        report["status"] = "complete"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {report['status']}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

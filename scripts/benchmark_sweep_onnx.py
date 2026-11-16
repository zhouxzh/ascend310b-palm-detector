#!/usr/bin/env python3
"""Measure batch-one ONNX Runtime latency under a fixed local protocol."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from palm_detector.dataset_contract import sha256_file
from palm_detector.postprocess import preprocess_bgr


def benchmark(session: ort.InferenceSession, tensor: np.ndarray, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        session.run(["output"], {"data": tensor})
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        session.run(["output"], {"data": tensor})
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    values = np.asarray(timings)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": statistics.fmean(timings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 20 or args.cpu_threads < 1:
        parser.error("warmup and threads must be positive; repeats must be at least 20")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    width, height = config["data"]["val"]["input_size"]
    mean, std = config["data"]["val"]["pipeline"]["normalize"]
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"cannot decode {args.image}")
    tensor, _ = preprocess_bgr(image, input_size=(height, width), mean=mean, std=std)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.cpu_threads
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if args.gpu:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError(f"requested provider was not activated: {session.get_providers()}")
    result = {
        "onnx": str(args.onnx), "onnx_sha256": sha256_file(args.onnx),
        "config_sha256": sha256_file(args.config), "image_sha256": sha256_file(args.image),
        "input_shape": list(tensor.shape), "provider": session.get_providers()[0],
        "cpu_threads": args.cpu_threads, "warmup": args.warmup, "repeats": args.repeats,
        "metric": benchmark(session, tensor, warmup=args.warmup, repeats=args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

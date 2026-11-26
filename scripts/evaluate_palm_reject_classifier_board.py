#!/usr/bin/env python3
"""Evaluate palm/reject classifier ONNX and OM models on an Ascend board."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from datasets import load_from_disk
from PIL import Image


IMAGE_SIZE = 416
MEAN = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    normalized = (canvas.astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32)


def softmax_palm(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32).reshape(-1, 2)
    value = value - np.max(value, axis=1, keepdims=True)
    probability = np.exp(value)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return probability[:, 0]


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores < threshold).astype(np.int64)
    palm_true = labels == 0
    palm_pred = prediction == 0
    tp_palm = int(np.sum(palm_true & palm_pred))
    fp_palm = int(np.sum(~palm_true & palm_pred))
    fn_palm = int(np.sum(palm_true & ~palm_pred))
    tp_reject = int(np.sum(~palm_true & ~palm_pred))
    fp_reject = int(np.sum(palm_true & ~palm_pred))
    fn_reject = int(np.sum(~palm_true & palm_pred))
    palm_total = tp_palm + fn_palm
    reject_total = tp_reject + fn_reject

    def f1(tp: int, fp: int, fn: int) -> float:
        denominator = 2 * tp + fp + fn
        return 0.0 if denominator == 0 else 2 * tp / denominator

    palm_f1 = f1(tp_palm, fp_palm, fn_palm)
    reject_f1 = f1(tp_reject, fp_reject, fn_reject)
    return {
        "threshold": threshold,
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": (palm_f1 + reject_f1) / 2.0,
        "palm_precision": 0.0 if tp_palm + fp_palm == 0 else tp_palm / (tp_palm + fp_palm),
        "palm_recall": 0.0 if palm_total == 0 else tp_palm / palm_total,
        "palm_f1": palm_f1,
        "palm_false_reject_rate": 0.0 if palm_total == 0 else fn_palm / palm_total,
        "reject_precision": 0.0 if tp_reject + fp_reject == 0 else tp_reject / (tp_reject + fp_reject),
        "reject_recall": 0.0 if reject_total == 0 else tp_reject / reject_total,
        "reject_f1": reject_f1,
        "reject_false_accept_rate": 0.0 if reject_total == 0 else fp_palm / reject_total,
        "predicted_reject_rate": float(np.mean(prediction == 1)),
        "confusion": {
            "palm_as_palm": tp_palm,
            "reject_as_palm": fp_palm,
            "palm_as_reject": fn_palm,
            "reject_as_reject": tp_reject,
        },
    }


class AclRunner:
    def __init__(self, path: Path) -> None:
        import acl

        self.acl = acl
        self.path = path
        self.context = None
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.buffers: list[dict[str, Any]] = []
        self.input_shapes: list[tuple[int, ...]] = []
        self.output_shapes: list[tuple[int, ...]] = []
        self.input_dtypes: list[np.dtype] = []
        self.output_dtypes: list[np.dtype] = []
        self.closed = False
        self._init()

    @staticmethod
    def _result_code(result: Any) -> int:
        value = result[-1] if isinstance(result, tuple) else result
        return int(value)

    @classmethod
    def _check(cls, result: Any, operation: str) -> Any:
        value = result[0] if isinstance(result, tuple) else None
        code = cls._result_code(result)
        if code != 0:
            raise RuntimeError(f"{operation} failed: {code}")
        return value

    @staticmethod
    def _dims(acl: Any, desc: Any, getter: Any, index: int) -> tuple[int, ...]:
        info = AclRunner._check(getter(desc, index), "get dimensions")
        return tuple(int(value) for value in info["dims"][: int(info["dimCount"])])

    @staticmethod
    def _dtype(acl: Any, value: Any) -> np.dtype:
        mapping = {
            int(getattr(acl, "ACL_FLOAT", 0)): np.dtype(np.float32),
            int(getattr(acl, "ACL_FLOAT16", 1)): np.dtype(np.float16),
        }
        if int(value) not in mapping:
            raise ValueError(f"unsupported OM dtype: {value}")
        return mapping[int(value)]

    def _init(self) -> None:
        acl = self.acl
        self._check(acl.init(), "acl.init")
        self._check(acl.rt.set_device(0), "acl.rt.set_device")
        self.context = self._check(acl.rt.create_context(0), "acl.rt.create_context")
        self.model_id = self._check(acl.mdl.load_from_file(str(self.path)), "acl.mdl.load_from_file")
        self.desc = acl.mdl.create_desc()
        if self.desc is None:
            raise RuntimeError("acl.mdl.create_desc returned None")
        self._check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")
        input_count = int(acl.mdl.get_num_inputs(self.desc))
        output_count = int(acl.mdl.get_num_outputs(self.desc))
        if input_count != 1 or output_count != 1:
            raise ValueError(f"expected one input/output, got {input_count}/{output_count}")
        self.input_shapes = [self._dims(acl, self.desc, acl.mdl.get_input_dims, index) for index in range(input_count)]
        self.output_shapes = [self._dims(acl, self.desc, acl.mdl.get_output_dims, index) for index in range(output_count)]
        self.input_dtypes = [self._dtype(acl, acl.mdl.get_input_data_type(self.desc, index)) for index in range(input_count)]
        self.output_dtypes = [self._dtype(acl, acl.mdl.get_output_data_type(self.desc, index)) for index in range(output_count)]
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()
        if self.input_dataset is None or self.output_dataset is None:
            raise RuntimeError("failed to create ACL datasets")
        for index in range(input_count):
            self._allocate(index, True)
        for index in range(output_count):
            self._allocate(index, False)

    def _allocate(self, index: int, is_input: bool) -> None:
        acl = self.acl
        getter = acl.mdl.get_input_size_by_index if is_input else acl.mdl.get_output_size_by_index
        size = int(getter(self.desc, index))
        pointer = self._check(acl.rt.malloc(size, 0), "acl.rt.malloc")
        data_buffer = acl.create_data_buffer(pointer, size)
        if data_buffer is None:
            raise RuntimeError("acl.create_data_buffer returned None")
        dataset = self.input_dataset if is_input else self.output_dataset
        self._check(acl.mdl.add_dataset_buffer(dataset, data_buffer), "acl.mdl.add_dataset_buffer")
        self.buffers.append({"input": is_input, "pointer": pointer, "data_buffer": data_buffer, "size": size})

    def run(self, value: np.ndarray) -> np.ndarray:
        acl = self.acl
        if tuple(value.shape) != self.input_shapes[0]:
            raise ValueError(f"input shape mismatch: {value.shape} != {self.input_shapes[0]}")
        input_buffer = self.buffers[0]
        output_buffer = self.buffers[1]
        source = np.ascontiguousarray(value, dtype=self.input_dtypes[0])
        self._check(
            acl.rt.memcpy(
                input_buffer["pointer"],
                input_buffer["size"],
                acl.util.numpy_to_ptr(source),
                source.nbytes,
                int(getattr(acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1)),
            ),
            "host_to_device",
        )
        self._check(acl.rt.set_context(self.context), "acl.rt.set_context")
        self._check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "acl.mdl.execute")
        synchronize = getattr(acl.rt, "synchronize_device", None)
        if callable(synchronize):
            self._check(synchronize(), "acl.rt.synchronize_device")
        output = np.empty(self.output_shapes[0], dtype=self.output_dtypes[0])
        self._check(
            acl.rt.memcpy(
                acl.util.numpy_to_ptr(output),
                output.nbytes,
                output_buffer["pointer"],
                output_buffer["size"],
                int(getattr(acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2)),
            ),
            "device_to_host",
        )
        return output.astype(np.float32, copy=False)

    def close(self) -> None:
        if self.closed:
            return
        acl = self.acl
        for buffer in reversed(self.buffers):
            acl.destroy_data_buffer(buffer["data_buffer"])
            acl.rt.free(buffer["pointer"])
        for dataset in (self.output_dataset, self.input_dataset):
            if dataset is not None:
                acl.mdl.destroy_dataset(dataset)
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(0)
        acl.finalize()
        self.closed = True


def convert_onnx_to_om(onnx_path: Path, output_base: Path, *, soc_version: str, precision_mode: str, log_path: Path) -> dict[str, Any]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    om_path = output_base.with_suffix(".om")
    if om_path.exists():
        om_path.unlink()
    command = [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_base}",
        f"--soc_version={soc_version}",
        "--input_format=NCHW",
        "--input_shape=images:1,3,416,416",
        f"--precision_mode={precision_mode}",
        "--output_type=FP32",
        "--log=error",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not om_path.is_file():
        raise RuntimeError(f"ATC failed with exit {completed.returncode}; log={log_path}")
    return {"path": str(om_path), "bytes": om_path.stat().st_size, "sha256": sha256_file(om_path), "command": command, "log": str(log_path)}


def verify_transform(dataset: Any, contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["numerical_check"]["sample_results"]
    differences = []
    session = ort.InferenceSession(next(Path(contract["_model_dir"]).glob("*.onnx")).as_posix(), providers=["CPUExecutionProvider"])
    for index, expected_row in enumerate(expected):
        if dataset[index]["sample_id"] != expected_row["sample_id"]:
            raise ValueError(f"reference order mismatch at {index}")
        logits = session.run(["logits"], {"images": preprocess(dataset[index]["image"])})[0]
        probability = float(softmax_palm(logits)[0])
        differences.append(abs(probability - float(expected_row["onnx_palm_probability"])))
    maximum = max(differences, default=0.0)
    if maximum > 1e-4:
        raise ValueError(f"board preprocessing differs from recorded contract: max probability delta={maximum}")
    return {"samples": len(differences), "max_probability_delta": maximum, "passed": True}


def evaluate_onnx(onnx_path: Path, contract: dict[str, Any], dataset: Any) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    threshold = float(contract["palm_acceptance"]["threshold"])
    scores = []
    labels = []
    durations = []
    for index in range(len(dataset)):
        if index % 100 == 0:
            print(f"ONNX_PROGRESS {index}/{len(dataset)}", flush=True)
        tensor = preprocess(dataset[index]["image"])
        started = time.perf_counter()
        output = session.run(["logits"], {"images": tensor})[0]
        durations.append((time.perf_counter() - started) * 1000.0)
        scores.append(float(softmax_palm(output)[0]))
        labels.append(int(dataset[index]["label_id"]))
    return {"metrics": metrics(np.asarray(scores), np.asarray(labels), threshold), "latency_ms": latency_summary(durations), "_scores": np.asarray(scores)}


def evaluate_om(om_path: Path, contract: dict[str, Any], dataset: Any) -> dict[str, Any]:
    runner = AclRunner(om_path)
    threshold = float(contract["palm_acceptance"]["threshold"])
    scores = []
    labels = []
    durations = []
    try:
        for index in range(len(dataset)):
            if index % 100 == 0:
                print(f"NPU_PROGRESS {index}/{len(dataset)}", flush=True)
            tensor = preprocess(dataset[index]["image"])
            started = time.perf_counter()
            output = runner.run(tensor)
            durations.append((time.perf_counter() - started) * 1000.0)
            scores.append(float(softmax_palm(output)[0]))
            labels.append(int(dataset[index]["label_id"]))
    finally:
        runner.close()
    return {"metrics": metrics(np.asarray(scores), np.asarray(labels), threshold), "latency_ms": latency_summary(durations), "_scores": np.asarray(scores)}


class JsonRecordDataset:
    def __init__(self, path: Path) -> None:
        self.root = path.parent
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        self.records = self.payload.get("records")
        if not isinstance(self.records, list) or not self.records:
            raise ValueError(f"annotation file has no records: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = str(record["label"])
        if label not in {"palm", "reject"}:
            raise ValueError(f"invalid label at {index}: {label!r}")
        image_path = self.root / str(record["image"])
        return {
            "sample_id": str(record["sample_id"]),
            "image": Image.open(image_path).convert("RGB"),
            "label": label,
            "label_id": 0 if label == "palm" else 1,
        }


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="datasets.save_to_disk directory")
    source.add_argument("--annotations-json", type=Path, help="existing test.json with relative image paths")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--soc-version", default="Ascend310B1")
    parser.add_argument("--precision-mode", default="allow_fp32_to_fp16")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-om", action="store_true")
    parser.add_argument("--model-id", action="append", help="evaluate only the named model directory; repeatable")
    args = parser.parse_args()
    dataset = load_from_disk(str(args.dataset))["test"] if args.dataset else JsonRecordDataset(args.annotations_json)
    manifest = json.loads((args.model_root / "palm-reject-models.json").read_text(encoding="utf-8"))
    selected_models = set(args.model_id or [])
    available_models = {entry["dirname"] for entry in manifest["models"]}
    unknown_models = selected_models - available_models
    if unknown_models:
        raise ValueError(f"unknown model IDs: {sorted(unknown_models)}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "task": "palm_reject_roi_classification",
        "started_at": utc_now(),
        "dataset_size": len(dataset),
        "split": "test",
        "soc_version": args.soc_version,
        "precision_mode": args.precision_mode,
        "models": {},
    }
    for entry in manifest["models"]:
        slug = entry["dirname"]
        if selected_models and slug not in selected_models:
            continue
        print(f"MODEL_BEGIN {slug}", flush=True)
        model_dir = args.model_root / slug
        contract = json.loads((model_dir / "contract.json").read_text(encoding="utf-8"))
        contract["_model_dir"] = str(model_dir)
        onnx_path = next(model_dir.glob("*.onnx"))
        model_report: dict[str, Any] = {
            "display_name": entry["display_name"],
            "onnx": {"path": str(onnx_path), "bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
            "transform_check": verify_transform(dataset, contract),
        }
        onnx_result = None if args.skip_onnx else evaluate_onnx(onnx_path, contract, dataset)
        if onnx_result is not None:
            model_report["onnx"].update(public_result(onnx_result))
        if not args.skip_om:
            om_dir = args.work_dir / "om"
            om_meta = convert_onnx_to_om(
                onnx_path,
                om_dir / f"palm-reject-{slug}",
                soc_version=args.soc_version,
                precision_mode=args.precision_mode,
                log_path=args.work_dir / "logs" / f"{slug}-atc.log",
            )
            om_result = evaluate_om(Path(om_meta["path"]), contract, dataset)
            model_report["npu"] = {**om_meta, **public_result(om_result)}
            if onnx_result is not None:
                score_delta = np.abs(onnx_result["_scores"] - om_result["_scores"])
                model_report["onnx_vs_npu"] = {
                    "max_abs_probability_delta": float(np.max(score_delta)),
                    "mean_abs_probability_delta": float(np.mean(score_delta)),
                    "decision_agreement": float(np.mean((onnx_result["_scores"] >= contract["palm_acceptance"]["threshold"]) == (om_result["_scores"] >= contract["palm_acceptance"]["threshold"]))),
                }
        report["models"][slug] = model_report
        report["updated_at"] = utc_now()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"MODEL_COMPLETE {slug} onnx={bool(onnx_result is not None)} om={bool(not args.skip_om)}", flush=True)
    report["finished_at"] = utc_now()
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

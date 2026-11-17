"""Bounded ONNX inference for the local hand-surface review desk."""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import yaml

from .dataset_contract import sha256_file
from .postprocess import center_priors, decode_nanodet_plus, detections_to_json, preprocess_bgr, preprocess_bgr_letterbox


@dataclass(frozen=True)
class ReviewModel:
    id: str
    path: Path
    sha256: str
    class_names: tuple[str, ...]
    label: str
    input_size: tuple[int, int]
    strides: tuple[int, ...]
    reg_max: int
    center_offset: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    letterbox: bool

    def public(self) -> dict[str, object]:
        return {"id": self.id, "label": self.label, "run_id": self.id, "class_names": list(self.class_names), "sha256": self.sha256}


class OnnxReviewModels:
    def __init__(self, runs_root: Path, *, cache_size: int = 32):
        self.models: dict[str, ReviewModel] = {}
        self.sessions: dict[str, object] = {}
        self.cache: OrderedDict[tuple[str, str, int, int], object] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()
        for report_path in runs_root.glob("*/onnx-check.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                config = yaml.safe_load((report_path.parent / "config.yml").read_text(encoding="utf-8"))
                head = config["model"]["arch"]["head"]
                width, height = config["data"]["val"]["input_size"]
                input_size = (height, width)
                letterbox = bool(config["data"]["val"].get("pipeline", {}).get("palm_letterbox", False))
                if config["data"]["val"].get("keep_ratio", False) and not letterbox:
                    continue
                strides = tuple(int(value) for value in head["strides"])
                reg_max = int(head["reg_max"])
                mean, std = config["data"]["val"]["pipeline"]["normalize"]
                class_names = tuple(config["class_names"])
                if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
                    continue
                outputs = report.get("outputs")
                if not isinstance(outputs, list) or len(outputs) != 1:
                    continue
                channels = outputs[0].get("shape", [None, None, None])[-1]
                num_classes = report.get("num_classes", len(class_names))
                classes = {1: ("palm",), 2: ("left_palm", "right_palm")}.get(num_classes)
                if not report.get("valid") or classes is None:
                    continue
                if class_names != classes:
                    continue
                if report.get("inputs") != [{"name": "data", "shape": [1, 3, height, width]}]:
                    continue
                priors = len(center_priors(height, width, strides=strides, center_offset=0.0 if config["model"]["arch"]["name"] == "NanoDetPlus" else 0.5))
                if report.get("outputs") != [{"name": "output", "shape": [1, priors, 4 * (reg_max + 1) + len(classes)]}]:
                    continue
                candidates = list(report_path.parent.glob("*.onnx"))
                if len(candidates) != 1 or sha256_file(candidates[0]) != report.get("sha256"):
                    continue
                path = candidates[0]
                run_id = report_path.parent.name
                stamp = re.search(r"__(\d{8})-\d{4}$", run_id)
                date = f"{stamp[1][:4]}-{stamp[1][4:6]}-{stamp[1][6:8]}" if stamp else ""
                if "portable-handed" in run_id:
                    source = "左右手 · 便携HaGRIDv2"
                elif "portable-hagridv2-palm" in run_id:
                    source = "单类手 · 便携HaGRIDv2"
                else:
                    source = f"{'左右手' if len(classes) == 2 else '单类手'} · {run_id}"
                architecture = "NanoDet-Plus" if config["model"]["arch"]["name"] == "NanoDetPlus" else "NanoDet"
                label = f"{architecture} {width} · {source}{f' · {date}' if date else ''}"
                model = ReviewModel(run_id, path, report["sha256"], classes, label, input_size, strides, reg_max, 0.0 if architecture == "NanoDet-Plus" else 0.5, tuple(mean), tuple(std), letterbox)
                self.models[model.id] = model
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError):
                continue

    def catalog(self) -> list[dict[str, object]]:
        return [model.public() for model in sorted(self.models.values(), key=lambda item: item.path.stat().st_mtime, reverse=True)]

    def _session(self, model: ReviewModel):
        with self.lock:
            if model.id not in self.sessions:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                session = ort.InferenceSession(str(model.path), sess_options=options, providers=["CPUExecutionProvider"])
                if [entry.name for entry in session.get_inputs()] != ["data"] or [entry.name for entry in session.get_outputs()] != ["output"]:
                    raise ValueError(f"unexpected ONNX inputs or outputs: {model.id}")
                self.sessions[model.id] = session
            return self.sessions[model.id]

    def infer(self, image_path: Path, model_ids: list[str], score_threshold: float) -> dict[str, object]:
        if not isinstance(model_ids, list) or len(model_ids) > 2 or len(model_ids) != len(set(model_ids)):
            raise ValueError("select at most two distinct models")
        if not 0.05 <= score_threshold <= 0.95:
            raise ValueError("score threshold must be between 0.05 and 0.95")
        if any(model_id not in self.models for model_id in model_ids):
            raise ValueError("unknown ONNX model")
        if not model_ids:
            return {"models": []}

        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        image_stat = image_path.stat()
        results = []
        for model_id in model_ids:
            model = self.models[model_id]
            if model.letterbox:
                tensor, letterbox = preprocess_bgr_letterbox(image, input_size=model.input_size, mean=model.mean, std=model.std)
            else:
                tensor, _ = preprocess_bgr(image, input_size=model.input_size, mean=model.mean, std=model.std)
                letterbox = None
            key = (model_id, str(image_path.resolve()), image_stat.st_size, image_stat.st_mtime_ns)
            with self.lock:
                output = self.cache.get(key)
                if output is not None:
                    self.cache.move_to_end(key)
            if output is None:
                output = self._session(model).run(["output"], {"data": tensor})[0]
                with self.lock:
                    self.cache[key] = output
                    self.cache.move_to_end(key)
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            detections = decode_nanodet_plus(
                output, class_names=model.class_names, score_threshold=score_threshold,
                original_size=None if letterbox is not None else image.shape[:2], letterbox=letterbox, input_size=model.input_size,
                strides=model.strides, reg_max=model.reg_max,
                center_offset=model.center_offset,
            )[0]
            results.append({"id": model.id, "label": model.label, "detections": detections_to_json(detections[:50])})
        return {"models": results}

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from palm_detector.onnx_review import OnnxReviewModels


class _Session:
    def __init__(self, channels: int, num_priors: int = 2125, score_index: int | None = None):
        self.channels = channels
        self.num_priors = num_priors
        self.score_index = channels - 33 if score_index is None else score_index
        self.calls = 0

    def run(self, names, inputs):
        self.calls += 1
        self.input_shape = inputs["data"].shape
        output = np.zeros((1, self.num_priors, self.channels), dtype=np.float32)
        output[0, 0, self.score_index] = 0.95
        return [output]


class OnnxReviewTest(unittest.TestCase):
    def test_single_and_dual_models_decode_and_cache_raw_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, channels in (("single", 33), ("dual", 34)):
                run = root / name
                run.mkdir()
                onnx = run / "model.onnx"
                onnx.write_bytes(f"fake-{name}".encode())
                (run / "config.yml").write_text(yaml.safe_dump({
                    "model": {"arch": {"name": "NanoDetPlus", "head": {"strides": [8, 16, 32, 64], "reg_max": 7}}},
                    "class_names": ["palm"] if name == "single" else ["left_palm", "right_palm"],
                    "data": {"val": {"input_size": [320, 320], "pipeline": {"normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}}},
                }))
                (run / "onnx-check.json").write_text(json.dumps({
                    "valid": True,
                    "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
                    "inputs": [{"name": "data", "shape": [1, 3, 320, 320]}],
                    "outputs": [{"name": "output", "shape": [1, 2125, channels]}],
                }))
            models = OnnxReviewModels(root)
            self.assertEqual({model["id"] for model in models.catalog()}, {"single", "dual"})
            single = _Session(33)
            dual = _Session(34)
            models.sessions.update({"single": single, "dual": dual})
            image = root / "sample.jpg"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8)))
            result = models.infer(image, ["single", "dual"], 0.9)
            self.assertEqual([entry["detections"][0]["class_name"] for entry in result["models"]], ["palm", "right_palm"])
            self.assertEqual(single.input_shape, (1, 3, 320, 320))
            models.infer(image, ["single", "dual"], 0.8)
            self.assertEqual((single.calls, dual.calls), (1, 1))
            with self.assertRaisesRegex(ValueError, "at most two distinct"):
                models.infer(image, ["single", "single"], 0.35)

    def test_catalog_and_inference_use_per_model_size_and_legacy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variants = (
                ("plus416", 416, "NanoDetPlus", [8, 16, 32, 64], 7, ["left_palm", "right_palm"], [104.0, 115.0, 123.0], [58.0, 57.0, 58.0]),
                ("legacy512", 512, "OneStageDetector", [8, 16, 32], 16, ["palm"], [127.0, 127.0, 127.0], [128.0, 128.0, 128.0]),
                ("plus640", 640, "NanoDetPlus", [8, 16, 32, 64], 7, ["palm"], [103.53, 116.28, 123.675], [57.375, 57.12, 58.395]),
            )
            sessions = {}
            for name, size, arch, strides, reg_max, classes, mean, std in variants:
                run = root / name
                run.mkdir()
                onnx = run / "model.onnx"
                onnx.write_bytes(name.encode())
                priors = sum(((size + stride - 1) // stride) ** 2 for stride in strides)
                channels = len(classes) + 4 * (reg_max + 1)
                config = {
                    "model": {"arch": {"name": arch, "head": {"strides": strides, "reg_max": reg_max}}},
                    "class_names": classes,
                    "data": {"val": {"input_size": [size, size], "keep_ratio": False, "pipeline": {"normalize": [mean, std]}}},
                }
                (run / "config.yml").write_text(yaml.safe_dump(config))
                (run / "onnx-check.json").write_text(json.dumps({
                    "valid": True,
                    "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
                    "inputs": [{"name": "data", "shape": [1, 3, size, size]}],
                    "outputs": [{"name": "output", "shape": [1, priors, channels]}],
                }))
                sessions[name] = _Session(channels, priors, len(classes) - 1)

            models = OnnxReviewModels(root)
            self.assertEqual({model["id"] for model in models.catalog()}, {"plus416", "legacy512", "plus640"})
            models.sessions.update(sessions)
            image = root / "source.jpg"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((180, 260, 3), dtype=np.uint8)))
            result = models.infer(image, ["plus416", "legacy512"], 0.9)
            self.assertEqual([item["detections"][0]["class_name"] for item in result["models"]], ["right_palm", "palm"])
            self.assertEqual(sessions["plus416"].input_shape, (1, 3, 416, 416))
            self.assertEqual(sessions["legacy512"].input_shape, (1, 3, 512, 512))
            plus640 = models.infer(image, ["plus640"], 0.9)["models"][0]
            self.assertEqual(sessions["plus640"].input_shape, (1, 3, 640, 640))
            self.assertEqual(plus640["detections"][0]["class_name"], "palm")
            self.assertEqual(models.models["legacy512"].center_offset, 0.5)

    def test_catalog_rejects_letterboxed_config_that_shared_preprocessor_cannot_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "letterbox"
            run.mkdir()
            onnx = run / "model.onnx"
            onnx.write_bytes(b"model")
            (run / "config.yml").write_text(yaml.safe_dump({
                "model": {"arch": {"name": "NanoDetPlus", "head": {"strides": [8, 16, 32, 64], "reg_max": 7}}},
                "class_names": ["palm"],
                "data": {"val": {"input_size": [320, 320], "keep_ratio": True, "pipeline": {"normalize": [[1, 1, 1], [1, 1, 1]]}}},
            }))
            (run / "onnx-check.json").write_text(json.dumps({
                "valid": True, "sha256": hashlib.sha256(b"model").hexdigest(),
                "inputs": [{"name": "data", "shape": [1, 3, 320, 320]}],
                "outputs": [{"name": "output", "shape": [1, 2125, 33]}],
            }))
            self.assertEqual(OnnxReviewModels(root).catalog(), [])

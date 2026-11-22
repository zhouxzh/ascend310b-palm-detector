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
    def __init__(self, channels: int, priors: int):
        self.channels = channels
        self.priors = priors
        self.calls = 0

    def run(self, names, inputs):
        self.calls += 1
        self.input_shape = inputs["data"].shape
        output = np.zeros((1, self.priors, self.channels), dtype=np.float32)
        output[0, 0, 2] = 0.95
        return [output]


class OnnxReviewTest(unittest.TestCase):
    def test_catalog_only_accepts_surface3_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, classes in (("surface", ["palm", "back", "side"]), ("incomplete", ["palm", "back"])):
                run = root / name
                run.mkdir()
                onnx = run / "model.onnx"
                onnx.write_bytes(name.encode())
                channels = len(classes) + 32
                (run / "config.yml").write_text(yaml.safe_dump({
                    "model": {"arch": {"name": "NanoDetPlus", "head": {"strides": [8, 16, 32, 64], "reg_max": 7}}},
                    "class_names": classes,
                    "data": {"val": {"input_size": [320, 320], "keep_ratio": False, "pipeline": {"normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}}},
                }))
                (run / "onnx-check.json").write_text(json.dumps({
                    "valid": True, "num_classes": len(classes), "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
                    "inputs": [{"name": "data", "shape": [1, 3, 320, 320]}], "outputs": [{"name": "output", "shape": [1, 2125, channels]}],
                }))
            models = OnnxReviewModels(root)
            self.assertEqual([item["id"] for item in models.catalog()], ["surface"])
            models.sessions["surface"] = _Session(35, 2125)
            image = root / "sample.jpg"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8)))
            output = models.infer(image, ["surface"], 0.9)
            self.assertEqual(output["models"][0]["detections"][0]["class_name"], "side")

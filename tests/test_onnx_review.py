from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from palm_detector.onnx_review import OnnxReviewModels


class _Session:
    def __init__(self, channels: int):
        self.channels = channels
        self.calls = 0

    def run(self, names, inputs):
        self.calls += 1
        self.input_shape = inputs["data"].shape
        output = np.zeros((1, 2125, self.channels), dtype=np.float32)
        output[0, 0, self.channels - 33] = 0.95
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

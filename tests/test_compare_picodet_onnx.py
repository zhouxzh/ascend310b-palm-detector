from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ComparePicoDetOnnxTest(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "scripts/compare_picodet_onnx.py"
        spec = importlib.util.spec_from_file_location("compare_picodet_onnx", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_exact_class_matched_boxes_pass_and_mismatched_classes_fail(self) -> None:
        expected = [{"category_id": 2, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.8}]
        actual = [{"category_id": 2, "bbox_xyxy": [10.03, 19.98, 40.01, 60.02], "score": 0.8002}]
        passed = self.module.compare_image(expected, actual, coordinate_tolerance=0.05, score_tolerance=0.001)
        self.assertTrue(passed["valid"])
        self.assertGreater(passed["min_iou"], 0.99)
        actual[0]["category_id"] = 1
        failed = self.module.compare_image(expected, actual, coordinate_tolerance=0.05, score_tolerance=0.001)
        self.assertFalse(failed["valid"])

    def test_sample_selects_rare_side_images_before_common_classes(self) -> None:
        images = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        annotations = [
            {"image_id": 1, "category_id": 1}, {"image_id": 2, "category_id": 2},
            {"image_id": 3, "category_id": 3}, {"image_id": 4, "category_id": 3},
        ]
        selected = self.module.deterministic_sample(images, annotations, 3)
        self.assertEqual([item["id"] for item in selected], [3, 4, 2])

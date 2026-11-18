from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class PrepareSurface3CocoTest(unittest.TestCase):
    def test_side_is_oversampled_and_unknown_images_are_excluded(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_surface3_coco.py"
        spec = importlib.util.spec_from_file_location("prepare_surface3_coco", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {
            "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}, {"id": 3, "file_name": "c.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "surface_view": "palm", "surface_candidate_view": "palm", "bbox": [1, 1, 2, 2]},
                {"id": 2, "image_id": 2, "surface_view": "unknown", "surface_candidate_view": "side", "bbox": [1, 1, 2, 2]},
                {"id": 3, "image_id": 3, "surface_view": "unknown", "surface_candidate_view": "unknown", "bbox": [1, 1, 2, 2]},
            ],
            "licenses": [],
        }
        unique, unique_report = module._convert_split(source, allowed_images=None, side_repeat=1)
        balanced, balanced_report = module._convert_split(source, allowed_images=None, side_repeat=3)
        self.assertEqual([item["name"] for item in unique["categories"]], ["palm", "back", "side"])
        self.assertEqual(len(unique["images"]), 2)
        self.assertEqual([item["category_id"] for item in unique["annotations"]], [1, 3])
        self.assertEqual(unique_report["unknown_images_excluded"], 1)
        self.assertEqual(len(balanced["images"]), 4)
        self.assertEqual(balanced_report["balanced_class_counts"], {"palm": 1, "side": 3})

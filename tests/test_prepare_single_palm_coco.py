from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareSinglePalmCocoTest(unittest.TestCase):
    def test_collapse_excludes_side_images_and_keeps_only_coco_fields(self) -> None:
        path = ROOT / "scripts/prepare_single_palm_coco.py"
        spec = importlib.util.spec_from_file_location("prepare_single_palm_coco", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {
            "categories": [{"id": 1, "name": "palm"}, {"id": 2, "name": "back"}, {"id": 3, "name": "side"}],
            "images": [
                {"id": 1, "file_name": "a.jpg", "height": 10, "width": 20, "sha256": "a", "subject_id": "s1", "session_id": "r1", "source_image": "a.jpg", "extra": "drop"},
                {"id": 2, "file_name": "b.jpg", "height": 10, "width": 20, "sha256": "b", "subject_id": "s2", "session_id": "r2", "source_image": "b.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0, "handedness_score": 0.9},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [2, 3, 4, 5], "area": 20, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 3, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0},
            ],
        }
        with self.subTest("collapse"):
            import json
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                annotation = Path(temporary) / "source.json"
                annotation.write_text(json.dumps(source), encoding="utf-8")
                result, counts = module.collapse_split(annotation)
        self.assertEqual([item["id"] for item in result["images"]], [1])
        self.assertEqual([item["category_id"] for item in result["annotations"]], [1, 1])
        self.assertNotIn("handedness_score", result["annotations"][0])
        self.assertNotIn("extra", result["images"][0])
        self.assertEqual(counts["images_excluded_with_side"], 1)

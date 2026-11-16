from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


class SweepValidationTest(unittest.TestCase):
    def test_split_keeps_subjects_and_annotations_together(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_sweep_validation.py"
        spec = importlib.util.spec_from_file_location("prepare_sweep_validation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        images = [
            {"id": index, "subject_id": f"s{index // 3}"}
            for index in range(30)
        ]
        source = {
            "images": images,
            "annotations": [{"id": index, "image_id": index, "category_id": 1} for index in range(30)],
            "categories": [{"id": 1, "name": "left_palm"}],
        }
        dev, gate = module.split_validation(source, gate_fraction=0.3, seed=42)
        self.assertEqual(len(dev["images"]) + len(gate["images"]), 30)
        self.assertFalse({item["subject_id"] for item in dev["images"]} & {item["subject_id"] for item in gate["images"]})
        for payload in (dev, gate):
            self.assertEqual({item["id"] for item in payload["images"]}, {item["image_id"] for item in payload["annotations"]})
        self.assertEqual((dev, gate), module.split_validation(source, gate_fraction=0.3, seed=42))

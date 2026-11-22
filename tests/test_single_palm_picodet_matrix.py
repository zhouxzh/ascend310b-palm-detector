from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SinglePalmPicoDetMatrixTest(unittest.TestCase):
    def test_both_geometry_matrices_cover_the_same_23_coco_variants(self) -> None:
        matrices = [
            yaml.safe_load((ROOT / "configs/picodet-single-palm-letterbox-v001.yml").read_text(encoding="utf-8")),
            yaml.safe_load((ROOT / "configs/picodet-single-palm-stretch-v001.yml").read_text(encoding="utf-8")),
        ]
        identifiers = []
        for matrix in matrices:
            self.assertEqual(matrix["expected_variants"], 23)
            self.assertEqual(matrix["classes"], ["palm_surface"])
            self.assertEqual(len(matrix["variants"]), 23)
            self.assertEqual(len({item["id"] for item in matrix["variants"]}), 23)
            identifiers.append([item["id"] for item in matrix["variants"]])
        self.assertEqual(identifiers[0], identifiers[1])
        self.assertIn("legacy-esnet-l-640", identifiers[0])
        self.assertIn("legacy-resnet18-640", identifiers[0])
        self.assertTrue(all("_coco" in item["config"] for item in matrices[0]["variants"]))

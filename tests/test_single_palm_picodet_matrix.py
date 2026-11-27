from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SinglePalmPicoDetMatrixTest(unittest.TestCase):
    def test_letterbox_matrix_covers_the_23_coco_variants(self) -> None:
        matrix = yaml.safe_load((ROOT / "configs/picodet-single-palm-letterbox-v001.yml").read_text(encoding="utf-8"))
        self.assertEqual(matrix["expected_variants"], 23)
        self.assertEqual(matrix["classes"], ["palm_surface"])
        self.assertEqual(len(matrix["variants"]), 23)
        self.assertEqual(len({item["id"] for item in matrix["variants"]}), 23)
        identifiers = [item["id"] for item in matrix["variants"]]
        self.assertIn("legacy-esnet-l-640", identifiers)
        self.assertIn("legacy-resnet18-640", identifiers)
        self.assertTrue(all("_coco" in item["config"] for item in matrix["variants"]))
        m416 = next(item for item in matrix["variants"] if item["id"] == "picodet-m-416")
        self.assertEqual(m416["batch_size"], 32)
        self.assertEqual(m416["reference_batch_size"], 48)

    def test_m416_generator_scales_learning_rate_with_reduced_microbatch(self) -> None:
        path = ROOT / "scripts" / "prepare_picodet_sweep.py"
        spec = importlib.util.spec_from_file_location("prepare_picodet_sweep", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        matrix_path = ROOT / "configs/picodet-single-palm-letterbox-v001.yml"
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        m416 = next(item for item in matrix["variants"] if item["id"] == "picodet-m-416")
        generated = module.build_config(m416, matrix, ROOT / "third_party/PaddleDetection")
        self.assertEqual(generated["batch_size"], 32)
        self.assertEqual(generated["reference_batch"], 48)
        self.assertEqual(generated["scaled_lr"], 0.04)
        self.assertEqual(generated["training_config"]["TrainReader"]["batch_size"], 32)
        self.assertEqual(generated["training_config"]["LearningRate"]["base_lr"], 0.04)

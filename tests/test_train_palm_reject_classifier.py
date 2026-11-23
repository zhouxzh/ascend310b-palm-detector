from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_palm_reject_classifier", ROOT / "scripts/train_palm_reject_classifier.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PalmRejectClassifierTest(unittest.TestCase):
    def test_supported_backbones_are_fixed(self) -> None:
        self.assertEqual(
            MODULE.BACKBONES,
            ("nanodet_shufflenetv2_x1", "lcnet_100", "efficientnet_b0"),
        )

    def test_perfect_threshold_metrics(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.1, 0.05], dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = MODULE.metrics_at_threshold(scores, labels, 0.5)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["palm_true_positive"], 2)
        self.assertEqual(result["reject_true_positive"], 2)

    def test_threshold_selection_uses_macro_f1(self) -> None:
        scores = np.asarray([0.9, 0.7, 0.3, 0.1], dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = MODULE.select_threshold(scores, labels)
        self.assertGreater(result["threshold"], 0.3)
        self.assertLessEqual(result["threshold"], 0.7)
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

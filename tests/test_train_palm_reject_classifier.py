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

    def test_threshold_selection_respects_false_accept_constraint(self) -> None:
        scores = np.asarray([0.9, 0.7, 0.3, 0.1], dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = MODULE.select_threshold(scores, labels, max_false_accept_rate=0.0)
        self.assertGreater(result["threshold"], 0.3)
        self.assertLessEqual(result["threshold"], 0.7 + 1e-6)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["reject_false_accept_rate"], 0.0)
        self.assertEqual(result["palm_false_reject_rate"], 0.0)
        self.assertEqual(result["roc_auc"], 1.0)

    def test_false_accept_rate_uses_reject_support(self) -> None:
        scores = np.asarray([0.9, 0.6, 0.4, 0.2], dtype=np.float32)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        result = MODULE.metrics_at_threshold(scores, labels, 0.5)
        self.assertEqual(result["reject_false_accept_rate"], 0.0)
        self.assertEqual(result["palm_false_reject_rate"], 0.0)
        self.assertEqual(result["predicted_reject_rate"], 0.5)

    def test_realistic_augmentation_profile_builds_tensor(self) -> None:
        from PIL import Image

        transform = MODULE.NanoDetTransform(train=True, profile="realistic")
        result = transform(Image.new("RGB", (96, 128), color=(120, 100, 80)))
        self.assertEqual(tuple(result.shape), (3, MODULE.IMAGE_SIZE, MODULE.IMAGE_SIZE))
        self.assertEqual(str(result.dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()

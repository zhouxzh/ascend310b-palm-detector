from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class PreprocessExtensionsTest(unittest.TestCase):
    def test_v002_recipe_declares_semantic_flip_and_letterbox_for_both_splits(self) -> None:
        import yaml

        config = yaml.safe_load((ROOT / "configs/nanodet-plus-m-320-mediapipe-letterbox-v002.yml").read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["train"]["pipeline"]["flip"], 0.5)
        self.assertTrue(config["data"]["train"]["pipeline"]["swap_handedness_on_flip"])
        self.assertTrue(config["data"]["train"]["pipeline"]["palm_letterbox"])
        self.assertTrue(config["data"]["val"]["pipeline"]["palm_letterbox"])

    def test_nanodet_letterbox_flip_swaps_anatomical_labels(self) -> None:
        sys.path.insert(0, str(ROOT / "third_party/nanodet"))
        from palm_detector.nanodet_extensions import install_nanodet_extensions

        install_nanodet_extensions()
        from nanodet.data.transform.warp import ShapeTransform

        transform = ShapeTransform(
            keep_ratio=False, palm_letterbox=True, swap_handedness_on_flip=True,
            scale=(1.0, 1.0), stretch=((1.0, 1.0), (1.0, 1.0)), rotation=0,
            shear=0, perspective=0, translate=0, flip=1.0,
        )
        sample = {
            "img": np.full((100, 200, 3), 127, dtype=np.uint8),
            "gt_bboxes": np.array([[20, 10, 80, 90]], dtype=np.float32),
            "gt_labels": np.array([0], dtype=np.int64),
        }
        result = transform(sample, (320, 320))
        self.assertEqual(result["img"].shape, (320, 320, 3))
        self.assertEqual(result["gt_labels"].tolist(), [1])
        np.testing.assert_allclose(result["gt_bboxes"], [[192, 96, 288, 224]], atol=1e-5)

    def test_paddledetection_transforms_swap_labels_and_center_boxes(self) -> None:
        sys.path.insert(0, str(ROOT / "third_party/PaddleDetection"))
        from palm_detector.paddledetection_extensions import install_paddledetection_extensions

        install_paddledetection_extensions()
        from ppdet.data import transform

        sample = {
            "image": np.full((100, 200, 3), 127, dtype=np.uint8),
            "gt_bbox": np.array([[20, 10, 80, 90]], dtype=np.float32),
            "gt_class": np.array([[0]], dtype=np.int32),
        }
        flipped = transform.PalmHandednessFlip(prob=1.0)(sample)
        self.assertEqual(flipped["gt_class"].tolist(), [[1]])
        np.testing.assert_allclose(flipped["gt_bbox"], [[120, 10, 180, 90]])
        letterboxed = transform.PalmLetterBox(target_size=[320, 320])(flipped)
        np.testing.assert_allclose(letterboxed["gt_bbox"], [[192, 96, 288, 224]])
        np.testing.assert_allclose(letterboxed["letterbox_pad"], [80, 0])

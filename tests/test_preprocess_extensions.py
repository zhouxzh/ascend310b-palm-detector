from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


class PreprocessExtensionsTest(unittest.TestCase):
    def test_surface_recipe_uses_label_preserving_flip(self) -> None:
        config = yaml.safe_load((ROOT / "configs/nanodet-plus-m-320-surface3-letterbox-v004.yml").read_text(encoding="utf-8"))
        pipeline = config["data"]["train"]["pipeline"]
        self.assertEqual(pipeline["flip"], 0.5)
        self.assertTrue(pipeline["palm_letterbox"])
        self.assertNotIn("label_remap", pipeline)

    def test_letterbox_keeps_surface_label_ids(self) -> None:
        sys.path.insert(0, str(ROOT / "third_party/nanodet"))
        from palm_detector.nanodet_extensions import install_nanodet_extensions

        install_nanodet_extensions()
        from nanodet.data.transform.warp import ShapeTransform

        transform = ShapeTransform(keep_ratio=False, palm_letterbox=True, scale=(1.0, 1.0), stretch=((1.0, 1.0), (1.0, 1.0)), rotation=0, shear=0, perspective=0, translate=0, flip=1.0)
        sample = {"img": np.full((100, 200, 3), 127, dtype=np.uint8), "gt_bboxes": np.array([[20, 10, 80, 90]], dtype=np.float32), "gt_labels": np.array([2], dtype=np.int64)}
        result = transform(sample, (320, 320))
        self.assertEqual(result["gt_labels"].tolist(), [2])
        np.testing.assert_allclose(result["gt_bboxes"], [[192, 96, 288, 224]], atol=1e-5)

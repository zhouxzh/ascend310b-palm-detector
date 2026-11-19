from __future__ import annotations

import unittest

import numpy as np

from palm_detector.letterbox import compute_letterbox_transform
from palm_detector.picodet_postprocess import decode_picodet, preprocess_picodet_bgr_letterbox


def _dfl(locations: int, *, selected: int, distance_bin: int, reg_max: int = 7) -> np.ndarray:
    values = np.full((1, locations, 4, reg_max + 1), -20.0, dtype=np.float32)
    values[:, :, :, 0] = 20.0
    values[0, selected, :, :] = -20.0
    values[0, selected, :, distance_bin] = 20.0
    return values.reshape(1, locations, -1)


class PicoDetPostprocessTest(unittest.TestCase):
    def test_preprocess_converts_bgr_to_rgb_before_imagenet_normalization(self) -> None:
        image = np.array([[[0, 0, 255]]], dtype=np.uint8)
        tensor, transform = preprocess_picodet_bgr_letterbox(
            image, input_size=(1, 1), mean_rgb=(0.0, 0.0, 0.0), std_rgb=(1.0, 1.0, 1.0)
        )
        self.assertEqual((transform.pad_top, transform.pad_left), (0, 0))
        np.testing.assert_allclose(tensor, np.array([[[[1.0]], [[0.0]], [[0.0]]]], dtype=np.float32))

    def test_decodes_raw_per_level_probabilities_and_dfl_at_half_cell_centers(self) -> None:
        classes0 = np.zeros((1, 16, 2), dtype=np.float32)
        classes1 = np.zeros((1, 4, 2), dtype=np.float32)
        classes0[0, 0, 1] = 0.95
        result = decode_picodet(
            [classes0, classes1, _dfl(16, selected=0, distance_bin=1), _dfl(4, selected=0, distance_bin=1)],
            class_names=("palm", "back"), input_size=(32, 32), strides=(8, 16), score_threshold=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        detection = result[0][0]
        self.assertEqual(detection.class_name, "back")
        self.assertAlmostEqual(detection.score, 0.95)
        np.testing.assert_allclose(detection.xyxy, (0.0, 0.0, 12.0, 12.0), atol=1e-4)

    def test_unmaps_zero_padded_letterbox_and_rejects_malformed_outputs(self) -> None:
        classes0 = np.zeros((1, 16, 1), dtype=np.float32)
        classes1 = np.zeros((1, 4, 1), dtype=np.float32)
        classes0[0, 5, 0] = 0.95
        transform = compute_letterbox_transform(100, 200, 32, 32)
        result = decode_picodet(
            [classes0, classes1, _dfl(16, selected=5, distance_bin=1), _dfl(4, selected=0, distance_bin=1)],
            class_names=("palm",), input_size=(32, 32), strides=(8, 16), score_threshold=0.9, letterbox=transform,
        )
        np.testing.assert_allclose(result[0][0].xyxy, (25.0, 0.0, 125.0, 75.0), atol=1e-4)
        with self.assertRaisesRegex(ValueError, "class and DFL"):
            decode_picodet(
                [classes0, classes1, _dfl(16, selected=0, distance_bin=1), np.zeros((1, 3, 32), dtype=np.float32)],
                class_names=("palm",), input_size=(32, 32), strides=(8, 16),
            )

from __future__ import annotations

import unittest

import numpy as np

from palm_detector.letterbox import compute_letterbox_transform, letterbox_bgr
from palm_detector.postprocess import center_priors, decode_nanodet_plus, preprocess_bgr_letterbox


class LetterboxTest(unittest.TestCase):
    def test_geometry_preserves_aspect_ratio_and_round_trips_boxes(self) -> None:
        transform = compute_letterbox_transform(100, 200, 320, 320)
        self.assertEqual((transform.resized_height, transform.resized_width), (160, 320))
        self.assertEqual((transform.pad_top, transform.pad_left), (80, 0))
        boxes = np.array([[20.0, 10.0, 180.0, 90.0]], dtype=np.float32)
        np.testing.assert_allclose(transform.map_boxes(boxes), [[32.0, 96.0, 288.0, 224.0]])
        np.testing.assert_allclose(transform.unmap_boxes(transform.map_boxes(boxes)), boxes)

    def test_black_padding_is_normalized_like_media_pipe_border(self) -> None:
        image = np.full((100, 200, 3), 200, dtype=np.uint8)
        canvas, transform = letterbox_bgr(image, target_size=(320, 320))
        self.assertTrue(np.all(canvas[:transform.pad_top] == 0))
        tensor, from_preprocess = preprocess_bgr_letterbox(image, input_size=(320, 320))
        self.assertEqual(transform, from_preprocess)
        expected_black = -np.array([103.53, 116.28, 123.675], dtype=np.float32) / np.array([57.375, 57.12, 58.395], dtype=np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected_black)

    def test_decode_removes_padding_before_returning_source_coordinates(self) -> None:
        transform = compute_letterbox_transform(100, 200, 320, 320)
        output = np.zeros((1, 2125, 34), dtype=np.float32)
        index = 10 * 40 + 20
        output[0, index, 0] = 0.99
        output[0, index, 2:] = -50
        output[0, index, [3, 11, 19, 27]] = 50
        detection = decode_nanodet_plus(output, score_threshold=0.9, letterbox=transform)[0][0]
        prior = center_priors()[index]
        expected = transform.unmap_boxes(np.array([[prior[0] - 8, prior[1] - 8, prior[0] + 8, prior[1] + 8]], dtype=np.float32))[0]
        np.testing.assert_allclose(detection.xyxy, expected, atol=1e-5)

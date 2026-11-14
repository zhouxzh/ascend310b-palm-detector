from __future__ import annotations

import unittest

import numpy as np

from palm_detector.postprocess import INPUT_HEIGHT, INPUT_WIDTH, center_priors, decode_nanodet_plus


class PostprocessTest(unittest.TestCase):
    def test_static_prior_count_and_dfl_decode(self) -> None:
        priors = center_priors()
        self.assertEqual(priors.shape, (2125, 3))
        np.testing.assert_array_equal(priors[0], [0, 0, 8])
        np.testing.assert_array_equal(priors[41], [8, 8, 8])
        np.testing.assert_array_equal(priors[1600], [0, 0, 16])
        np.testing.assert_array_equal(priors[2000], [0, 0, 32])
        np.testing.assert_array_equal(priors[2100], [0, 0, 64])
        output = np.zeros((1, 2125, 34), dtype=np.float32)
        output[0, 0, 0] = 0.95
        detections = decode_nanodet_plus(output, score_threshold=0.9, nms_iou_threshold=0.6)
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(detections[0]), 1)
        detection = detections[0][0]
        self.assertEqual(detection.class_name, "left_palm")
        self.assertAlmostEqual(detection.xyxy[0], 0.0)
        self.assertAlmostEqual(detection.xyxy[1], 0.0)
        self.assertAlmostEqual(detection.xyxy[2], 28.0)
        self.assertAlmostEqual(detection.xyxy[3], 28.0)

    def test_bad_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_nanodet_plus(np.zeros((1, 1, 34), dtype=np.float32))

    def test_single_class_onnx_output_uses_its_own_label(self) -> None:
        output = np.zeros((1, 2125, 33), dtype=np.float32)
        output[0, 0, 0] = 0.95
        detections = decode_nanodet_plus(output, class_names=("palm",), score_threshold=0.9)
        self.assertEqual([item.class_name for item in detections[0]], ["palm"])

    def test_decoded_box_uses_upstream_grid_origin_at_each_stride(self) -> None:
        for index, stride in ((41, 8), (1621, 16), (2011, 32), (2106, 64)):
            with self.subTest(stride=stride):
                output = np.zeros((1, 2125, 34), dtype=np.float32)
                output[0, index, 0] = 0.99
                output[0, index, 2:] = -50
                output[0, index, [2, 10, 18, 26]] = 50
                detected = decode_nanodet_plus(output, score_threshold=0.9, original_size=(640, 480))[0][0]
                prior = center_priors()[index]
                expected = (prior[0] * 480 / 320, prior[1] * 640 / 320)
                self.assertAlmostEqual(detected.xyxy[0], expected[0], places=4)
                self.assertAlmostEqual(detected.xyxy[1], expected[1], places=4)

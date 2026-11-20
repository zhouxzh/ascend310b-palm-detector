from __future__ import annotations

import unittest

import cv2
import numpy as np

from palm_detector.postprocess import INPUT_HEIGHT, INPUT_WIDTH, center_priors, decode_nanodet_plus, preprocess_bgr


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

    def test_multisize_preprocessing_and_prior_counts(self) -> None:
        image = np.zeros((800, 400, 3), dtype=np.uint8)
        for size, plus_count, legacy_count in ((320, 2125, 2100), (416, 3598, 3549), (512, 5440, 5376), (640, 8500, 8400)):
            with self.subTest(size=size):
                tensor, scale = preprocess_bgr(image, input_size=(size, size))
                self.assertEqual(tensor.shape, (1, 3, size, size))
                self.assertEqual(scale, (size / 400, size / 800))
                self.assertEqual(center_priors(size, size).shape, (plus_count, 3))
                self.assertEqual(center_priors(size, size, strides=(8, 16, 32), center_offset=0.5).shape, (legacy_count, 3))

    def test_preprocessing_respects_configured_normalization(self) -> None:
        image = np.full((32, 16, 3), (127, 128, 129), dtype=np.uint8)
        tensor, _ = preprocess_bgr(image, input_size=(416, 416), mean=(127, 128, 129), std=(128, 128, 128))
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        # The direct-stretch path mirrors NanoDet's warpPerspective sampling;
        # verify an in-bounds source-aligned pixel rather than border pixels.
        np.testing.assert_array_equal(tensor[0, :, 0, 0], np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "mean and std"):
            preprocess_bgr(image, std=(1, 0, 1))

    def test_direct_stretch_matches_nanodet_warp_sampling(self) -> None:
        image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        tensor, scales = preprocess_bgr(image, input_size=(4, 8), mean=(0, 0, 0), std=(1, 1, 1))
        matrix = np.array([[8 / 5, 0.0, 0.0], [0.0, 4 / 3, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        expected = cv2.warpPerspective(image, matrix, dsize=(8, 4)).astype(np.float32).transpose(2, 0, 1)[None, ...]
        np.testing.assert_array_equal(tensor, expected)
        self.assertEqual(scales, (8 / 5, 4 / 3))

    def test_legacy_half_cell_origin_and_ten_bin_regression(self) -> None:
        size = 416
        strides = (8, 16, 32)
        priors = center_priors(size, size, strides=strides, center_offset=0.5)
        np.testing.assert_array_equal(priors[0], [4, 4, 8])
        np.testing.assert_array_equal(priors[52 * 52], [8, 8, 16])
        output = np.zeros((1, len(priors), 2 + 4 * 11), dtype=np.float32)
        index = 100
        output[0, index, 0] = 0.95
        output[0, index, 2:] = -50
        output[0, index, [2, 13, 24, 35]] = 50
        result = decode_nanodet_plus(
            output, input_size=(size, size), strides=strides, reg_max=10,
            center_offset=0.5, score_threshold=0.9, original_size=(832, 624),
        )[0][0]
        self.assertAlmostEqual(result.xyxy[0], priors[index, 0] * 624 / size)
        self.assertAlmostEqual(result.xyxy[1], priors[index, 1] * 832 / size)
        with self.assertRaisesRegex(ValueError, "expected"):
            decode_nanodet_plus(output, input_size=(size, size), strides=strides, reg_max=7, center_offset=0.5)

    def test_plus_640_coordinates_rescale_to_source_image(self) -> None:
        size = 640
        priors = center_priors(size, size)
        index = 95
        output = np.zeros((1, len(priors), 34), dtype=np.float32)
        output[0, index, 1] = 0.99
        output[0, index, 2:] = -50
        output[0, index, [2, 10, 18, 26]] = 50
        result = decode_nanodet_plus(output, input_size=(size, size), original_size=(960, 1280), score_threshold=0.9)[0][0]
        self.assertEqual(result.class_name, "right_palm")
        self.assertAlmostEqual(result.xyxy[0], priors[index, 0] * 2)
        self.assertAlmostEqual(result.xyxy[1], priors[index, 1] * 1.5)

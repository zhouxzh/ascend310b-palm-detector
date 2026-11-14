from __future__ import annotations

import unittest

from palm_detector.task_metrics import task_metrics


class TaskMetricsTest(unittest.TestCase):
    def test_wrong_handedness_and_duplicates_are_not_hidden(self) -> None:
        ground_truth = {
            "images": [{"id": 1, "device": "camera-a", "lighting": "normal", "occlusion": "none", "subject_id": "subject-a"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}],
        }
        predictions = [
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        ]
        result = task_metrics(ground_truth, predictions)
        counts = result["groups"]["all"]["all"]
        self.assertEqual(counts["matched"], 1)
        self.assertEqual(counts["wrong_handedness"], 1)
        self.assertEqual(counts["duplicate"], 1)
        self.assertEqual(counts["missed"], 0)

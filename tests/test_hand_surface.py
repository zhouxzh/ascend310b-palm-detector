from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("annotate_hand_surface", scripts / "annotate_hand_surface.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(scripts), *sys.path]):
        spec.loader.exec_module(module)
    return module


def point(x: float, y: float, z: float = 0.0):
    return SimpleNamespace(x=x, y=y, z=z)


class HandSurfaceTest(unittest.TestCase):
    def test_surface_reverses_with_anatomical_handedness(self) -> None:
        module = _module()
        points = [point(0, 0) for _ in range(21)]
        points[5], points[17] = point(1, 0), point(0, 1)
        self.assertEqual(module.surface_from_world_landmarks(points, 1, 0.7, 0.2)[0], "palm")
        self.assertEqual(module.surface_from_world_landmarks(points, 2, 0.7, 0.2)[0], "back")
        points[17] = point(0, 0, 1)
        self.assertEqual(module.surface_from_world_landmarks(points, 1, 0.7, 0.2)[0], "side")
        self.assertEqual(module.surface_from_world_landmarks([], 1, 0.7, 0.2), ("unknown", None))

    def test_saved_instance_matches_only_the_same_detected_hand(self) -> None:
        module = _module()
        left = [[0.2, 0.4] for _ in range(21)]
        right = [[0.8, 0.4] for _ in range(21)]
        instances = [
            {"mediapipe": {"palm_bbox_xyxy": [0.1, 0.2, 0.3, 0.6], "full21_keypoints": left}},
            {"mediapipe": {"palm_bbox_xyxy": [0.7, 0.2, 0.9, 0.6], "full21_keypoints": right}},
        ]
        annotations = [
            {"bbox": [10, 20, 20, 40], "category_id": 1},
            {"bbox": [70, 20, 20, 40], "category_id": 2},
        ]
        result = SimpleNamespace(
            hand_landmarks=[[point(*xy) for xy in right], [point(*xy) for xy in left]],
            hand_world_landmarks=[[point(0, 0) for _ in range(21)] for _ in range(2)],
            handedness=[
                [SimpleNamespace(category_name="Right")],
                [SimpleNamespace(category_name="Left")],
            ],
        )
        image = {"width": 100, "height": 100}
        self.assertEqual(module._match_instances(annotations, instances, image, result, 0.1), [1, 0])
        result.handedness[1][0].category_name = "Right"
        self.assertEqual(module._match_instances(annotations, instances, image, result, 0.1), [None, 0])

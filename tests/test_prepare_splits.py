from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


class PrepareSplitsTest(unittest.TestCase):
    def test_independent_balanced_components_follow_requested_ratios(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_splits.py"
        spec = importlib.util.spec_from_file_location("prepare_splits", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for targets in ({"train": 70, "val": 15, "test": 15}, {"train": 80, "val": 10, "test": 10}):
            with self.subTest(targets=targets):
                names = list(targets)
                allocated_images = dict.fromkeys(names, 0)
                allocated_classes = {name: Counter() for name in names}
                target_classes = {name: Counter({1: count, 2: count}) for name, count in targets.items()}
                for image_id in range(100):
                    counts = Counter({1: 1, 2: 1})
                    name = module._choose_split(
                        [{"id": image_id}], counts, allocated_images, allocated_classes,
                        targets, target_classes, names,
                    )
                    allocated_images[name] += 1
                    allocated_classes[name].update(counts)
                self.assertEqual(allocated_images, targets)
                self.assertEqual(allocated_classes, target_classes)

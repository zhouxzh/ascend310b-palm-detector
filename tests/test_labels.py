from __future__ import annotations

import unittest

from palm_detector.labels import CLASS_NAMES, SURFACE_CATEGORY_IDS, validate_categories


class LabelsTest(unittest.TestCase):
    def test_surface_contract_has_fixed_order(self) -> None:
        self.assertEqual(CLASS_NAMES, ("palm", "back", "side"))
        self.assertEqual(SURFACE_CATEGORY_IDS, {1: "palm", 2: "back", 3: "side"})
        self.assertEqual(validate_categories([
            {"id": 3, "name": "side"}, {"id": 1, "name": "palm"}, {"id": 2, "name": "back"},
        ]), [])

    def test_incomplete_categories_are_rejected(self) -> None:
        self.assertTrue(validate_categories([{"id": 1, "name": "palm"}, {"id": 2, "name": "back"}]))

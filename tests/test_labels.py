from __future__ import annotations

import unittest

from palm_detector.labels import CLASS_NAMES, validate_categories


class LabelsTest(unittest.TestCase):
    def test_contract_has_fixed_anatomical_order(self) -> None:
        self.assertEqual(CLASS_NAMES, ("left_palm", "right_palm"))
        self.assertEqual(validate_categories([{"id": 2, "name": "right_palm"}, {"id": 1, "name": "left_palm"}]), [])

    def test_single_palm_category_is_rejected(self) -> None:
        self.assertTrue(validate_categories([{"id": 1, "name": "palm"}]))

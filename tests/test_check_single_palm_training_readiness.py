from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CheckSinglePalmTrainingReadinessTest(unittest.TestCase):
    def test_script_imports_without_side_effects(self) -> None:
        path = ROOT / "scripts/check_single_palm_training_readiness.py"
        spec = importlib.util.spec_from_file_location("check_single_palm_training_readiness", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

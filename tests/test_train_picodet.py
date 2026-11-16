from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class TrainPicoDetTest(unittest.TestCase):
    def test_newest_resume_prefix_selects_largest_complete_epoch(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/train_picodet.py"
        spec = importlib.util.spec_from_file_location("train_picodet", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for epoch in (5, 15):
                (root / f"{epoch}.pdopt").write_bytes(b"optimizer")
                (root / f"{epoch}.pdparams").write_bytes(b"model")
            (root / "20.pdopt").write_bytes(b"incomplete")
            self.assertEqual(module.newest_resume_prefix(root), root / "15")

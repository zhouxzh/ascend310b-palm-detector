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

    def test_failed_precheckpoint_run_can_sync_a_repaired_config(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "train_picodet.py"
        spec = importlib.util.spec_from_file_location("train_picodet_config_sync", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.yml"
            copied = root / "run" / "config.yml"
            copied.parent.mkdir()
            source.write_text("batch_size: 32\n", encoding="utf-8")
            copied.write_text("batch_size: 48\n", encoding="utf-8")
            save_dir = copied.parent / "paddle-output"
            save_dir.mkdir()
            module.synchronize_run_config(source, copied, save_dir)
            self.assertEqual(copied.read_text(encoding="utf-8"), "batch_size: 32\n")
            (save_dir / "5.pdparams").write_bytes(b"checkpoint")
            source.write_text("batch_size: 24\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "after checkpoints exist"):
                module.synchronize_run_config(source, copied, save_dir)

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CreateRunTest(unittest.TestCase):
    def test_runtime_lock_placeholders_can_be_initialized(self) -> None:
        spec = importlib.util.spec_from_file_location("create_run", ROOT / "scripts" / "create_run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.assertTrue(module.is_reusable_run_directory(run_dir))
            run_dir.mkdir()
            for name in (".run.lock", ".training.lock", "train.log"):
                (run_dir / name).touch()
            self.assertTrue(module.is_reusable_run_directory(run_dir))
            (run_dir / "config.yml").write_text("run: immutable\n", encoding="utf-8")
            self.assertFalse(module.is_reusable_run_directory(run_dir))

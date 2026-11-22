from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainNanoDetSweepTest(unittest.TestCase):
    def test_resume_early_stop_history_accepts_old_and_current_metric_logs(self) -> None:
        path = ROOT / "scripts" / "train_nanodet_sweep.py"
        spec = importlib.util.spec_from_file_location("train_nanodet_sweep", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                "[NanoDet]INFO:Val_metrics: {'mAP': np.float64(0.55)}\n"
                "[NanoDet]INFO:Val_metrics: {'mAP': 0.60}\n"
                "[NanoDet]INFO:Val_metrics: {'mAP': 0.59}\n",
                encoding="utf-8",
            )
            self.assertEqual(module._historical_bad_checks(log, 0.60, 0.001), 1)

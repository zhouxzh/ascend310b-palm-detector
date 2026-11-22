from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class SummarizeSinglePalmModelsTest(unittest.TestCase):
    def test_summary_reports_expected_coverage_and_paired_geometry(self) -> None:
        path = ROOT / "scripts/summarize_single_palm_models.py"
        spec = importlib.util.spec_from_file_location("summarize_single_palm_models", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for geometry in ("letterbox", "direct_stretch"):
                folder = root / "nanodet"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"variant-inventory-{geometry}.json").write_text(json.dumps({"variants": [{"id": "m"}]}), encoding="utf-8")
                run = folder / "runs" / f"single-palm-v001-{geometry}-m__data-v001__seed-42"
                run.mkdir(parents=True)
                (run / "training-summary.json").write_text(json.dumps({"status": "early_stopped", "best_dev_map": 0.5, "early_stop": {"bad_checks": 8}}), encoding="utf-8")
                (run / "onnx-check.json").write_text(json.dumps({"valid": True, "onnx": "model.onnx", "sha256": "x"}), encoding="utf-8")
                test = run / "evaluations" / "test"
                test.mkdir(parents=True)
                (test / "coco-metrics.json").write_text(json.dumps({"metrics": {"mAP50_95": 0.6 if geometry == "letterbox" else 0.5, "AP50": 0.9}, "per_class": {"palm_surface": {}}, "ground_truth": {}}), encoding="utf-8")
            output = root / "summary.json"
            with mock.patch("sys.argv", ["summarize_single_palm_models.py", "--experiment-root", str(root), "--output", str(output)]):
                self.assertEqual(module.main(), 0)
            report = json.loads(output.read_text())
        self.assertEqual(report["expected_task_count"], 2)
        self.assertEqual(report["complete_task_count"], 2)
        self.assertEqual(report["paired_geometry"][0]["letterbox_map"], 0.6)
        self.assertEqual(report["paired_geometry"][0]["direct_stretch_map"], 0.5)

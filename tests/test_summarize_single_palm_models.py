from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class SummarizeSinglePalmModelsTest(unittest.TestCase):
    def _load_module(self):
        path = ROOT / "scripts/summarize_single_palm_models.py"
        spec = importlib.util.spec_from_file_location("summarize_single_palm_models", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_summary_reports_expected_coverage_and_paired_geometry(self) -> None:
        module = self._load_module()
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
                (run / "onnx-comparison.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
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
        aggregate = {(item["framework"], item["geometry"]): item for item in report["geometry_aggregate"]}
        self.assertEqual(aggregate[("NanoDet", "letterbox")]["mean_map50_95"], 0.6)
        self.assertEqual(aggregate[("NanoDet", "direct_stretch")]["median_map50_95"], 0.5)
        self.assertEqual(aggregate[("NanoDet", "paired")]["letterbox_wins"], 1)
        self.assertEqual(aggregate[("NanoDet", "paired")]["preferred_geometry"], "LetterBox")

    def test_scope_excludes_historical_matrix_results(self) -> None:
        module = self._load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for geometry, variant, score in (
                ("letterbox", "plus-m-320", 0.6),
                ("direct_stretch", "plus-m-320", 0.5),
            ):
                run = root / "nanodet" / "runs" / f"single-palm-v001-{geometry}-{variant}__data-v001__seed-42"
                run.mkdir(parents=True)
                (run / "training-summary.json").write_text(json.dumps({"status": "early_stopped"}), encoding="utf-8")
                (run / "onnx-check.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
                (run / "onnx-comparison.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
                test = run / "evaluations" / "test"
                test.mkdir(parents=True)
                (test / "coco-metrics.json").write_text(json.dumps({"metrics": {"mAP50_95": score, "AP50": 0.9}}), encoding="utf-8")
            scope = root / "scope.json"
            scope.write_text(json.dumps({"tasks": [{"family": "nanodet", "geometry": "letterbox", "variant": "plus-m-320"}]}), encoding="utf-8")
            output = root / "summary.json"
            with mock.patch("sys.argv", ["summarize_single_palm_models.py", "--experiment-root", str(root), "--scope", str(scope), "--output", str(output)]):
                self.assertEqual(module.main(), 0)
            report = json.loads(output.read_text())
        self.assertEqual(report["expected_task_count"], 1)
        self.assertEqual(report["complete_task_count"], 1)
        self.assertEqual(report["completed"][0]["geometry"], "letterbox")

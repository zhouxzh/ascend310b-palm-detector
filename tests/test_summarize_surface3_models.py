from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class SummarizeSurface3ModelsTest(unittest.TestCase):
    def test_summarize_writes_consistent_table_for_nanodet_and_picodet(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/summarize_surface3_models.py"
        spec = importlib.util.spec_from_file_location("summarize_surface3_models", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nano = root / "nano"
            pico = root / "pico"
            for directory, pico_flag in ((nano, False), (pico, True)):
                (directory / ("test-eval" if pico_flag else "evaluations/test")).mkdir(parents=True)
                (directory / "training-summary.json").write_text(json.dumps({
                    "status": "early_stopped", "best_dev_map": 0.5,
                    "early_stop": {"bad_checks": 8},
                }))
                (directory / ("picodet-onnx-check.json" if pico_flag else "onnx-check.json")).write_text(json.dumps({
                    "valid": True, "onnx": "/tmp/model.onnx", "sha256": "a" * 64,
                    "inputs": [], "outputs": [],
                }))
                (directory / ("test-eval" if pico_flag else "evaluations/test") / "coco-metrics.json").write_text(json.dumps({
                    "metrics": {"mAP50_95": 0.5, "AP50": 0.8},
                    "per_class": {name: {"AP50": value} for name, value in (("palm", 0.9), ("back", 0.8), ("side", 0.1))},
                    "ground_truth": {"path": "test.json"},
                }))
            nano_summary = module.summarize("nano", nano)
            pico_summary = module.summarize("pico", pico)
            self.assertEqual((nano_summary["framework"], pico_summary["framework"]), ("NanoDet", "PicoDet"))
            rendered = module.markdown({"models": [nano_summary, pico_summary]})
            self.assertIn("| nano | NanoDet |", rendered)
            self.assertIn("| pico | PicoDet |", rendered)


if __name__ == "__main__":
    unittest.main()

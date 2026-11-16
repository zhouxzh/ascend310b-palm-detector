from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class PicoDetPostSweepTest(unittest.TestCase):
    def test_scale_path_handles_nested_optimizer_value(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_picodet_tuning.py"
        spec = importlib.util.spec_from_file_location("prepare_picodet_tuning", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {"LearningRate": {"base_lr": 0.08}, "OptimizerBuilder": {"regularizer": {"factor": 0.00004}}}
        module.scale_path(payload, "LearningRate.base_lr", 0.75)
        module.scale_path(payload, "OptimizerBuilder.regularizer.factor", 2.0)
        self.assertEqual(payload["LearningRate"]["base_lr"], 0.06)
        self.assertEqual(payload["OptimizerBuilder"]["regularizer"]["factor"], 0.00008)

    def test_ranking_orders_completed_runs_by_map(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/rank_picodet_sweep.py"
        spec = importlib.util.spec_from_file_location("rank_picodet_sweep", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"variants": {}}
            for variant, score in (("b", 0.7), ("a", 0.8)):
                run = root / variant
                run.mkdir()
                (run / "training-summary.json").write_text(json.dumps({"status": "early_stopped", "best_dev_map": score}))
                state["variants"][variant] = {"status": "trained", "run_dir": str(run)}
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state))
            output = root / "ranking.json"
            with patch("sys.argv", ["rank", "--state", str(state_path), "--output", str(output)]):
                self.assertEqual(module.main(), 0)
            self.assertEqual([item["variant"] for item in json.loads(output.read_text())["ranking"]], ["a", "b"])

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
import yaml
from onnx import TensorProto, helper


class VerifyOnnxMultisizeTest(unittest.TestCase):
    def test_config_driven_plus_and_legacy_output_contracts(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/verify_onnx.py"
        spec = importlib.util.spec_from_file_location("verify_onnx_multisize", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = (
            (640, (8, 16, 32, 64), 7, 8500, 34),
            (512, (8, 16, 32), 10, 5376, 46),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size, strides, reg_max, priors, channels in cases:
                with self.subTest(size=size, reg_max=reg_max):
                    config = root / f"config-{size}.yml"
                    config.write_text(yaml.safe_dump({
                        "data": {
                            "train": {"input_size": [size, size]},
                            "val": {"input_size": [size, size]},
                        },
                        "model": {"arch": {"head": {
                            "num_classes": 2, "strides": list(strides), "reg_max": reg_max,
                        }}},
                    }))
                    model_path = root / f"model-{size}.onnx"
                    graph = helper.make_graph(
                        [helper.make_node("ConstantOfShape", ["shape"], ["output"])],
                        f"nanodet-{size}",
                        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 3, size, size])],
                        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, priors, channels])],
                        [helper.make_tensor("shape", TensorProto.INT64, [3], [1, priors, channels])],
                    )
                    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)]), model_path)
                    report_path = root / f"report-{size}.json"
                    with patch.object(sys, "argv", ["verify_onnx.py", "--config", str(config), "--onnx", str(model_path), "--output", str(report_path)]):
                        self.assertEqual(module.main(), 0)
                    report = json.loads(report_path.read_text())
                    self.assertTrue(report["valid"])
                    self.assertEqual(report["outputs"], [{"name": "output", "shape": [1, priors, channels]}])
                    self.assertEqual(report["strides"], list(strides))
                    self.assertEqual(report["reg_max"], reg_max)

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExportNanoDetOnnxTest(unittest.TestCase):
    def test_exporter_preserves_opset_11_then_falls_back_for_unflatten(self) -> None:
        source = (ROOT / "scripts" / "export_nanodet_onnx.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)}
        self.assertIn("export_with_compatible_opset", functions)
        function_source = ast.get_source_segment(source, functions["export_with_compatible_opset"])
        self.assertIn("opset_version=11", function_source)
        self.assertIn("aten::unflatten", function_source)
        self.assertIn("opset_version=13", function_source)

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class EvaluatePicoDetTest(unittest.TestCase):
    def test_success_path_uses_immediate_exit_after_prediction_is_written(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate_picodet.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
            and node.func.attr == "_exit"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].value, 0)


if __name__ == "__main__":
    unittest.main()

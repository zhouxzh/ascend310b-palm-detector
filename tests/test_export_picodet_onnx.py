from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class ExportPicoDetOnnxTest(unittest.TestCase):
    def test_preprocessing_contract_requires_the_deployment_transforms(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/export_picodet_onnx.py"
        spec = importlib.util.spec_from_file_location("export_picodet_onnx", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = {
            "TestReader": {"sample_transforms": [
                {"Decode": {}},
                {"PalmLetterBox": {"target_size": [416, 416], "pad_value": 0}},
                {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"Permute": {}},
            ]},
        }
        self.assertEqual(module.preprocessing_contract(config), {
            "decode_color": "RGB (OpenCV BGR decode converted by PaddleDetection Decode)",
            "geometry": "centered_zero_padded_letterbox",
            "target_size": [416, 416],
            "pad_value": 0,
            "keep_ratio": True,
            "normalize": {"scale": 1.0 / 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "layout": "NCHW",
        })
        with self.assertRaisesRegex(ValueError, "requires NormalizeImage"):
            module.preprocessing_contract({"TestReader": {"sample_transforms": []}})

    def test_pir_converter_copy_only_rewrites_resize_scale_constants(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/export_picodet_onnx.py"
        spec = importlib.util.spec_from_file_location("export_picodet_onnx", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {
            "program": {
                "regions": [{"blocks": [{"ops": [
                    {"#": "1.nearest_interp", "A": [{"N": "scale", "AT": {"D": [
                        {"#": "0.a_f64", "D": 2.0}, {"#": "0.a_f64", "D": 2.0}
                    ]}}]},
                    {"#": "1.nearest_interp", "A": [{"N": "scale", "AT": {"D": [
                        {"#": "0.a_f64", "D": 2.0}, {"#": "0.a_f64", "D": 2.0}
                    ]}}]},
                    {"#": "1.full", "A": [{"N": "value", "AT": {"D": 1.0}}]},
                ]}]}]
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.json"
            params = root / "model.pdiparams"
            model.write_text(json.dumps(payload), encoding="utf-8")
            params.write_bytes(b"params")
            copied, patch_count = module.prepare_pir_for_paddle2onnx(model, root / "conversion")
            self.assertEqual(patch_count, 4)
            self.assertEqual(json.loads(model.read_text(encoding="utf-8")), payload)
            converted = json.loads(copied.read_text(encoding="utf-8"))
            scales = [
                value["#"]
                for op in converted["program"]["regions"][0]["blocks"][0]["ops"]
                if op["#"] == "1.nearest_interp"
                for value in op["A"][0]["AT"]["D"]
            ]
            self.assertEqual(scales, ["0.a_f32"] * 4)
            self.assertEqual((root / "conversion/model.pdiparams").read_bytes(), b"params")

    def test_preprocessing_contract_supports_direct_stretch_without_letterbox(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/export_picodet_onnx.py"
        spec = importlib.util.spec_from_file_location("export_picodet_onnx_stretch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = {
            "TestReader": {"sample_transforms": [
                {"Decode": {}},
                {"Resize": {"interp": 2, "target_size": [320, 320], "keep_ratio": False}},
                {"NormalizeImage": {"is_scale": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"Permute": {}},
            ]},
        }
        report = module.preprocessing_contract(config)
        self.assertEqual(report["geometry"], "direct_stretch")
        self.assertEqual(report["target_size"], [320, 320])
        self.assertEqual(report["interp"], 2)
        self.assertFalse(report["keep_ratio"])

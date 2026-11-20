from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PrepareSurface3StretchConfigsTest(unittest.TestCase):
    def test_direct_stretch_disables_only_letterbox_and_random_anisotropic_stretch(self) -> None:
        path = ROOT / "scripts/prepare_surface3_stretch_configs.py"
        spec = importlib.util.spec_from_file_location("prepare_surface3_stretch_configs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = yaml.safe_load((ROOT / "configs/nanodet-plus-m-320-surface3-letterbox-v004.yml").read_text(encoding="utf-8"))
        direct = module.build_config(source, input_size=320)
        self.assertEqual(direct["class_names"], ["palm", "back", "side"])
        self.assertFalse(direct["data"]["train"]["pipeline"]["palm_letterbox"])
        self.assertFalse(direct["data"]["val"]["pipeline"]["palm_letterbox"])
        self.assertEqual(direct["data"]["train"]["pipeline"]["stretch"], [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(direct["data"]["train"]["pipeline"]["flip"], 0.5)
        self.assertFalse(direct["data"]["train"]["keep_ratio"])
        self.assertEqual(direct["model"], source["model"])
        self.assertEqual(direct["schedule"], source["schedule"])

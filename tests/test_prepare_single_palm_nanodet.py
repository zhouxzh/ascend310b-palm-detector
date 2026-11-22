from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareSinglePalmNanoDetTest(unittest.TestCase):
    def test_configure_updates_class_contract_and_geometry_only(self) -> None:
        path = ROOT / "scripts/prepare_single_palm_nanodet.py"
        spec = importlib.util.spec_from_file_location("prepare_single_palm_nanodet", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {
            "model": {"arch": {"backbone": {"name": "EfficientNetLite", "pretrain": True}, "head": {"num_classes": 80}, "aux_head": {"num_classes": 80}}},
            "data": {
                "train": {"img_path": "old", "ann_path": "old", "input_size": [320, 320], "keep_ratio": True, "pipeline": {"normalize": [[1, 2, 3], [4, 5, 6]]}},
                "val": {"img_path": "old", "ann_path": "old", "input_size": [320, 320], "keep_ratio": True, "pipeline": {"normalize": [[1, 2, 3], [4, 5, 6]]}},
            },
            "device": {"batchsize_per_gpu": 64, "workers_per_gpu": 12},
            "schedule": {"optimizer": {"lr": 0.32}, "val_intervals": 10},
        }
        matrix = {"classes": ["palm_surface"], "augmentation": {"flip_probability": 0.5, "scale": [0.9, 1.1], "rotation": 10, "translate": 0.05, "brightness": 0.1, "contrast": [0.8, 1.2], "saturation": [0.8, 1.2]}}
        configured = module.configure(source, {"batch_size": 32}, matrix, Path("/dataset"), "letterbox")
        self.assertEqual(configured["class_names"], ["palm_surface"])
        self.assertEqual(configured["model"]["arch"]["head"]["num_classes"], 1)
        self.assertEqual(configured["model"]["arch"]["aux_head"]["num_classes"], 1)
        self.assertFalse(configured["model"]["arch"]["backbone"]["pretrain"])
        self.assertTrue(configured["data"]["train"]["pipeline"]["palm_letterbox"])
        self.assertFalse(configured["data"]["train"]["keep_ratio"])
        self.assertEqual(configured["schedule"]["optimizer"]["lr"], 0.16)

    def test_configure_disables_implicit_shufflenet_online_pretraining(self) -> None:
        path = ROOT / "scripts/prepare_single_palm_nanodet.py"
        spec = importlib.util.spec_from_file_location("prepare_single_palm_nanodet_shufflenet", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {
            "model": {"arch": {"backbone": {"name": "ShuffleNetV2", "model_size": "1.0x"}, "head": {"num_classes": 80}}},
            "data": {
                "train": {"img_path": "old", "ann_path": "old", "input_size": [320, 320], "keep_ratio": True, "pipeline": {}},
                "val": {"img_path": "old", "ann_path": "old", "input_size": [320, 320], "keep_ratio": True, "pipeline": {}},
            },
            "device": {"batchsize_per_gpu": 64, "workers_per_gpu": 8},
            "schedule": {"optimizer": {"lr": 0.32}, "val_intervals": 10},
        }
        matrix = {"classes": ["palm_surface"], "augmentation": {"flip_probability": 0.5, "scale": [0.9, 1.1], "rotation": 10, "translate": 0.05, "brightness": 0.1, "contrast": [0.8, 1.2], "saturation": [0.8, 1.2]}}
        configured = module.configure(source, {"batch_size": 32}, matrix, Path("/dataset"), "direct_stretch")
        self.assertFalse(configured["model"]["arch"]["backbone"]["pretrain"])

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


class PreparePicoDetSweepTest(unittest.TestCase):
    def test_generated_config_preserves_anatomical_classes(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_picodet_sweep.py"
        spec = importlib.util.spec_from_file_location("prepare_picodet_sweep", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            source = upstream / "configs/picodet/picodet_s_320_coco_lcnet.yml"
            source.parent.mkdir(parents=True)
            source.write_text("architecture: PicoDet\n")
            dataset = root / "dataset"
            annotations = dataset / "annotations"
            annotations.mkdir(parents=True)
            (annotations / "instances_train.json").write_text("{}")
            for split in ("dev", "gate", "test"):
                (root / f"{split}.json").write_text("{}")
            matrix = {
                "reference_gpu_count": 4,
                "single_gpu_count": 1,
                "classes": ["left_palm", "right_palm"],
                "dataset": str(dataset),
                "validation": {split: str(root / f"{split}.json") for split in ("dev", "gate", "test")},
                "seed": 42,
                "validation_interval": 5,
                "early_stop": {"patience": 8, "min_delta": 0.001},
                "pretrained": {"lcnet-x075": {"url": "https://weights.invalid/x075", "filename": "x075.pdparams"}},
            }
            item = {
                "id": "picodet-s-320", "config": "configs/picodet/picodet_s_320_coco_lcnet.yml",
                "input_size": 320, "batch_size": 64, "reference_lr": 0.32, "epochs": 300, "pretrained": "lcnet-x075",
            }
            manifest = module.build_config(item, matrix, upstream)
            generated = manifest["training_config"]
            self.assertEqual(generated["num_classes"], 2)
            self.assertEqual(manifest["pretrained"]["filename"], "x075.pdparams")
            self.assertEqual(generated["TrainReader"]["batch_size"], 64)
            self.assertEqual(generated["LearningRate"]["base_lr"], 0.08)
            transforms = generated["TrainReader"]["sample_transforms"]
            self.assertIn({"RandomFlip": {"prob": 0.0}}, transforms)
            self.assertEqual(generated["TrainReader"]["batch_transforms"][0]["BatchRandomResize"]["target_size"], [320, 320])
            self.assertEqual(generated["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"], [320, 320])

    def test_manifest_and_training_config_are_serializable(self) -> None:
        payload = {"classes": ["left_palm", "right_palm"], "training_config_path": "config.yml"}
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(yaml.safe_load(yaml.safe_dump(payload)), payload)

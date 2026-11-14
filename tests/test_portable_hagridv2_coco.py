from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from palm_detector.dataset_contract import SplitSpec, validate_splits
from palm_detector.labels import SINGLE_PALM_CATEGORY_IDS


@unittest.skipIf(pq is None, "Parquet conversion requires optional pyarrow")
class PortableHagridCocoTest(unittest.TestCase):
    def test_conversion_deduplicates_within_and_across_shards_and_splits(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_portable_hagridv2_coco.py"
        spec = importlib.util.spec_from_file_location("portable_coco", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def row(image_id, color, source_image):
            ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), color, dtype=np.uint8))
            self.assertTrue(ok)
            return {
                "image_id": str(image_id), "image": {"bytes": encoded.tobytes()},
                "width": 16, "height": 16, "source_image": source_image,
                "instances": json.dumps([{"mediapipe": {"palm_bbox_xyxy": [0.1, 0.1, 0.9, 0.9]}}]),
            }

        shards = {
            "train-a.parquet": [row(1, 0, "a.jpg"), row(2, 0, "duplicate-a.jpg")],
            "train-b.parquet": [row(3, 0, "duplicate-a-2.jpg"), row(4, 40, "b.jpg"), row(5, 80, "a.jpg")],
            "val.parquet": [row(6, 0, "duplicate-a-3.jpg"), row(7, 120, "c.jpg"), row(8, 160, "b.jpg")],
            "test.parquet": [row(9, 40, "duplicate-b.jpg"), row(10, 200, "d.jpg")],
        }
        split_shards = {"train": ("train-a.parquet", "train-b.parquet"), "val": ("val.parquet",), "test": ("test.parquet",)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows in shards.items():
                pq.write_table(pa.Table.from_pylist(rows), root / name)
            args = SimpleNamespace(dataset_dir=root, output_dir=root / "output", batch_size=1, overwrite=False)
            with patch.object(module, "parse_args", return_value=args), \
                 patch.object(module, "SPLIT_SHARDS", split_shards), \
                 patch.object(module, "EXPECTED_IMAGES", {"train": 5, "val": 3, "test": 2}), patch("builtins.print"):
                self.assertEqual(module.main(), 0)
            manifest = json.loads((root / "output/source-manifest.json").read_text())
            self.assertEqual(manifest["splits"]["train"]["duplicate_content_dropped"], 2)
            self.assertEqual(manifest["splits"]["train"]["duplicate_source_path_dropped"], 1)
            self.assertEqual(manifest["splits"]["val"]["duplicate_content_dropped"], 1)
            self.assertEqual(manifest["splits"]["val"]["duplicate_source_path_dropped"], 1)
            self.assertEqual(manifest["splits"]["test"]["duplicate_content_dropped"], 1)
            self.assertEqual([manifest["splits"][split]["exported_images"] for split in split_shards], [2, 1, 1])
            self.assertEqual(len(list((root / "output/images").glob("*/*.jpg"))), 4)
            report = validate_splits(
                [SplitSpec(split, root / f"output/annotations/instances_{split}.json", root / "output/images") for split in split_shards],
                category_ids=SINGLE_PALM_CATEGORY_IDS, require_subject_id=False, verify_image_hashes=True,
            )
            self.assertTrue(report["valid"], report["errors"])

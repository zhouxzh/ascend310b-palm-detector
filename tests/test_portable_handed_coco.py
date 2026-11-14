from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from palm_detector.dataset_contract import SplitSpec, sha256_file, validate_splits


class PortableHandedCocoTest(unittest.TestCase):
    def test_two_hands_match_by_landmarks_even_if_detection_order_changes(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "portable_handed", scripts / "prepare_portable_hagridv2_handed_coco.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(module)
        left = [[0.2, 0.3] for _ in range(21)]
        right = [[0.8, 0.3] for _ in range(21)]
        instances = [
            {"mediapipe": {"full21_keypoints": left}},
            {"mediapipe": {"full21_keypoints": right}},
        ]
        result = SimpleNamespace(
            hand_landmarks=[
                [SimpleNamespace(x=x, y=y) for x, y in right],
                [SimpleNamespace(x=x, y=y) for x, y in left],
            ],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.99)],
                [SimpleNamespace(category_name="Left", score=0.98)],
            ],
        )
        self.assertEqual(
            module.match_tasks_handedness(
                instances, result, min_score=0.8, max_landmark_distance=0.1
            ),
            [(1, 0.98, 0.0), (2, 0.99, 0.0)],
        )

    def test_tasks_handedness_is_used_for_the_saved_hand(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "portable_handed", scripts / "prepare_portable_hagridv2_handed_coco.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(module)
        points = [[0.5, 0.5] for _ in range(21)]
        points[0], points[5], points[17] = [0.5, 0.9], [0.3, 0.6], [0.7, 0.6]
        source = {"mediapipe": {"full21_keypoints": points}}
        result = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in points]],
            handedness=[[SimpleNamespace(category_name="Right", score=0.99)]],
        )
        self.assertEqual(
            module.match_tasks_handedness(
                [source], result, min_score=0.8, max_landmark_distance=0.1
            ),
            [(2, 0.99, 0.0)],
        )
        self.assertIsNone(
            module.match_tasks_handedness(
                [{"mediapipe": None}],
                result,
                min_score=0.8,
                max_landmark_distance=0.1,
            )
        )

    def test_tasks_classes_and_unmatched_image_exclusion(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        spec = importlib.util.spec_from_file_location(
            "portable_handed", scripts / "prepare_portable_hagridv2_handed_coco.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec.loader.exec_module(module)

        def instance(category: int | None) -> dict:
            points = [[0.5, 0.5] for _ in range(21)]
            points[0] = [0.5, 0.9]
            points[5] = [0.3 if category == 1 else 0.7, 0.6]
            points[17] = [0.7 if category == 1 else 0.3, 0.6]
            if category is None:
                points[17] = [0.5, 0.6]
                points[5] = [0.5, 0.7]
            return {
                "mediapipe": {
                    "full21_keypoints": points,
                    "palm_bbox_xyxy": [0.1, 0.1, 0.9, 0.9],
                }
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            image_root = root / "images/palm"
            image_root.mkdir(parents=True)
            source_root = root / "source"
            shards = {
                "train": ("train.jsonl",),
                "val": ("valid.jsonl",),
                "test": ("test.jsonl",),
            }
            rows_by_split = {
                "train": [("left", 1), ("right", 2), ("uncertain", None)],
                "val": [("val-left", 1), ("val-right", 2)],
                "test": [("test-left", 1), ("test-right", 2)],
            }
            manifest_shards = []
            for split, rows in rows_by_split.items():
                source_dir = source_root / split
                source_dir.mkdir(parents=True)
                source_annotations = {}
                records = []
                for index, (name, category) in enumerate(rows):
                    color = {"train": 10, "val": 30, "test": 50}[split] + index
                    self.assertTrue(
                        cv2.imwrite(
                            str(image_root / f"{name}.jpg"),
                            np.full((16, 16, 3), color, dtype=np.uint8),
                        )
                    )
                    source_annotations[name] = {"user_id": name}
                    records.append(
                        {
                            "split": split,
                            "source_image": f"palm/{name}.jpg",
                            "width": 16,
                            "height": 16,
                            "instances": [instance(category)],
                        }
                    )
                (source_dir / "palm.json").write_text(json.dumps(source_annotations))
                filename = shards[split][0]
                path = records_dir / filename
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
                manifest_shards.append(
                    {
                        "filename": filename.replace(".jsonl", ".parquet"),
                        "records_sha256": sha256_file(path),
                    }
                )
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "image_provenance": "local HaGRIDv2 512px images matched by source_image path and dimensions; portable Parquet JPEG bytes are not used",
                        "shards": manifest_shards,
                    }
                )
            )
            model = root / "model.task"
            model.write_bytes(b"test-model")
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "mapping": "mediapipe_tasks_Left=anatomical_left;Right=anatomical_right"
                    }
                )
            )
            args = SimpleNamespace(
                metadata_manifest=metadata,
                records_dir=records_dir,
                source_image_root=image_root.parent,
                source_annotations_root=source_root,
                model_path=model,
                calibration_record=calibration,
                output_dir=root / "output",
                min_handedness_score=0.8,
                max_landmark_distance=0.1,
            )

            def result(category: int | None):
                if category is None:
                    return SimpleNamespace(hand_landmarks=[], handedness=[])
                points = instance(category)["mediapipe"]["full21_keypoints"]
                label = "Left" if category == 1 else "Right"
                return SimpleNamespace(
                    hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in points]],
                    handedness=[[SimpleNamespace(category_name=label, score=0.99)]],
                )

            detector = MagicMock()
            detector.detect.side_effect = [
                result(category)
                for rows in rows_by_split.values()
                for _, category in rows
            ]
            mediapipe = MagicMock()
            mediapipe.tasks.vision.HandLandmarker.create_from_options.return_value.__enter__.return_value = (
                detector
            )
            with patch.object(
                module.argparse.ArgumentParser, "parse_args", return_value=args
            ), patch.object(module, "SPLIT_SHARDS", shards), patch.object(
                module,
                "EXPECTED_ROWS",
                {split: len(rows) for split, rows in rows_by_split.items()},
            ), patch.dict(
                sys.modules, {"mediapipe": mediapipe}
            ), patch(
                "builtins.print"
            ):
                self.assertEqual(module.main(), 0)
            output = args.output_dir
            train = json.loads(
                (output / "annotations/instances_train.json").read_text()
            )
            self.assertEqual(
                [item["category_id"] for item in train["annotations"]], [1, 2]
            )
            self.assertEqual(len(train["images"]), 2)
            review = json.loads((output / "handedness-review.json").read_text())
            self.assertEqual(
                [item["source_image"] for item in review], ["palm/uncertain.jpg"]
            )
            report = validate_splits(
                [
                    SplitSpec(
                        split,
                        output / f"annotations/instances_{split}.json",
                        output / "images",
                    )
                    for split in shards
                ],
                verify_image_hashes=True,
            )
            self.assertTrue(report["valid"], report["errors"])

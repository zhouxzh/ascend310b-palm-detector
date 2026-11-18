from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from palm_detector.dataset_contract import SplitSpec, connected_components, validate_splits
from palm_detector.labels import SINGLE_PALM_CATEGORY_IDS


def image(image_id: int, subject: str, session: str, file_name: str) -> dict:
    content = b"placeholder-image"
    return {"id": image_id, "file_name": file_name, "width": 10, "height": 10, "subject_id": subject, "session_id": session, "sha256": hashlib.sha256(content).hexdigest()}


def payload(images: list[dict]) -> dict:
    annotations = []
    annotation_id = 1
    for item in images:
        for category_id in (1, 2):
            annotations.append({"id": annotation_id, "image_id": item["id"], "category_id": category_id, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0})
            annotation_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "left_palm"}, {"id": 2, "name": "right_palm"}]}


class DatasetContractTest(unittest.TestCase):
    def test_nonfinite_bbox_and_area_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.bin").write_bytes(b"placeholder-image")
            annotations = root / "invalid.json"
            for value in (float("nan"), float("inf"), float("-inf")):
                for field in ("x", "y", "width", "height", "area"):
                    with self.subTest(value=value, field=field):
                        source = payload([image(1, "person-1", "session-a", "a.bin")])
                        annotation = source["annotations"][0]
                        if field == "area":
                            annotation["area"] = value
                        else:
                            annotation["bbox"][("x", "y", "width", "height").index(field)] = value
                        annotations.write_text(json.dumps(source))
                        report = validate_splits([SplitSpec("train", annotations, root)], decode_images=False)
                        self.assertFalse(report["valid"])
                        expected_error = "area must match" if field == "area" else "finite"
                        self.assertTrue(any(expected_error in error for error in report["errors"]))
                        json.dumps(report, allow_nan=False)

    def test_single_palm_contract_allows_unavailable_subject_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.bin").write_bytes(b"placeholder-image")
            item = image(1, "unused", "session-a", "a.bin")
            item.pop("subject_id")
            source = payload([item])
            source["categories"] = [{"id": 1, "name": "palm"}]
            source["annotations"] = [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}
            ]
            annotations = root / "single.json"
            annotations.write_text(json.dumps(source))
            report = validate_splits(
                [SplitSpec("train", annotations, root)],
                category_ids=SINGLE_PALM_CATEGORY_IDS,
                require_subject_id=False,
                decode_images=False,
            )
        self.assertTrue(report["valid"])
        self.assertIsNone(report["splits"][0]["subjects"])

    def test_subject_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.bin", "b.bin"):
                (root / name).write_bytes(b"placeholder-image")
            train = root / "train.json"
            val = root / "val.json"
            train.write_text(json.dumps(payload([image(1, "person-1", "session-a", "a.bin")])))
            val.write_text(json.dumps(payload([image(2, "person-1", "session-b", "b.bin")])))
            report = validate_splits([SplitSpec("train", train, root), SplitSpec("val", val, root)], decode_images=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any("subject IDs leakage" in error for error in report["errors"]))

    def test_low_count_gate_can_explicitly_allow_an_empty_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.bin").write_bytes(b"placeholder-image")
            source = payload([image(1, "person-1", "session-a", "a.bin")])
            source["categories"] = [
                {"id": 1, "name": "palm"},
                {"id": 2, "name": "back"},
                {"id": 3, "name": "side"},
            ]
            source["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}]
            annotations = root / "gate.json"
            annotations.write_text(json.dumps(source))
            report = validate_splits(
                [SplitSpec("gate", annotations, root)], decode_images=False,
                category_ids={1: "palm", 2: "back", 3: "side"}, require_all_classes=False,
            )
        self.assertTrue(report["valid"])

    def test_connected_components_join_subject_and_session(self) -> None:
        groups = connected_components([
            image(1, "person-1", "session-a", "a"),
            image(2, "person-1", "session-b", "b"),
            image(3, "person-2", "session-b", "c"),
            image(4, "person-3", "session-c", "d"),
        ])
        self.assertEqual(sorted(sorted(item["id"] for item in group) for group in groups), [[1, 2, 3], [4]])

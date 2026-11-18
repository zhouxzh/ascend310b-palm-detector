from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_onnx import _select_coco_images


class CompareOnnxSamplesTest(unittest.TestCase):
    def test_coco_selection_prioritizes_the_rare_class_and_records_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            payload = {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "palm"},
                    {"id": 2, "name": "back"},
                    {"id": 3, "name": "side"},
                ],
            }
            annotation_id = 1
            for image_id, category_id in enumerate((1, 1, 1, 2, 2, 2, 3), start=1):
                filename = f"image-{image_id}.jpg"
                content = f"sample-{image_id}".encode("ascii")
                (images / filename).write_bytes(content)
                payload["images"].append({"id": image_id, "file_name": filename})
                payload["annotations"].append({"id": annotation_id, "image_id": image_id, "category_id": category_id})
                annotation_id += 1
            annotations = root / "instances.json"
            annotations.write_text(json.dumps(payload), encoding="utf-8")

            paths, manifest = _select_coco_images(annotations, images, sample_count=4)

            self.assertEqual(len(paths), 4)
            self.assertEqual(manifest["selection"], "deterministic class-balanced, rare classes first")
            self.assertEqual(manifest["images"][0]["categories"], ["side"])
            self.assertEqual(
                manifest["images"][0]["sha256"],
                hashlib.sha256((images / "image-7.jpg").read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()

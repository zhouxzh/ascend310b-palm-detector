from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/surface_review_server.py"
    spec = importlib.util.spec_from_file_location("surface_review_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SurfaceReviewServerTest(unittest.TestCase):
    def test_deterministic_sampling_and_review_history(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            annotations = dataset / "annotations"
            images = dataset / "images"
            annotations.mkdir(parents=True)
            images.mkdir()
            manifest = {"splits": {}}
            identifier = 1
            for split in ("train", "val", "test"):
                split_images = []
                split_annotations = []
                for view in ("palm", "back", "side", "unknown"):
                    filename = f"{split}-{view}.jpg"
                    (images / filename).write_bytes(b"sample-image")
                    split_images.append({"id": identifier, "file_name": filename, "source_image": filename, "width": 20, "height": 20})
                    split_annotations.append({"id": identifier, "image_id": identifier, "category_id": 1, "bbox": [2, 2, 8, 8], "surface_view": "unknown" if view == "side" else view, "surface_candidate_view": view})
                    identifier += 1
                annotation_path = annotations / f"instances_{split}.json"
                annotation_path.write_text(json.dumps({"images": split_images, "annotations": split_annotations}))
                manifest["splits"][split] = {"output_annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest()}
            (dataset / "surface-manifest.json").write_text(json.dumps(manifest))
            reviews_path = root / "reviews.json"
            store = module.ReviewStore(dataset, reviews_path)
            first = store.sample(seed=42, size=8, split="all", view="all", mode="balanced")
            again = store.sample(seed=42, size=8, split="all", view="all", mode="balanced")
            self.assertEqual([item["id"] for item in first["items"]], [item["id"] for item in again["items"]])
            self.assertEqual({item["candidate_view"] for item in first["items"]}, {"palm", "back", "side", "unknown"})
            self.assertEqual(store.image_path(1).read_bytes(), b"sample-image")

            server = module.ReviewServer(("127.0.0.1", 0), store, module.OnnxReviewModels(root / "empty"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/sample?seed=42&size=8&split=all&view=all&mode=balanced") as response:
                    self.assertEqual(len(json.load(response)["items"]), 8)
                for view in ("palm", "back"):
                    note = "=2+2" if view == "back" else "checked"
                    body = json.dumps({"annotation_id": 1, "reviewer": "reviewer-1", "view": view, "note": note}).encode()
                    request = Request(base + "/api/review", data=body, headers={"Content-Type": "application/json", "Origin": base})
                    with urlopen(request) as response:
                        self.assertEqual(json.load(response)["view"], view)
                with urlopen(base + "/api/export.csv") as response:
                    self.assertIn("back,reviewer-1", response.read().decode("utf-8-sig"))
                self.assertIn("'=2+2", store.export_csv().decode("utf-8-sig"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            reopened = module.ReviewStore(dataset, reviews_path)
            self.assertEqual(len(reopened.data["reviews"]["1"]), 2)
            self.assertEqual(reopened.data["reviews"]["1"][-1]["view"], "back")
            (annotations / "instances_train.json").write_text('{"images":[],"annotations":[]}')
            with self.assertRaisesRegex(ValueError, "annotations do not match"):
                module.ReviewStore(dataset, reviews_path)

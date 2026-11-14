#!/usr/bin/env python3
"""Local review desk for provisional hand-surface COCO annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import os
import random
import sys
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from palm_detector.onnx_review import OnnxReviewModels

WEB_ROOT = ROOT / "web" / "surface_review"
DEFAULT_DATASET = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-surface-v001"
DEFAULT_REVIEWS = ROOT / "artifacts/reviews/hand-surface-v001.json"
VIEWS = ("palm", "back", "side", "unknown", "invalid")
SPLITS = ("train", "val", "test")


class ReviewStore:
    def __init__(self, dataset_dir: Path, reviews_path: Path):
        self.dataset_dir = dataset_dir.resolve()
        self.images_root = self.dataset_dir / "images"
        self.reviews_path = reviews_path
        self.lock = threading.Lock()
        manifest = self.dataset_dir / "surface-manifest.json"
        manifest_bytes = manifest.read_bytes()
        manifest_data = json.loads(manifest_bytes)
        self.dataset_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.images: dict[int, dict] = {}
        self.items: dict[int, dict] = {}
        for split in SPLITS:
            annotation_path = self.dataset_dir / "annotations" / f"instances_{split}.json"
            annotation_bytes = annotation_path.read_bytes()
            expected_hash = manifest_data["splits"][split]["output_annotation_sha256"]
            if hashlib.sha256(annotation_bytes).hexdigest() != expected_hash:
                raise ValueError(f"{split} annotations do not match the surface manifest")
            coco = json.loads(annotation_bytes)
            for image in coco["images"]:
                if image["id"] in self.images:
                    raise ValueError(f"duplicate image ID {image['id']}")
                self.images[image["id"]] = image
            for annotation in coco["annotations"]:
                identifier = annotation["id"]
                if identifier in self.items:
                    raise ValueError(f"duplicate annotation ID {identifier}")
                image = self.images[annotation["image_id"]]
                self.items[identifier] = {
                    "id": identifier,
                    "image_id": image["id"],
                    "split": split,
                    "source_image": image.get("source_image", image["file_name"]),
                    "width": image["width"],
                    "height": image["height"],
                    "bbox": annotation["bbox"],
                    "handedness": "left" if annotation["category_id"] == 1 else "right",
                    "view": annotation["surface_view"],
                    "candidate_view": annotation["surface_candidate_view"],
                    "facing": annotation.get("surface_facing"),
                }
        if reviews_path.is_file():
            self.data = json.loads(reviews_path.read_text(encoding="utf-8"))
            if self.data.get("dataset_sha256") != self.dataset_sha256:
                raise ValueError("review file belongs to a different dataset manifest")
        else:
            self.data = {"schema_version": 1, "dataset_sha256": self.dataset_sha256, "reviews": {}}

    def metadata(self) -> dict:
        with self.lock:
            reviewed = len(self.data["reviews"])
        return {
            "dataset": self.dataset_dir.name,
            "total": len(self.items),
            "reviewed": reviewed,
            "candidates": dict(Counter(item["candidate_view"] for item in self.items.values())),
            "splits": dict(Counter(item["split"] for item in self.items.values())),
        }

    def sample(self, *, seed: int, size: int, split: str, view: str, mode: str) -> dict:
        if not 1 <= size <= 500 or split not in ("all", *SPLITS) or view not in ("all", *VIEWS[:4]) or mode not in ("balanced", "random"):
            raise ValueError("invalid sample settings")
        pool = [item for item in self.items.values() if (split == "all" or item["split"] == split) and (view == "all" or item["candidate_view"] == view)]
        pool.sort(key=lambda item: item["id"])
        rng = random.Random(seed)
        if mode == "balanced" and view == "all":
            groups = {name: [item for item in pool if item["candidate_view"] == name] for name in VIEWS[:4]}
            for group in groups.values():
                rng.shuffle(group)
            selected = []
            while len(selected) < size and any(groups.values()):
                for name in VIEWS[:4]:
                    if groups[name] and len(selected) < size:
                        selected.append(groups[name].pop())
        else:
            selected = rng.sample(pool, min(size, len(pool)))
        with self.lock:
            history = self.data["reviews"]
            items = [{**item, "review": history.get(str(item["id"]), [None])[-1]} for item in selected]
        return {"items": items, "pool_size": len(pool), "seed": seed, "mode": mode}

    def save_review(self, payload: dict) -> dict:
        try:
            identifier = int(payload["annotation_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("annotation_id must be an integer") from exc
        if identifier not in self.items:
            raise ValueError("annotation not found")
        reviewer = str(payload.get("reviewer", "")).strip()
        view = payload.get("view")
        note = str(payload.get("note", "")).strip()
        if not reviewer or len(reviewer) > 100 or view not in VIEWS or len(note) > 2000:
            raise ValueError("reviewer, view, or note is invalid")
        record = {
            "annotation_id": identifier,
            "reviewer": reviewer,
            "view": view,
            "note": note,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            key = str(identifier)
            history = self.data["reviews"].setdefault(key, [])
            history.append(record)
            try:
                self._write()
            except Exception:
                history.pop()
                if not history:
                    del self.data["reviews"][key]
                raise
        return record

    def _write(self) -> None:
        self.reviews_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.reviews_path.parent, prefix=".surface-review-", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(self.data, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.reviews_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def export_csv(self) -> bytes:
        def cell(value: object) -> object:
            text = str(value)
            return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else value

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(("split", "annotation_id", "source_image", "handedness", "auto_view", "candidate_view", "reviewed_view", "reviewer", "reviewed_at_utc", "note"))
        with self.lock:
            for key, history in sorted(self.data["reviews"].items(), key=lambda entry: int(entry[0])):
                item = self.items[int(key)]
                review = history[-1]
                writer.writerow(tuple(cell(value) for value in (
                    item["split"], key, item["source_image"], item["handedness"],
                    item["view"], item["candidate_view"], review["view"],
                    review["reviewer"], review["reviewed_at_utc"], review["note"],
                )))
        return output.getvalue().encode("utf-8-sig")

    def image_path(self, image_id: int) -> Path:
        image = self.images.get(image_id)
        if image is None:
            raise ValueError("image not found")
        relative = Path(image["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid image path")
        path = self.images_root / relative
        if not path.is_file():
            raise ValueError("image file is missing")
        return path


class Handler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def _send(self, body: bytes, content_type: str, status: int = 200, *, download: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: dict, status: int = 200) -> None:
        self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path == "/api/metadata":
                return self._json(self.server.store.metadata())
            if path == "/api/models":
                return self._json({"models": self.server.models.catalog()})
            if path == "/api/sample":
                query = parse_qs(parsed.query)
                return self._json(self.server.store.sample(
                    seed=int(query.get("seed", ["42"])[0]),
                    size=int(query.get("size", ["100"])[0]),
                    split=query.get("split", ["all"])[0],
                    view=query.get("view", ["all"])[0],
                    mode=query.get("mode", ["balanced"])[0],
                ))
            if path == "/api/export.csv":
                return self._send(self.server.store.export_csv(), "text/csv; charset=utf-8", download="hand-surface-reviews.csv")
            if path.startswith("/api/image/"):
                image_path = self.server.store.image_path(int(path.rsplit("/", 1)[1]))
                return self._send(image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream")
            if path in ("/", "/assets/app.css", "/assets/app.js"):
                filename = {"/": "index.html", "/assets/app.css": "app.css", "/assets/app.js": "app.js"}[path]
                mime = {"index.html": "text/html", "app.css": "text/css", "app.js": "text/javascript"}[filename]
                return self._send((WEB_ROOT / filename).read_bytes(), f"{mime}; charset=utf-8")
            self._json({"error": "not found"}, 404)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/api/review", "/api/infer"):
            return self._json({"error": "not found"}, 404)
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if origin and origin != expected:
            return self._json({"error": "origin is not allowed"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json" or not 0 < length <= 8192:
                raise ValueError("expected a JSON review")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            if path == "/api/review":
                return self._json(self.server.store.save_review(payload))
            image_id = payload.get("image_id")
            if not isinstance(image_id, int) or isinstance(image_id, bool):
                raise ValueError("image_id must be an integer")
            model_ids = payload.get("model_ids")
            if not isinstance(model_ids, list) or any(not isinstance(value, str) for value in model_ids):
                raise ValueError("model_ids must be a list of model IDs")
            threshold = float(payload.get("score_threshold", 0.35))
            return self._json(self.server.models.infer(self.server.store.image_path(image_id), model_ids, threshold))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self.log_error("inference failed: %s", exc)
            self._json({"error": "model inference failed"}, 500)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: ReviewStore, models: OnnxReviewModels | None = None):
        super().__init__(address, Handler)
        self.store = store
        self.models = models or OnnxReviewModels(ROOT / "artifacts/runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("review server must bind to 127.0.0.1")
    store = ReviewStore(args.dataset_dir, args.reviews)
    with ReviewServer((args.host, args.port), store) as server:
        print(f"Surface review: http://{args.host}:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

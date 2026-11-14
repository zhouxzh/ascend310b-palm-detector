#!/usr/bin/env python3
"""Publish COCO labels that reference the existing portable HaGRIDv2 images."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from collections import Counter
from pathlib import Path

from palm_detector.dataset_contract import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
DEFAULT_SOURCE = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-v001"
DEFAULT_SURFACE = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-surface-v001"
DEFAULT_OUTPUT = ROOT / "artifacts/hf-release/portable-handed-v001-annotations"
DEFAULT_KEY = ROOT / "artifacts/hf-release/subject-key.bin"
SOURCE_REPO = "zhouxzh/portable-hagridv2-mediapipe-hand"
SOURCE_REVISION = "e343025bd4fd37719dbc269201d6b24124f0c432"


def _anonymous_id(source_id: str, key: bytes, kind: str) -> str:
    digest = hmac.new(key, f"{kind}:{source_id}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{kind}-{digest[:20]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subject-key", type=Path, default=DEFAULT_KEY)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")
    args.subject_key.parent.mkdir(parents=True, exist_ok=True)
    if args.subject_key.exists():
        key = args.subject_key.read_bytes()
    else:
        key = secrets.token_bytes(32)
        args.subject_key.write_bytes(key)
        args.subject_key.chmod(0o600)
    if len(key) != 32:
        raise ValueError("subject key must be 32 bytes")

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "annotations").mkdir()
    (args.output_dir / "surface_annotations").mkdir()
    manifest: dict[str, object] = {
        "source_dataset": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "image_policy": "No image bytes are republished; source_image maps each COCO image to the pinned source Parquet row",
        "subject_id_policy": "HMAC-SHA256 pseudonyms; release key stays local",
        "surface_label_status": "automatic candidates, not human-reviewed ground truth",
        "splits": {},
    }
    for split in SPLITS:
        source_path = args.source_dir / "annotations" / f"instances_{split}.json"
        surface_path = args.surface_dir / "annotations" / f"instances_{split}.json"
        coco = json.loads(source_path.read_text(encoding="utf-8"))
        surface = json.loads(surface_path.read_text(encoding="utf-8"))
        if coco["images"] != surface["images"] or [item["id"] for item in coco["annotations"]] != [item["id"] for item in surface["annotations"]]:
            raise ValueError(f"{split}: surface annotations do not match detector annotations")
        for image, surface_image in zip(coco["images"], surface["images"]):
            source_image = image["source_image"]
            if not source_image or source_image.startswith("/") or ".." in Path(source_image).parts:
                raise ValueError(f"invalid source image path: {source_image}")
            source_file = args.source_dir / "images" / image["file_name"]
            if sha256_file(source_file) != image["sha256"]:
                raise ValueError(f"source image hash mismatch: {source_file}")
            subject = _anonymous_id(image["subject_id"], key, "subject")
            session = _anonymous_id(image["session_id"], key, "session")
            for output_image in (image, surface_image):
                output_image["subject_id"] = subject
                output_image["session_id"] = session
                output_image["file_name"] = source_image
                output_image["source_image"] = source_image
                output_image["source_dataset"] = SOURCE_REPO
        outputs = (
            (coco, args.output_dir / "annotations" / f"instances_{split}.json"),
            (surface, args.output_dir / "surface_annotations" / f"instances_{split}.json"),
        )
        for payload, output_path in outputs:
            output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        counts = Counter(ann["category_id"] for ann in coco["annotations"])
        manifest["splits"][split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": {item["name"]: counts[item["id"]] for item in coco["categories"]},
            "source_annotation_sha256": sha256_file(source_path),
            "surface_annotation_sha256": sha256_file(surface_path),
            "release_annotation_sha256": sha256_file(outputs[0][1]),
            "release_surface_sha256": sha256_file(outputs[1][1]),
        }
        print(f"{split}: {len(coco['images'])} image references, {len(coco['annotations'])} palms", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "---\nlicense: other\nlicense_name: hagridv2-public-license-with-attribution-and-conditions-reserved\n"
        "license_link: https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf\n"
        "task_categories:\n- object-detection\n---\n\n"
        "# Portable HaGRIDv2 palm annotations\n\n"
        f"This repository contains annotations only. Image bytes remain in "
        f"[`{SOURCE_REPO}`](https://huggingface.co/datasets/{SOURCE_REPO}) "
        f"at revision `{SOURCE_REVISION}`. Each COCO `images[]` entry has "
        "`source_image`, `file_name`, and `sha256` to identify and verify the "
        "corresponding source image. The source dataset embeds JPEG bytes in "
        "Parquet; these files are not duplicated here.\n\n"
        "`annotations/` contains COCO palm-region boxes with category 1 = "
        "anatomical left palm and category 2 = anatomical right palm. Subject "
        "and session IDs are release-specific pseudonyms.\n\n"
        "`surface_annotations/` adds provisional palm/back/unknown views. "
        "Those view labels were generated automatically and have not been "
        "human reviewed. They are not validated palmprint admission labels.\n\n"
        "See `manifest.json` for counts, provenance, and annotation SHA-256.\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

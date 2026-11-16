#!/usr/bin/env python3
"""Split the existing validation data by subject for search and merge checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from palm_detector.dataset_contract import SplitSpec, sha256_file, validate_splits


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/portable-hagridv2-mediapipe-hand/coco-handed-v001"
OUTPUT = ROOT / "artifacts/sweeps/palm-v001/validation"


def split_validation(source: dict, *, gate_fraction: float = 0.27, seed: int = 20261114) -> tuple[dict, dict]:
    if not 0 < gate_fraction < 1:
        raise ValueError("gate_fraction must be in (0, 1)")
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for image in source["images"]:
        by_subject[str(image["subject_id"])].append(image)
    subjects = sorted(
        by_subject,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    selected: set[str] = set()
    target = round(len(source["images"]) * gate_fraction)
    count = 0
    for subject in subjects:
        if count >= target and selected:
            break
        selected.add(subject)
        count += len(by_subject[subject])
    outputs = []
    for gate in (False, True):
        images = [image for image in source["images"] if (str(image["subject_id"]) in selected) == gate]
        image_ids = {image["id"] for image in images}
        payload = {
            **{key: source[key] for key in ("info", "licenses") if key in source},
            "images": images,
            "annotations": [item for item in source["annotations"] if item["image_id"] in image_ids],
            "categories": source["categories"],
        }
        outputs.append(payload)
    return outputs[0], outputs[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE / "annotations/instances_val.json")
    parser.add_argument("--images-root", type=Path, default=SOURCE / "images")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--gate-fraction", type=float, default=0.27)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory exists: {args.output_dir}")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    dev, gate = split_validation(source, gate_fraction=args.gate_fraction, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    specs = []
    for name, payload in (("dev", dev), ("gate", gate)):
        path = args.output_dir / f"instances_{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        specs.append(SplitSpec(name, path, args.images_root))
    audit = validate_splits(specs, verify_image_hashes=True)
    if not audit["valid"]:
        raise ValueError(f"validation partition failed audit: {audit['errors'][:5]}")
    report = {
        "source": str(args.source), "source_sha256": sha256_file(args.source),
        "seed": args.seed, "gate_fraction": args.gate_fraction,
        "splits": {
            spec.name: {
                "images": len(payload["images"]),
                "annotations": len(payload["annotations"]),
                "subjects": len({image["subject_id"] for image in payload["images"]}),
                "classes": dict(Counter(item["category_id"] for item in payload["annotations"])),
                "sha256": sha256_file(spec.annotation_path),
            }
            for spec, payload in zip(specs, (dev, gate))
        },
    }
    (args.output_dir / "split-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_root = args.source.parent.parent
    full_specs = [
        SplitSpec("train", source_root / "annotations/instances_train.json", args.images_root),
        *specs,
        SplitSpec("test", source_root / "annotations/instances_test.json", args.images_root),
    ]
    full_audit = validate_splits(full_specs, verify_image_hashes=True)
    if not full_audit["valid"]:
        raise ValueError(f"full sweep split audit failed: {full_audit['errors'][:5]}")
    (args.output_dir.parent / "sweep-data-audit.json").write_text(json.dumps(full_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_dir / "split-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

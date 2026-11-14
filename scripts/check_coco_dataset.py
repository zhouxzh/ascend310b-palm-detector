#!/usr/bin/env python3
"""Fail closed on invalid or leaking COCO data under an explicit category contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palm_detector.dataset_contract import SplitSpec, validate_splits
from palm_detector.labels import COCO_CATEGORY_IDS, SINGLE_PALM_CATEGORY_IDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs=3, action="append", metavar=("NAME", "ANNOTATIONS", "IMAGES_ROOT"), required=True)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--skip-image-decode", action="store_true")
    parser.add_argument("--category-contract", choices=("left-right", "single-palm"), default="left-right")
    parser.add_argument("--allow-missing-subject-id", action="store_true")
    parser.add_argument("--allow-missing-session-id", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [item[0] for item in args.split]
    if len(names) != len(set(names)):
        parser.error("each --split NAME must be unique")
    category_ids = COCO_CATEGORY_IDS if args.category_contract == "left-right" else SINGLE_PALM_CATEGORY_IDS
    report = validate_splits(
        [SplitSpec(name, Path(annotation), Path(images_root)) for name, annotation, images_root in args.split],
        verify_image_hashes=args.verify_image_hashes,
        decode_images=not args.skip_image_decode,
        category_ids=category_ids,
        require_subject_id=not args.allow_missing_subject_id,
        require_session_id=not args.allow_missing_session_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if report['valid'] else 'FAIL'}")
    for error in report["errors"][:20]:
        print(f"ERROR: {error}")
    if len(report["errors"]) > 20:
        print(f"ERROR: {len(report['errors']) - 20} additional violations are recorded in {args.output}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

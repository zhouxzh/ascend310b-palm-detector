#!/usr/bin/env python3
"""Compare static ONNX and PyTorch raw outputs plus decoded detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from palm_detector.postprocess import decode_nanodet_plus, detections_to_json, preprocess_bgr, preprocess_bgr_letterbox


def _export_equivalent_output(pytorch_output: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Mirror NanoDetHead and NanoDetPlusHead ONNX forward class sigmoid.

    Normal PyTorch forward returns class logits; the static ONNX branch applies
    sigmoid to the class channels before concatenating unmodified DFL bins.
    """
    converted = np.array(pytorch_output, copy=True)
    logits = converted[:, :, :num_classes]
    converted[:, :, :num_classes] = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    return converted


def _iou(left: dict[str, object], right: dict[str, object]) -> float:
    left_box = np.asarray(left["bbox_xyxy"], dtype=np.float32)
    right_box = np.asarray(right["bbox_xyxy"], dtype=np.float32)
    top_left = np.maximum(left_box[:2], right_box[:2])
    bottom_right = np.minimum(left_box[2:], right_box[2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod()
    left_area = np.maximum(left_box[2] - left_box[0], 0) * np.maximum(left_box[3] - left_box[1], 0)
    right_area = np.maximum(right_box[2] - right_box[0], 0) * np.maximum(right_box[3] - right_box[1], 0)
    return float(intersection / max(left_area + right_area - intersection, 1e-8))


def _compare_detections(reference: list[dict[str, object]], candidate: list[dict[str, object]], minimum_iou: float) -> tuple[bool, float]:
    if len(reference) != len(candidate):
        return False, 0.0
    remaining = set(range(len(candidate)))
    ious: list[float] = []
    for item in reference:
        matches = [(index, _iou(item, candidate[index])) for index in remaining if candidate[index]["category_id"] == item["category_id"]]
        if not matches:
            return False, 0.0
        index, iou = max(matches, key=lambda pair: pair[1])
        if iou < minimum_iou:
            return False, iou
        remaining.remove(index)
        ious.append(iou)
    return True, min(ious, default=1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_coco_images(annotations: Path, images_root: Path, sample_count: int) -> tuple[list[Path], dict[str, object]]:
    """Select a deterministic, class-balanced set of readable COCO images."""
    if sample_count < 1:
        raise ValueError("sample count must be positive")
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in payload["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    by_category = {category_id: set() for category_id in categories}
    for annotation in payload["annotations"]:
        category_id = int(annotation["category_id"])
        image_id = int(annotation["image_id"])
        if category_id in by_category and image_id in image_by_id:
            by_category[category_id].add(image_id)

    selected_ids: list[int] = []
    # Rare categories are considered first, so the runtime check exercises them
    # even when the test split is highly imbalanced.
    quota = max(1, (sample_count + len(categories) - 1) // len(categories))
    for category_id in sorted(categories, key=lambda item: (len(by_category[item]), categories[item])):
        for image_id in sorted(by_category[category_id]):
            if image_id not in selected_ids:
                selected_ids.append(image_id)
            if len(selected_ids) >= sample_count or sum(
                item in by_category[category_id] for item in selected_ids
            ) >= quota:
                break
        if len(selected_ids) >= sample_count:
            break
    if len(selected_ids) < sample_count:
        for image_id in sorted(image_by_id):
            if image_id not in selected_ids:
                selected_ids.append(image_id)
            if len(selected_ids) >= sample_count:
                break

    paths: list[Path] = []
    records: list[dict[str, object]] = []
    for image_id in selected_ids:
        filename = str(image_by_id[image_id]["file_name"])
        path = images_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        records.append({
            "image_id": image_id,
            "file_name": filename,
            "path": str(path),
            "sha256": _sha256(path),
            "categories": [categories[item] for item in sorted(categories) if image_id in by_category[item]],
        })
    manifest = {
        "schema_version": 1,
        "annotations": str(annotations),
        "annotations_sha256": _sha256(annotations),
        "selection": "deterministic class-balanced, rare classes first",
        "requested_count": sample_count,
        "images": records,
    }
    return paths, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--coco-annotations", type=Path, help="COCO annotations used for deterministic representative-image selection")
    parser.add_argument("--images-root", type=Path, help="Image directory paired with --coco-annotations")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--sample-manifest", type=Path, help="Optional persisted manifest for COCO-selected images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    parser.add_argument("--minimum-decoded-iou", type=float, default=0.999)
    args = parser.parse_args()
    if args.images:
        if args.coco_annotations or args.images_root or args.sample_manifest:
            parser.error("--images cannot be combined with COCO sampling options")
        images = args.images
        sample_manifest = None
    else:
        if not args.coco_annotations or not args.images_root:
            parser.error("provide --images or both --coco-annotations and --images-root")
        if not args.coco_annotations.is_file() or not args.images_root.is_dir():
            parser.error("COCO annotations or image root is missing")
        images, sample_manifest = _select_coco_images(args.coco_annotations, args.images_root, args.sample_count)
        if args.sample_manifest:
            args.sample_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.sample_manifest.write_text(json.dumps(sample_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "third_party/nanodet"))
    from nanodet.model.arch import build_model
    from nanodet.util import Logger, cfg, load_config, load_model_weight
    import onnxruntime as ort

    load_config(cfg, str(args.config))
    letterbox_mode = bool(cfg.data.val.pipeline.get("palm_letterbox", False))
    if cfg.data.val.keep_ratio and not letterbox_mode:
        raise ValueError("compare_onnx requires direct resize or the fixed Palm LetterBox mode")
    input_width, input_height = (int(value) for value in cfg.data.val.input_size)
    mean, std = (tuple(values) for values in cfg.data.val.pipeline.normalize)
    head = cfg.model.arch.head
    decode_options = {
        "class_names": tuple(cfg.class_names),
        "input_size": (input_height, input_width),
        "strides": tuple(int(value) for value in head.strides),
        "reg_max": int(head.reg_max),
        "center_offset": 0.0 if cfg.model.arch.name == "NanoDetPlus" else 0.5,
    }
    model = build_model(cfg.model).eval()
    load_model_weight(model, torch.load(args.checkpoint, map_location="cpu"), Logger(-1, str(args.output.parent), False))
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    report_images = []
    all_ok = True
    with torch.no_grad():
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot read image {image_path}")
            if letterbox_mode:
                tensor, letterbox = preprocess_bgr_letterbox(image, input_size=(input_height, input_width), mean=mean, std=std)
            else:
                tensor, _ = preprocess_bgr(image, input_size=(input_height, input_width), mean=mean, std=std)
                letterbox = None
            pytorch_logits = model(torch.from_numpy(tensor)).numpy()
            pytorch_output = _export_equivalent_output(pytorch_logits, len(cfg.class_names))
            onnx_output = session.run(["output"], {"data": tensor})[0]
            difference = np.abs(pytorch_output - onnx_output)
            max_error = float(difference.max())
            decoded_torch = detections_to_json(decode_nanodet_plus(pytorch_logits, scores_are_logits=True, original_size=None if letterbox else image.shape[:2], letterbox=letterbox, **decode_options)[0])
            decoded_onnx = detections_to_json(decode_nanodet_plus(onnx_output, original_size=None if letterbox else image.shape[:2], letterbox=letterbox, **decode_options)[0])
            decoded_match, minimum_iou = _compare_detections(decoded_torch, decoded_onnx, args.minimum_decoded_iou)
            passed = max_error <= args.max_abs_error and decoded_match
            all_ok = all_ok and passed
            report_images.append({"image": str(image_path), "raw_max_abs_error": max_error, "raw_mean_abs_error": float(difference.mean()), "decoded_class_and_iou_match": decoded_match, "decoded_minimum_iou": minimum_iou, "pytorch_detections": decoded_torch, "onnx_detections": decoded_onnx, "passed": passed})
    report = {"valid": all_ok, "raw_tensor_contract": "PyTorch class logits sigmoid-transformed to match NanoDet ONNX export; DFL bins unchanged", "preprocess": "mediapipe_letterbox" if letterbox_mode else "direct_resize", "input_size": [input_width, input_height], "strides": list(decode_options["strides"]), "reg_max": decode_options["reg_max"], "center_offset": decode_options["center_offset"], "max_abs_error_limit": args.max_abs_error, "decoded_iou_limit": args.minimum_decoded_iou, "sample_manifest": str(args.sample_manifest) if sample_manifest is not None and args.sample_manifest else None, "images": report_images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Deployment-side preprocessing and DFL/NMS decoding for static NanoDet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .labels import CLASS_NAMES

INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BGR_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
BGR_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
STRIDES = (8, 16, 32, 64)
REG_MAX = 7


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    label: int
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def class_name(self) -> str:
        return self.class_names[self.label]


def preprocess_bgr(
    image: np.ndarray,
    *,
    input_size: tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH),
    mean: tuple[float, float, float] | np.ndarray = BGR_MEAN,
    std: tuple[float, float, float] | np.ndarray = BGR_STD,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return NCHW input and x/y scale for an ``(height, width)`` model size."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HWC BGR image with three channels")
    input_height, input_width = input_size
    if input_height <= 0 or input_width <= 0:
        raise ValueError("model input dimensions must be positive")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("mean and std must be finite BGR triplets with positive std")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for deployment preprocessing") from exc
    height, width = image.shape[:2]
    resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - mean) / std
    return normalized.transpose(2, 0, 1)[None, ...], (input_width / width, input_height / height)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def center_priors(
    input_height: int = INPUT_HEIGHT,
    input_width: int = INPUT_WIDTH,
    *,
    strides: tuple[int, ...] = STRIDES,
    center_offset: float = 0.0,
) -> np.ndarray:
    if input_height <= 0 or input_width <= 0 or not strides or any(stride <= 0 for stride in strides):
        raise ValueError("input dimensions and strides must be positive")
    if center_offset not in (0.0, 0.5):
        raise ValueError("center_offset must be 0.0 (Plus) or 0.5 (legacy)")
    priors: list[np.ndarray] = []
    for stride in strides:
        feature_height = int(np.ceil(input_height / stride))
        feature_width = int(np.ceil(input_width / stride))
        yy, xx = np.meshgrid(np.arange(feature_height), np.arange(feature_width), indexing="ij")
        priors.append(
            np.stack(
                ((xx.reshape(-1) + center_offset) * stride, (yy.reshape(-1) + center_offset) * stride, np.full(xx.size, stride)),
                axis=1,
            )
        )
    return np.concatenate(priors, axis=0).astype(np.float32)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=1)
    box_area = np.maximum(box[2] - box[0], 0) * np.maximum(box[3] - box[1], 0)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return intersection / np.maximum(box_area + areas - intersection, 1e-8)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    keep: list[int] = []
    order = scores.argsort()[::-1]
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        order = order[1:][_iou_one_to_many(boxes[index], boxes[order[1:]]) <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_nanodet_plus(
    output: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
    input_size: tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH),
    strides: tuple[int, ...] = STRIDES,
    reg_max: int = REG_MAX,
    center_offset: float = 0.0,
    score_threshold: float = 0.35,
    nms_iou_threshold: float = 0.6,
    scores_are_logits: bool = False,
    original_size: tuple[int, int] | None = None,
) -> list[list[Detection]]:
    """Decode Plus or legacy raw output without ONNX NMS ops."""
    output = np.asarray(output, dtype=np.float32)
    if not class_names:
        raise ValueError("class_names must not be empty")
    if reg_max < 1:
        raise ValueError("reg_max must be positive")
    input_height, input_width = input_size
    num_classes = len(class_names)
    expected_channels = num_classes + 4 * (reg_max + 1)
    priors = center_priors(input_height, input_width, strides=strides, center_offset=center_offset)
    if output.ndim != 3 or output.shape[1:] != (len(priors), expected_channels):
        raise ValueError(f"expected [N, {len(priors)}, {expected_channels}] NanoDet output, got {output.shape}")
    class_scores = output[:, :, :num_classes]
    if scores_are_logits:
        class_scores = _sigmoid(class_scores)
    distributions = output[:, :, num_classes:].reshape(output.shape[0], len(priors), 4, reg_max + 1)
    distances = (_softmax(distributions) * np.arange(reg_max + 1, dtype=np.float32)).sum(axis=-1) * priors[None, :, 2:3]
    boxes = np.concatenate((priors[None, :, :2] - distances[:, :, :2], priors[None, :, :2] + distances[:, :, 2:]), axis=-1)
    boxes[:, :, [0, 2]] = np.clip(boxes[:, :, [0, 2]], 0, input_width)
    boxes[:, :, [1, 3]] = np.clip(boxes[:, :, [1, 3]], 0, input_height)
    if original_size is not None:
        original_height, original_width = original_size
        boxes[:, :, [0, 2]] *= original_width / input_width
        boxes[:, :, [1, 3]] *= original_height / input_height
    decoded: list[list[Detection]] = []
    for batch_boxes, batch_scores in zip(boxes, class_scores):
        detections: list[Detection] = []
        for label in range(num_classes):
            candidates = np.flatnonzero(batch_scores[:, label] >= score_threshold)
            if not candidates.size:
                continue
            kept = candidates[nms(batch_boxes[candidates], batch_scores[candidates, label], nms_iou_threshold)]
            detections.extend(Detection(tuple(float(value) for value in batch_boxes[index]), float(batch_scores[index, label]), label, class_names) for index in kept)
        decoded.append(sorted(detections, key=lambda item: item.score, reverse=True))
    return decoded


def detections_to_json(detections: Iterable[Detection]) -> list[dict[str, object]]:
    return [{"bbox_xyxy": list(item.xyxy), "score": item.score, "category_id": item.label + 1, "class_name": item.class_name} for item in detections]

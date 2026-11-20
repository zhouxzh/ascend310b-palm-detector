"""Deployment preprocessing and raw-head decoding for exported PicoDet ONNX."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .letterbox import LetterboxTransform, letterbox_bgr
from .postprocess import Detection, _softmax, nms

PICODET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PICODET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PICODET_STRIDES = (8, 16, 32, 64)
PICODET_REG_MAX = 7
PICODET_CELL_OFFSET = 0.5


def preprocess_picodet_bgr(
    image: np.ndarray,
    *,
    input_size: tuple[int, int],
    mean_rgb: np.ndarray | Sequence[float] = PICODET_MEAN_RGB,
    std_rgb: np.ndarray | Sequence[float] = PICODET_STD_RGB,
    interpolation: int = 2,
) -> np.ndarray:
    """Create the PicoDet direct-stretch RGB / 255 / ImageNet-normalized input."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HWC BGR image with three channels")
    mean = np.asarray(mean_rgb, dtype=np.float32)
    std = np.asarray(std_rgb, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("mean and std must be finite RGB triplets with positive std")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for PicoDet preprocessing") from exc
    height, width = input_size
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    normalized = (rgb - mean) / std
    return normalized.transpose(2, 0, 1)[None, ...]


def preprocess_picodet_bgr_letterbox(
    image: np.ndarray,
    *,
    input_size: tuple[int, int],
    mean_rgb: np.ndarray | Sequence[float] = PICODET_MEAN_RGB,
    std_rgb: np.ndarray | Sequence[float] = PICODET_STD_RGB,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Create the exact RGB / 255 / ImageNet-normalized PicoDet ONNX input."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HWC BGR image with three channels")
    mean = np.asarray(mean_rgb, dtype=np.float32)
    std = np.asarray(std_rgb, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("mean and std must be finite RGB triplets with positive std")
    canvas, letterbox = letterbox_bgr(image, target_size=input_size, pad_value=0)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    normalized = (rgb - mean) / std
    return normalized.transpose(2, 0, 1)[None, ...], letterbox


def _picodet_priors(
    input_size: tuple[int, int], strides: tuple[int, ...], expected_counts: Sequence[int]
) -> np.ndarray:
    input_height, input_width = input_size
    if input_height <= 0 or input_width <= 0 or len(strides) != len(expected_counts):
        raise ValueError("input size, strides, and output levels must agree")
    priors: list[np.ndarray] = []
    for stride, expected_count in zip(strides, expected_counts):
        if stride <= 0:
            raise ValueError("PicoDet strides must be positive")
        feature_height = int(np.ceil(input_height / stride))
        feature_width = int(np.ceil(input_width / stride))
        if feature_height * feature_width != expected_count:
            raise ValueError(
                f"PicoDet level stride {stride} expects {feature_height * feature_width} locations, got {expected_count}"
            )
        yy, xx = np.meshgrid(np.arange(feature_height), np.arange(feature_width), indexing="ij")
        priors.append(
            np.stack(
                ((xx.reshape(-1) + PICODET_CELL_OFFSET) * stride, (yy.reshape(-1) + PICODET_CELL_OFFSET) * stride),
                axis=1,
            )
        )
    return np.concatenate(priors, axis=0).astype(np.float32)


def decode_picodet(
    outputs: Sequence[np.ndarray],
    *,
    class_names: tuple[str, ...],
    input_size: tuple[int, int],
    strides: tuple[int, ...] = PICODET_STRIDES,
    reg_max: int = PICODET_REG_MAX,
    score_threshold: float = 0.35,
    nms_iou_threshold: float = 0.6,
    keep_top_k: int | None = 100,
    letterbox: LetterboxTransform | None = None,
) -> list[list[Detection]]:
    """Decode raw PicoDet outputs: class probabilities then DFL values per level.

    ``export.post_process=False`` produces one class tensor for each feature
    level followed by its matching DFL tensor. The head's class tensors are
    already sigmoid probabilities, so applying sigmoid again is incorrect.
    """
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("score and NMS thresholds must be in [0, 1]")
    if keep_top_k is not None and keep_top_k <= 0:
        raise ValueError("keep_top_k must be positive or None")
    if reg_max < 1:
        raise ValueError("reg_max must be positive")
    if len(outputs) == 0 or len(outputs) % 2:
        raise ValueError("PicoDet output must contain class and DFL tensors for each level")
    levels = len(outputs) // 2
    if len(strides) != levels:
        raise ValueError(f"expected {len(strides)} PicoDet levels, got {levels}")
    class_outputs = [np.asarray(output, dtype=np.float32) for output in outputs[:levels]]
    dfl_outputs = [np.asarray(output, dtype=np.float32) for output in outputs[levels:]]
    expected_dfl_channels = 4 * (reg_max + 1)
    batch_size: int | None = None
    counts: list[int] = []
    for index, (classes, dfl) in enumerate(zip(class_outputs, dfl_outputs)):
        if classes.ndim != 3 or classes.shape[2] != len(class_names):
            raise ValueError(f"PicoDet class output {index} must have shape [N, locations, {len(class_names)}], got {classes.shape}")
        if dfl.ndim != 3 or dfl.shape[2] != expected_dfl_channels:
            raise ValueError(f"PicoDet DFL output {index} must have shape [N, locations, {expected_dfl_channels}], got {dfl.shape}")
        if classes.shape[:2] != dfl.shape[:2]:
            raise ValueError(f"PicoDet class and DFL output {index} have incompatible shapes")
        if batch_size is None:
            batch_size = classes.shape[0]
        if classes.shape[0] != batch_size:
            raise ValueError("all PicoDet outputs must have the same batch size")
        counts.append(classes.shape[1])
    assert batch_size is not None
    priors = _picodet_priors(input_size, strides, counts)
    scores = np.concatenate(class_outputs, axis=1)
    raw_dfl = np.concatenate(dfl_outputs, axis=1).reshape(batch_size, len(priors), 4, reg_max + 1)
    stride_per_prior = np.concatenate([np.full(count, stride, dtype=np.float32) for count, stride in zip(counts, strides)])
    distances = (_softmax(raw_dfl) * np.arange(reg_max + 1, dtype=np.float32)).sum(axis=-1) * stride_per_prior[None, :, None]
    boxes = np.concatenate((priors[None, :, :] - distances[:, :, :2], priors[None, :, :] + distances[:, :, 2:]), axis=-1)
    input_height, input_width = input_size
    boxes[:, :, [0, 2]] = np.clip(boxes[:, :, [0, 2]], 0, input_width)
    boxes[:, :, [1, 3]] = np.clip(boxes[:, :, [1, 3]], 0, input_height)
    if letterbox is not None:
        if (letterbox.target_height, letterbox.target_width) != input_size:
            raise ValueError("letterbox target dimensions do not match model input")
        boxes = letterbox.unmap_boxes(boxes)
    decoded: list[list[Detection]] = []
    for batch_boxes, batch_scores in zip(boxes, scores):
        detections: list[Detection] = []
        for label in range(len(class_names)):
            candidates = np.flatnonzero(batch_scores[:, label] >= score_threshold)
            if not candidates.size:
                continue
            ranked = candidates[np.argsort(-batch_scores[candidates, label], kind="stable")]
            candidates = ranked[:1000]
            kept = candidates[nms(batch_boxes[candidates], batch_scores[candidates, label], nms_iou_threshold)]
            detections.extend(
                Detection(tuple(float(value) for value in batch_boxes[index]), float(batch_scores[index, label]), label, class_names)
                for index in kept
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        decoded.append(detections if keep_top_k is None else detections[:keep_top_k])
    return decoded

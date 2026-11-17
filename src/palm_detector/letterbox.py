"""MediaPipe-style fixed-canvas letterbox geometry for palm detectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Forward mapping from an original image into a centered padded canvas."""

    original_height: int
    original_width: int
    target_height: int
    target_width: int
    scale: float
    pad_top: int
    pad_left: int
    resized_height: int
    resized_width: int

    def map_boxes(self, boxes: np.ndarray) -> np.ndarray:
        mapped = np.asarray(boxes, dtype=np.float32).copy()
        if mapped.size:
            mapped[..., [0, 2]] = mapped[..., [0, 2]] * self.scale + self.pad_left
            mapped[..., [1, 3]] = mapped[..., [1, 3]] * self.scale + self.pad_top
        return mapped

    def unmap_boxes(self, boxes: np.ndarray) -> np.ndarray:
        restored = np.asarray(boxes, dtype=np.float32).copy()
        if restored.size:
            restored[..., [0, 2]] = (restored[..., [0, 2]] - self.pad_left) / self.scale
            restored[..., [1, 3]] = (restored[..., [1, 3]] - self.pad_top) / self.scale
            restored[..., [0, 2]] = np.clip(restored[..., [0, 2]], 0, self.original_width)
            restored[..., [1, 3]] = np.clip(restored[..., [1, 3]], 0, self.original_height)
        return restored


def compute_letterbox_transform(
    original_height: int, original_width: int, target_height: int, target_width: int
) -> LetterboxTransform:
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError("image and target dimensions must be positive")
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, min(target_width, int(round(original_width * scale))))
    resized_height = max(1, min(target_height, int(round(original_height * scale))))
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    return LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        target_height=target_height,
        target_width=target_width,
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        resized_height=resized_height,
        resized_width=resized_width,
    )


def letterbox_bgr(
    image: np.ndarray,
    *,
    target_size: tuple[int, int],
    pad_value: int | tuple[int, int, int] = 0,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize BGR image by one scale and center it on a fixed black canvas."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HWC BGR image with three channels")
    target_height, target_width = target_size
    transform = compute_letterbox_transform(image.shape[0], image.shape[1], target_height, target_width)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for letterbox preprocessing") from exc
    resized = cv2.resize(image, (transform.resized_width, transform.resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    if pad_value != 0:
        canvas[...] = pad_value
    top, left = transform.pad_top, transform.pad_left
    canvas[top:top + transform.resized_height, left:left + transform.resized_width] = resized
    return canvas, transform

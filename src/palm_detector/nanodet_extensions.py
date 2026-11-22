"""Runtime extensions for MediaPipe-style NanoDet preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np


def install_nanodet_extensions() -> None:
    """Install the fixed-canvas LetterBox transform used by surface models."""
    from nanodet.data.transform import pipeline, warp

    if getattr(warp.ShapeTransform, "_palm_letterbox_installed", False):
        return

    original_init = warp.ShapeTransform.__init__
    original_call = warp.ShapeTransform.__call__
    original_pipeline_call = pipeline.Pipeline.__call__

    def patched_init(self: Any, keep_ratio: bool, *args: Any, **kwargs: Any) -> None:
        self.palm_letterbox = bool(kwargs.pop("palm_letterbox", False))
        original_init(self, keep_ratio, *args, **kwargs)

    def patched_call(self: Any, meta_data: dict[str, Any], dst_shape: tuple[int, int]) -> dict[str, Any]:
        if not self.palm_letterbox:
            return original_call(self, meta_data, dst_shape)

        raw_image = meta_data["img"]
        height, width = raw_image.shape[:2]
        target_width, target_height = dst_shape
        center = np.eye(3)
        center[0, 2] = -width / 2
        center[1, 2] = -height / 2
        center = warp.get_perspective_matrix(self.perspective) @ center
        center = warp.get_scale_matrix(self.scale_ratio) @ center
        center = warp.get_stretch_matrix(*self.stretch_ratio) @ center
        center = warp.get_rotation_matrix(self.rotation_degree) @ center
        center = warp.get_shear_matrix(self.shear_degree) @ center
        flip = warp.get_flip_matrix(self.flip_prob)
        center = flip @ center
        affine = warp.get_translate_matrix(self.translate_ratio, width, height) @ center

        scale = min(target_width / width, target_height / height)
        resized_width = max(1, min(target_width, int(round(width * scale))))
        resized_height = max(1, min(target_height, int(round(height * scale))))
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        letterbox = np.array([[scale, 0.0, pad_left], [0.0, scale, pad_top], [0.0, 0.0, 1.0]])
        matrix = letterbox @ affine
        meta_data["img"] = warp.cv2.warpPerspective(
            raw_image, matrix, dsize=(target_width, target_height), borderValue=(0, 0, 0)
        )
        meta_data["warp_matrix"] = matrix
        mask = warp.cv2.warpPerspective(
            np.full((height, width), 255, dtype=np.uint8), matrix,
            dsize=(target_width, target_height), flags=warp.cv2.INTER_NEAREST, borderValue=0
        )
        meta_data["_palm_letterbox_mask"] = mask > 0
        if "gt_bboxes" in meta_data:
            meta_data["gt_bboxes"] = warp.warp_boxes(meta_data["gt_bboxes"], matrix, target_width, target_height)
        if "gt_bboxes_ignore" in meta_data:
            meta_data["gt_bboxes_ignore"] = warp.warp_boxes(meta_data["gt_bboxes_ignore"], matrix, target_width, target_height)
        if "gt_masks" in meta_data:
            meta_data["gt_masks"] = [
                warp.cv2.warpPerspective(mask_item, matrix, dsize=(target_width, target_height))
                for mask_item in meta_data["gt_masks"]
            ]
        return meta_data

    def patched_pipeline_call(self: Any, dataset: Any, meta: dict[str, Any], dst_shape: tuple[int, int]) -> dict[str, Any]:
        result = original_pipeline_call(self, dataset, meta, dst_shape)
        mask = result.pop("_palm_letterbox_mask", None)
        if mask is not None:
            mean, std = self.color.keywords["kwargs"]["normalize"]
            black = -np.asarray(mean, dtype=np.float32) / np.asarray(std, dtype=np.float32)
            result["img"][~mask] = black
        return result

    warp.ShapeTransform.__init__ = patched_init
    warp.ShapeTransform.__call__ = patched_call
    warp.ShapeTransform._palm_letterbox_installed = True
    pipeline.Pipeline.__call__ = patched_pipeline_call

"""PaddleDetection extensions for anatomical flip and fixed LetterBox input."""

from __future__ import annotations

from typing import Any

import numpy as np

from .letterbox import letterbox_bgr


def install_paddledetection_extensions() -> None:
    """Register the v002 transforms and PicoDet evaluation architecture once."""
    import paddle
    from ppdet.data.transform import BaseOperator
    from ppdet.data import transform
    from ppdet.modeling.architectures.picodet import PicoDet

    if getattr(transform, "PalmLetterBox", None) is not None:
        return

    class PalmLetterBox(BaseOperator):
        """Center a scaled image on a fixed zero-padded canvas.

        ``letterbox_pad`` is retained for PalmLetterboxPicoDet, which removes
        the pad before development-set metric evaluation.
        """

        def __init__(self, target_size: int | list[int] | tuple[int, int], pad_value: int = 0):
            super().__init__()
            if isinstance(target_size, int):
                target_size = [target_size, target_size]
            if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
                raise TypeError("target_size must be an int or [height, width]")
            self.target_size = (int(target_size[0]), int(target_size[1]))
            self.pad_value = int(pad_value)

        def apply(self, sample: dict[str, Any], context: Any = None) -> dict[str, Any]:
            image, geometry = letterbox_bgr(sample["image"], target_size=self.target_size, pad_value=self.pad_value)
            sample["image"] = image.astype(np.float32)
            boxes = sample.get("gt_bbox")
            if boxes is not None and len(boxes):
                sample["gt_bbox"] = geometry.map_boxes(boxes)
            sample["im_shape"] = np.asarray([geometry.resized_height, geometry.resized_width], dtype=np.float32)
            sample["scale_factor"] = np.asarray([geometry.scale, geometry.scale], dtype=np.float32)
            sample["letterbox_pad"] = np.asarray([geometry.pad_top, geometry.pad_left], dtype=np.float32)
            return sample

    if not getattr(PicoDet, "_palm_letterbox_installed", False):
        def letterbox_forward(self: PicoDet):
            body_feats = self.backbone(self.inputs)
            fpn_feats = self.neck(body_feats)
            head_outs = self.head(fpn_feats, self.export_post_process)
            if self.training or not self.export_post_process:
                return head_outs, None
            pred_scores, pred_boxes = head_outs
            pads = self.inputs.get("letterbox_pad")
            if pads is not None:
                pad_y, pad_x = paddle.split(pads, 2, axis=-1)
                offsets = paddle.concat([pad_x, pad_y, pad_x, pad_y], axis=-1).reshape([-1, 1, 4])
                pred_boxes = pred_boxes - offsets
            return self.head.post_process(
                (pred_scores, pred_boxes), self.inputs["scale_factor"],
                export_nms=self.export_nms, nms_cpu=self.nms_cpu)

        PicoDet._forward = letterbox_forward
        PicoDet._palm_letterbox_installed = True

    transform.PalmLetterBox = PalmLetterBox

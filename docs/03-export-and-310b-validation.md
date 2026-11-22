# Export and Ascend 310B validation

The formal candidate export is static ONNX with input `data: float32[1,3,416,416]`.
It contains only the forward graph and emits `output: float32[1,3598,35]`.
The 35 fields per prior are three sigmoid class scores plus four 8-bin DFL
distributions (`reg_max=7`). Decode and class-aware NMS are host-side in
`src/palm_detector/postprocess.py`; this avoids making deployment depend on
ONNX NMS conversion behavior.

The preprocessing contract is OpenCV BGR, centered zero-padded LetterBox to
`416x416`, then channelwise BGR normalization with mean
`[103.53, 116.28, 123.675]` and standard deviation
`[57.375, 57.12, 58.395]`. The inverse LetterBox transform maps decoded boxes
back to original-image coordinates. The display layer must not alter image
geometry before inference.

The host DFL decoder uses NanoDet's grid origin `arange(size) * stride`, with
no half-cell offset. An older host decoder added `0.5 * stride`, shifting ONNX
review boxes right and down after scaling back to the source image. Corrected
decoding was checked against the pinned NanoDet head and 100 test images; the
checkpoint and ONNX weights did not change. Historical host-decoded overlays
and comparisons made before this correction should be regenerated before use.

## Required evidence

1. `scripts/export_onnx.sh` exports the selected checkpoint, runs ONNX checker
   and shape inference, and writes `onnx-check.json` plus a model card.
2. Run `scripts/compare_onnx.py` on at least 20 fixed representative images.
   Save raw output error, decoded box IoU, class agreement, scores, and image
   hashes in the run directory.
3. Hand the checkpoint-independent ONNX, model card, and comparison report to
   the Atlas task. They are not evidence that ATC is successful.
4. On Atlas 310B, run and record both default ATC and
   `--precision_mode=allow_fp32_to_fp16` attempts, with CANN/driver/firmware,
   exact command, logs, failure operator if any, and each OM SHA-256.
5. For every successfully compiled OM, compare ACL raw outputs and host decoded
   detections to ONNX on the same images. Recompute task quality and throughput
   separately for every precision path.

An ONNX export, a successful ATC conversion, ACL numeric agreement, test-set
quality, and performance are distinct gates. A candidate is not "adapted to
310B" until all relevant records exist and pass their agreed acceptance limits.

The local exporter `scripts/export_nanodet_onnx.py` mirrors the pinned upstream
export contract while avoiding a modification to its checkout. It disables only
the upstream tool's verbose graph print; it prefers opset 11, uses opset 13
only when a backbone needs `aten::unflatten`, and preserves fixed shape, the
same input/output names, graph simplification, and host-side postprocessing.
The actual opset is recorded in `onnx-check.json` for each model.

The target board operator must supply the actual `--soc_version`; this project
intentionally does not guess it from the training server.

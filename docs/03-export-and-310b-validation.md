# Export and Ascend 310B validation

The candidate export is static ONNX with input `data: float32[1,3,320,320]`.
It contains only the forward graph and emits `output: float32[1,2125,34]`.
The 34 fields per prior are two sigmoid class scores plus four 8-bin DFL
distributions (`reg_max=7`). Decode and class-aware NMS are host-side in
`src/palm_detector/postprocess.py`; this avoids making deployment depend on
ONNX NMS conversion behavior.

The preprocessing contract is OpenCV BGR, direct bilinear resize to `320x320`,
then channelwise BGR normalization with mean `[103.53, 116.28, 123.675]` and
standard deviation `[57.375, 57.12, 58.395]`. There is no image flip and no
letterbox padding in the baseline. The display layer must not mirror an image
without updating how anatomical labels are presented.

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
the upstream tool's verbose graph print; it keeps opset 11, fixed shape, the
same input/output names, graph simplification, and host-side postprocessing.

`scripts/build_310b_handoff.py` prepares an ignored package containing the
ONNX, model card, 20 NCHW preprocessed `.npy` inputs, ONNX reference raw
outputs, expected host-side detections, and a hash manifest. Each input is also
provided as a little-endian float32 `.bin` file for ACL runners. Copy that package
to the board, use `scripts/compile_atc_310b.sh` after sourcing the board CANN
environment, run an ACL inference runner on every input, and compare each
output with `scripts/compare_310b_raw_output.py`. The target board operator
must supply the actual `--soc_version`; this project intentionally does not
guess it from the training server.

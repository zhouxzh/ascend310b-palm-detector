# MediaPipe LetterBox v002

This is a fresh two-model experiment, not a modification of any v001 output.
It retrains NanoDet-Plus-m 320 and PicoDet-S 320 on the same two-class palm
dataset with fixed-canvas LetterBox geometry matching the behavior of the
MediaPipe palm detector.

## Geometry

MediaPipe's palm graph preserves aspect ratio, places the scaled image inside a
square zero-padded input tensor, and removes the padding when projecting
detections back to the original image. v002 uses the same geometry with a
`320x320` tensor because both retrained models use static 320 inputs.

- Scale each image by `min(320 / width, 320 / height)`.
- Center it on a `320x320` BGR-black canvas.
- Transform ground-truth boxes by the same scale and offset.
- In NanoDet ONNX inference, remove the offset and divide by the one scale.
- In PicoDet evaluation, the registered LetterBox extension removes the offset
  before the standard PicoDet scale-factor postprocess, while retaining the
  normal `PicoDet` architecture name for image-only ONNX export.

MediaPipe's detector normalizes zero padding into its `[0, 1]` tensor. These
models retain black pixels before their own documented normalization, producing
the corresponding normalized black value. Geometry, padding placement, and
box projection are equivalent.

## Augmentation

The objective is to improve robustness without corrupting the visual evidence
used to distinguish anatomical left and right hands.

- Horizontal flip: probability `0.5`, with the image and boxes mirrored and
  `left_palm`/`right_palm` class IDs exchanged in the same operation.
- No anisotropic stretch: it is fixed at `1.0` for NanoDet v002.
- NanoDet: mild isotropic scale `0.9-1.1`, rotation up to 10 degrees,
  translation up to 5 percent, brightness, contrast, and saturation jitter.
- PicoDet: random crop and color distortion, followed by semantic flip and
  fixed LetterBox.
- Validation, export, review, benchmark, and deployment use no random
  augmentation and the identical fixed LetterBox contract.

## Runs

Run the two models serially on GPU 0:

```bash
./scripts/run_mediapipe_letterbox_v002.sh
```

The runner creates an independent NanoDet run under `artifacts/runs/` and an
independent PicoDet v002 run under `artifacts/picodet-v002/runs/`. It does not
overwrite v001 models, metrics, or ONNX files. The PicoDet run starts only
after the NanoDet run has reached its configured early-stop or epoch limit.

Use `scripts/monitor_mediapipe_letterbox_v002_gpu.sh` to record GPU state and
stop the complete v002 controller after three consecutive samples at or above
82C.

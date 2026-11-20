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
It then exports a static ONNX model, validates 20 Paddle/ONNX raw tensors,
evaluates the frozen left/right test split, and compares decoded ONNX and
native Paddle detections on 20 test images. A one-time checkpoint resume is
attempted only if a transient CUDA failure occurs before a terminal training
summary is written.

## Completed PicoDet-S 320 Run

The prior v002 PicoDet materialized configuration was trained as a separate
run. It remains a two-class anatomical handedness detector; it is not
comparable to the later `palm`/`back`/`side` surface models.

| Item | Result |
| --- | --- |
| Run | `picodet-v002-picodet-s-320-seed-42` |
| Classes | `left_palm`, `right_palm` |
| Input | `float32 [1, 3, 320, 320]`, RGB, centered zero-padded LetterBox, `/255`, ImageNet mean/std, NCHW |
| Checkpoint selection | Early-stopped at epoch 85; best development mAP `0.6717` at epoch 45 |
| Frozen test mAP50-95 | `0.6642` |
| Frozen test AP50 | `0.9848` |
| Left / right AP50 | `0.9877` / `0.9820` |
| ONNX | `artifacts/picodet-v002/runs/picodet-v002-picodet-s-320-seed-42/picodet-320x320.onnx` |

The ONNX export passed checker validation and 20 raw Paddle/ONNX tensor
comparisons with maximum absolute error `4.53e-6`. Its image-level decoder
gate also passed on 20 rare-class-first test images: maximum raw error
`7.27e-5`, minimum decoded-box IoU `0.999997`, and maximum coordinate error
`6.1e-5` pixels. The deployment helper is:

```bash
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/run_picodet_onnx.py \
  --run-dir artifacts/picodet-v002/runs/picodet-v002-picodet-s-320-seed-42 \
  --image <BGR-image-path>
```

Use `scripts/monitor_mediapipe_letterbox_v002_gpu.sh` to record GPU state and
stop the complete v002 controller after three consecutive samples at or above
82C.

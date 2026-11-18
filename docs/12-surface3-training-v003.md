# Surface3 Training v003

This formal run trains `palm`, `back`, and `side` surface classes. It does not
distinguish anatomical left and right hands. `unknown` is intentionally not a
detector class: it remains a deployment rejection state and is excluded from
both supervision and metrics to avoid turning ambiguous visible hands into
background negatives.

## Data

The source `coco-handed-surface-v001` annotations are user-confirmed human
labels. `scripts/prepare_surface3_coco.py` produces an immutable derivative:

- Unique training: palm 5,774, back 941, side 32.
- Balanced training: side-containing images repeat 32 times, producing palm
  5,929, back 941, side 1,024 annotations.
- Development: palm 503, back 76, side 2.
- Gate: palm 187, back 34, side 0.
- Test: palm 1,280, back 263, side 13.

Development/gate scores select the checkpoint using aggregate mAP. Side AP is
too low-sample for tuning; it is reported separately on the natural,
subject-disjoint test split after the model is frozen.

Only a high-confidence `palm` prediction may proceed to palmprint recognition.
`back`, `side`, no detection, low confidence, or a close class decision are
all surfaced to the calling application as the non-model `unknown` rejection
state.

## Models And Augmentation

| Framework | Variant | Input | Optimizer recipe |
| --- | --- | ---: | --- |
| NanoDet | Plus-m1.5 | 416 | AdamW 0.001, weight decay 0.05, 500-step warmup, cosine 300 epochs |
| PicoDet | L / LCNet x2.0 | 416 | base LR 0.03 on one GPU, upstream 250-epoch schedule |

Both models use MediaPipe-style centered zero-padded LetterBox at 416 square.
Training applies label-preserving horizontal flip at 0.5 probability, mild
isotropic scale (NanoDet), rotation/translation (NanoDet), and photometric
distortion. Neither uses anisotropic stretch. Validation, ONNX preprocessing,
and deployment use only deterministic LetterBox.

## Launch

```bash
./scripts/run_surface3_v003.sh
```

The controller runs NanoDet first, exports a verified ONNX model, then runs
PicoDet and exports a verified ONNX model. It is resumable and paired with
`scripts/monitor_surface3_v003_gpu.sh`, which stops the v003 controller after
three 30-second GPU samples at or above 82C.

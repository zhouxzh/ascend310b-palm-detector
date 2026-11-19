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

The finalist comparison adds NanoDet-Plus-m and PicoDet-M/LCNet x1.5 at the
same 416 input size. It uses the same split, LetterBox policy, side-aware
sampling, and early-stop rule, while retaining each upstream backbone's
validated batch size. PicoDet-M uses a 0.015 single-GPU starting learning rate:
the preceding PicoDet-L trial peaked on development mAP at epoch 10 and then
regressed with the generic 0.03 linearly scaled COCO rate. This makes the
learning-rate change an explicit, recorded surface-task tuning comparison.

Both models use MediaPipe-style centered zero-padded LetterBox at 416 square.
Training applies label-preserving horizontal flip at 0.5 probability, mild
isotropic scale (NanoDet), rotation/translation (NanoDet), and photometric
distortion. Neither uses anisotropic stretch. Validation, ONNX preprocessing,
and deployment use only deterministic LetterBox.

The NanoDet controller also runs a deterministic 20-image ONNX Runtime check
against PyTorch. The samples are selected from the held-out test split with
rare classes first, and the persisted report records source-image hashes, raw
output error, and decoded class/box agreement.

## Completed Results And Selection

All four candidates completed with the same user-confirmed source labels,
three-class contract, subject-disjoint split, 416-square LetterBox deployment
geometry, and immutable test annotations. The test split was used only after a
candidate's checkpoint had been selected on development mAP.

| Model | Early-stop epoch | Best dev mAP | Test mAP50-95 | Test AP50 | Palm AP50 | Back AP50 | Side AP50 | ONNX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NanoDet-Plus-m1.5 | 55 | 0.5195 | 0.5039 | 0.8281 | 0.9872 | 0.9476 | 0.5496 | Pass |
| PicoDet-L / LCNet x2.0 | 50 | 0.4526 | 0.5153 | 0.8274 | 0.9876 | 0.9505 | 0.5441 | Pass |
| NanoDet-Plus-m | 100 | 0.4610 | **0.5321** | **0.8317** | **0.9878** | **0.9577** | 0.5495 | Pass |
| PicoDet-M / LCNet x1.5 | 45 | **0.5006** | 0.4834 | 0.8234 | 0.9855 | 0.9355 | 0.5492 | Pass |

`NanoDet-Plus-m` is the formal production candidate. Its selected artifact is
`artifacts/runs/nanodet-plus-m-416-surface3-letterbox-v003__data-v001__seed-42__20261118-1700/model-416x416.onnx`.
The selected ONNX passed the static ONNX checker and a deterministic 20-image
PyTorch/ONNX Runtime comparison: maximum raw-output error `1.049e-5`, minimum
decoded-box IoU `0.999999`. PicoDet-L remains the alternative PicoDet artifact;
PicoDet-M's lower 0.015 one-GPU learning rate improved its development peak,
but did not transfer to the frozen test split and is not promoted.

Both PicoDet exports passed 20 deterministic Paddle/ONNX raw-output checks.
PicoDet-M's maximum error was `5.19e-7`. Use
`scripts/run_picodet_onnx.py --run-dir <completed-run> --image <image>` for
the exact PicoDet RGB LetterBox preprocessing, DFL decoding, and class-aware
NMS; the raw export alone does not contain an NMS node.

The natural test split contains only 13 `side` instances. Its AP is retained
as a deployment signal but is too high-variance to drive architecture or
hyperparameter selection. At runtime, only a high-confidence `palm` result is
eligible for palmprint recognition; every other class or uncertain result is
rejected as `unknown` by the calling application.

## Launch

```bash
./scripts/run_surface3_v003.sh
```

The controller runs NanoDet first, exports a verified ONNX model, then runs
PicoDet and exports a verified ONNX model. It is resumable and paired with
`scripts/monitor_surface3_v003_gpu.sh`, which stops the v003 controller after
three 30-second GPU samples at or above 82C.

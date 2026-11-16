# Training Recipes And Log Retention

This record preserves the actual v001 training recipes. The recipes are not
normalized across architectures: each NanoDet and PicoDet variant retains the
optimizer and schedule designed for that upstream model. This avoids turning a
model comparison into a comparison of arbitrary forced settings.

All values below are configuration values used for training, not suggestions
for future tuning. Generated run configurations under `artifacts/` remain the
per-run source of truth.

## Common Contract

- Dataset: `coco-handed-v001`, with anatomical `left_palm` and `right_palm`.
- Split: subject-disjoint audited development split; the gate and test splits
  are not used for schedule tuning.
- Horizontal flip: disabled. A conventional flip changes anatomical
  handedness and NanoDet/PicoDet do not swap class IDs during the transform.
- Resize: direct, fixed square input for each variant; no letterboxing.
- Seed: `42` for the sweep recipes.
- Validation: every 5 epochs.
- Selection: the best development mAP checkpoint, never the final epoch.

## NanoDet v001

The sweep keeps an effective batch of 32. The 512 model uses a microbatch of
16 with 2 gradient-accumulation steps; the 640 model uses 8 with 4 steps; all
other listed models use a microbatch of 32 without accumulation. Training is
FP32 on one GPU, uses gradient clipping at 35, and logs every 50 steps.

| Variants | Optimizer | LR / weight decay | Warmup | Schedule | Epochs |
| --- | --- | --- | --- | --- | ---: |
| Plus-m 320, Plus-m 416, Plus-m-1.5x 320, Plus-m-1.5x 416 | AdamW | 0.001 / 0.05 | linear, 500 steps, 0.0001 | cosine, eta min 0.00005 | 300 |
| Plus-ConvNeXt-nano 640 | AdamW | 0.001 / 0.05; backbone LR multiplier 0.1; no norm decay | linear, 500 steps, 0.0001 | cosine, eta min 0.0005 | 50 |
| Legacy-m 320 and 416; Legacy-m-1.5x 320 | SGD, momentum 0.9 | 0.02333333 / 0.0001 | linear, 300 steps, 0.1 | MultiStep at 240, 260, 275; gamma 0.1 | 280 |
| Legacy-m-1.5x 416 | SGD, momentum 0.9 | 0.02545455 / 0.0001 | linear, 300 steps, 0.1 | MultiStep at 240, 260, 275; gamma 0.1 | 280 |
| Legacy-m-0.5x 320 | SGD, momentum 0.9 | 0.02333333 / 0.0001 | linear, 1000 steps, 0.00001 | MultiStep at 130, 160, 175; gamma 0.1 | 180 |
| Legacy-G 416 | SGD, momentum 0.9 | 0.025 / 0.0001 | linear, 500 steps, 0.01 | MultiStep at 130, 160, 175, 185; gamma 0.1 | 190 |
| Legacy-Transformer 320 | SGD, momentum 0.9 | 0.028 / 0.0001 | linear, 500 steps, 0.01 | MultiStep at 140, 170, 180, 185; gamma 0.1 | 190 |
| Legacy-EfficientLite0 320 | SGD, momentum 0.9 | 0.032 / 0.0001 | linear, 500 steps, 0.01 | MultiStep at 140, 170, 180, 185; gamma 0.1 | 190 |
| Legacy-EfficientLite1 416; RepVGG-A0 416 | SGD, momentum 0.9 | 0.0224 / 0.0001 | linear, 500 steps, 0.01 | MultiStep at 130, 150, 160, 165; gamma 0.1 | 170 |
| Legacy-EfficientLite2 512 | SGD, momentum 0.9 | 0.032 / 0.0001 | linear, 300 steps, 0.1 | MultiStep at 90, 110, 120, 130; gamma 0.1 | 135 |

The generated configuration inventory is defined by
`configs/nanodet-sweep-v001.yml` and produced by
`scripts/prepare_nanodet_sweep.py`. It pins the dataset, fixed effective batch,
class contract, input geometry, and evaluation cadence while preserving each
upstream optimizer family and epoch schedule.

## PicoDet v001

PicoDet uses the PaddleDetection release/2.9 recipes with one GPU. The listed
reference learning rate is for four GPUs; generated configs use one quarter of
it as the actual base LR. All variants use two anatomical classes, direct fixed
resize, disabled random flip, the listed ImageNet-pretrained LCNet backbone,
and early stopping after 8 development validations without a 0.001 mAP gain.

| Variant | Input | Train batch | Reference LR | Actual base LR | Epochs | Pretrained LCNet |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| XS-320 | 320 | 64 | 0.32 | 0.08 | 300 | x0.35 |
| XS-416 | 416 | 56 | 0.28 | 0.07 | 300 | x0.35 |
| S-320 | 320 | 64 | 0.32 | 0.08 | 300 | x0.75 |
| S-416 | 416 | 48 | 0.24 | 0.06 | 300 | x0.75 |
| M-320 | 320 | 48 | 0.24 | 0.06 | 300 | x1.5 |
| M-416 | 416 | 48 | 0.24 | 0.06 | 250 | x1.5 |
| L-320 | 320 | 24 | 0.12 | 0.03 | 250 | x2.0 |
| L-416 | 416 | 24 | 0.12 | 0.03 | 250 | x2.0 |
| L-640 | 640 | 12 | 0.06 | 0.015 | 200 | x2.0 |

The immutable matrix is `configs/picodet-sweep-v001.yml`; generated run
configs are made by `scripts/prepare_picodet_sweep.py`.

## TensorBoard Retention

TensorBoard is operational evidence, not a permanent archive for every trial.
After a family comparison is decided, retain only the event files for the
selected ONNX candidate in that family and delete event files for weaker,
superseded, failed, or hyperparameter-exploration runs. Keep an active run's
event file only until it completes and is either exported and selected or
discarded.

Current retained completed candidates are:

| Family | Run | Development mAP | ONNX |
| --- | --- | ---: | --- |
| NanoDet | `nanodet-plus-m-320-portable-handed__data-v001__seed-43__20261114-1300` | 0.687546 | `nanodet-plus-m-320-palm.onnx` |
| PicoDet | `picodet-v001-picodet-s-320-seed-42` | 0.665458 | `picodet-320x320.onnx` |

PicoDet-S-416 is still training at the time of this record. Its event files
are retained temporarily so the active run remains observable. Once it has
completed, export and validate its ONNX model before selecting the family
candidate, then delete whichever PicoDet event files are not selected.

The TensorBoard launch scripts only scan the retained event files that remain
under `artifacts/runs` and `artifacts/picodet-v001/runs`; neither `artifacts/`
nor model outputs are committed to Git.

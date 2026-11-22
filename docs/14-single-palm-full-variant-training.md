# Single-Palm Full Variant Training

This experiment defines one class, `palm_surface`, for direct palm-region
detection. It is separate from the formal three-class palm-surface model.

The source labels are the human-confirmed `palm`/`back`/`side` annotations.
`palm` and `back` boxes become `palm_surface`; every image that contains a
`side` box is excluded. This ensures that the single-class detector is not
trained to treat a visible side hand as background.

The matrix contains 78 tasks, all using seed 42, fixed subject-disjoint splits,
development-set checkpoint selection, early stopping after eight validation
checks without a 0.001 mAP improvement, ONNX export, framework/ONNX agreement,
and one final frozen-test evaluation:

| Family | Variants | Geometry | Tasks |
| --- | ---: | --- | ---: |
| NanoDet | 16 upstream configurations | LetterBox and direct stretch | 32 |
| PicoDet | LCNet XS/S/M/L plus legacy ESNet and documented COCO backbone extensions | LetterBox and direct stretch | 46 |

The NanoDet entries are the 16 independent public Model Zoo recipes in the
pinned checkout: the five NanoDet-Plus variants, ConvNeXt-Nano, and the ten
legacy recipes. `nanodet-plus-m_416-yolo.yml` is intentionally not an extra
model: it has the same NanoDet-Plus-m-416 architecture and schedule as the
COCO recipe, changing only its annotation reader to Darknet TXT. The audited
input here is COCO and it would otherwise duplicate an experiment. PicoDet
similarly includes independent COCO detection recipes only; VOC, NPU, pruning,
layout/main-body, and pedestrian application files are task/export variants,
not extra palm-detector architectures.

Direct-stretch tasks use the same augmentations as LetterBox tasks except for
geometry. Random anisotropic stretch remains disabled because direct resize
already applies independent x/y scaling.

Launch the resumable controller with:

```bash
nohup ./scripts/run_single_palm_all_variants.sh \
  > artifacts/single-palm-v001/controller.log 2>&1 &
./scripts/start_single_palm_tensorboard.sh 6008
```

The controller writes status to `artifacts/single-palm-v001/training-state.json`
and the final rank-ordered comparison to `final-comparison.json` and
`final-comparison.md`. In addition to every model's frozen-test mAP, the report
contains matched LetterBox-versus-direct-stretch deltas and per-framework mean,
median, best-result, and geometry win/loss aggregates. Only completed matched
pairs contribute to a geometry conclusion.

Each run retries from its existing checkpoint up to three times after an
interrupted training, export, or validation stage. Change the limit or delay
with `SINGLE_PALM_MAX_RUN_ATTEMPTS` and `SINGLE_PALM_RETRY_DELAY_SECONDS`; a
task is marked failed only after all attempts are exhausted.

PicoDet pipelines use two shared GPU slots by default. This cap was validated
with two PicoDet M-320 runs plus two NanoDet runs at about 22.3 GiB of 32.6
GiB GPU memory; four concurrent PicoDet jobs previously exhausted memory.
Set `SINGLE_PALM_PICODET_PARALLELISM=1` for conservative serial recovery.

The 416-pixel PicoDet-M recipe uses a microbatch of 32 rather than its 48-image
reference microbatch because 48 overflowed the available GPU memory alongside
the planned NanoDet lanes. Its learning rate is linearly scaled from 0.06 to
0.04, and this exact recipe is shared by LetterBox and direct-stretch runs.

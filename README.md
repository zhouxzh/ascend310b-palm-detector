# ascend310b-palm-detector

Train and validate a static `320 x 320` NanoDet-Plus-m detector for the palm
region, with two anatomical-hand classes:

| COCO ID | Internal index | Class |
| ---: | ---: | --- |
| 1 | 0 | `left_palm` |
| 2 | 1 | `right_palm` |

`left_palm` and `right_palm` always mean the person's anatomical left and
right palm, never image-side position. The box is a MediaPipe palm-region box,
not a full-hand box or an aligned palmprint recognition crop.

## Current readiness

The portable HaGRIDv2 MediaPipe dataset has 6,997 train, 825 validation, and
1,600 test images after requiring matched anatomical handedness and auditing
image hashes plus subject/session separation. A 150-epoch two-class run and
independent COCO test completed; its best checkpoint is under `artifacts/runs/`.
The separate palm/back/side view labels are automatic candidates, with no
recorded human reviews yet.

The remaining release gates are documented in
[docs/04-training-readiness.md](docs/04-training-readiness.md): human-reviewed
surface-view labels and independent Atlas 310B ATC/ACL/quality/performance
evidence. The scripts fail closed on an invalid class table, missing image
metadata, cross-split subject/session leakage, or an unverified pretrained
checkpoint.

For the requested public single-class dataset training path, see
[docs/07-portable-hagridv2-nanodet.md](docs/07-portable-hagridv2-nanodet.md).

For local browser review of provisional palm/back labels, see
[docs/08-hand-surface-annotations.md](docs/08-hand-surface-annotations.md).

## Reproducible setup

Use a dedicated environment; do not mutate the system Python, system CUDA, or
the Atlas CANN installation.

```bash
conda env create -f environment.yml
conda activate palm-nanodet
./scripts/fetch_nanodet.sh
./scripts/fetch_paddledetection.sh
./scripts/setup_unified_env.sh palm-nanodet
./scripts/acquire_nanodet_pretrained.sh
```

NanoDet is fetched, rather than committed, at the exact upstream revision in
[docs/05-upstream-nanodet.md](docs/05-upstream-nanodet.md).

The verified direct Python dependency set is also in
`requirements.txt`. On this server, `palm-nanodet` is the single shared
environment for NanoDet, PicoDet, ONNX export, and future YOLO training. Run
`scripts/setup_unified_env.sh` rather than invoking `pip install -r` directly:
the two verified framework wheels use different CUDA runtime metadata, so the
script preserves their known-good binaries and validates real GPU arithmetic.

## Data preparation

Use the pinned portable dataset and conversion in
[docs/07-portable-hagridv2-nanodet.md](docs/07-portable-hagridv2-nanodet.md).
The current two-class release is `data/portable-hagridv2-mediapipe-hand/coco-handed-v001`.
Audit its split, image decoding, and SHA-256 checks with:

```bash
PYTHONPATH=src python scripts/check_coco_dataset.py \
  --split train data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_train.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
  --split val data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_val.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
  --split test data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_test.json data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
  --verify-image-hashes \
  --output reports/data-audit-v001.json
```

Create 320x320 review overlays using the training resize geometry:

```bash
PYTHONPATH=src python scripts/visualize_coco.py \
  --annotations data/portable-hagridv2-mediapipe-hand/coco-handed-v001/annotations/instances_train.json \
  --images-root data/portable-hagridv2-mediapipe-hand/coco-handed-v001/images \
  --output-dir artifacts/data-review/train-320
```

For a pipeline-only GPU smoke test, use `scripts/run_nanodet_pipeline_smoke.sh`.
Its output is explicitly labelled as a pipeline check, not a quality result.

Use `scripts/render_coco_predictions.py` with a frozen annotation/prediction
pair to generate visual error-review overlays; ground truth uses a thick box and
model predictions a thin box in the same class color.

## Train, evaluate, and export

The exact NanoDet/PicoDet optimizer, warmup, learning-rate schedule, batch
scaling, epoch limits, selected ONNX candidates, and TensorBoard retention
policy are recorded in [docs/10-training-recipes-and-retention.md](docs/10-training-recipes-and-retention.md).

Before training, fill a real immutable pretrained-weight manifest from
`data/manifests/pretrained-weights.template.json`. Its file hash must match the
checkpoint passed to the launcher.

```bash
./scripts/train_nanodet_portable_handed.sh \
  nanodet-plus-m-320-portable-handed__data-v001__seed-42__YYYYMMDD-HHMM \
  artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt \
  data/manifests/nanodet-plus-m_416_checkpoint.json

./scripts/evaluate_nanodet.sh artifacts/runs/<run-id> test
./scripts/export_onnx.sh artifacts/runs/<run-id>
```

`export_onnx.sh` produces static `1x3x320x320` ONNX, validates it with ONNX
checker and shape inference, and writes a model card. Compare raw outputs and
host-side decoded detections on fixed representative images before handing the
candidate to Atlas 310B; see [docs/03-export-and-310b-validation.md](docs/03-export-and-310b-validation.md).

## Local checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
./scripts/check_training_readiness.sh
```

Generated data, checkpoints, ONNX files, OM files, images, reports, and
training logs are intentionally ignored by Git.

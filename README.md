# ascend310b-palm-detector

This repository trains and validates palm-surface detectors for palmprint
admission. The only supported class contract is:

| COCO ID | Class | Deployment action |
| ---: | --- | --- |
| 1 | `palm` | Eligible for downstream palmprint recognition |
| 2 | `back` | Reject as `unknown` |
| 3 | `side` | Reject as `unknown` |

Low confidence, no detection, or a close class decision is also rejected as
`unknown` by the calling application. The task intentionally does not infer
anatomical left or right hand.

## Released Model

The released models are NanoDet-Plus-m and PicoDet-L / LCNet x2.0. Both use
the same human-confirmed data, static `416x416` ONNX input, and centered
zero-padded LetterBox geometry. They differ in runtime contract and
normalization, so each must use its matching host decoder.

| Model | ONNX input | Frozen test mAP50-95 | Frozen test AP50 | Role |
| --- | --- | ---: | ---: | --- |
| NanoDet-Plus-m | `data`, `1x3x416x416` | **0.5321** | **0.8317** | Primary model |
| PicoDet-L / LCNet x2.0 | `image`, `1x3x416x416` | 0.5153 | 0.8274 | Formal alternative |

NanoDet-Plus-m has the stronger frozen test result:

```text
data: float32 [1, 3, 416, 416]
output: float32 [1, 3598, 35]
classes: palm, back, side
```

The exact artifacts, hashes, preprocessing contracts, and ONNX checks are
recorded in [docs/12-surface3-training-v003.md](docs/12-surface3-training-v003.md).

## Geometry Results

The formal data is human-confirmed and uses deterministic evaluation. The
primary release uses LetterBox. A paired direct-stretch ablation used the same
data, seed, model recipes, early stopping, and frozen test split without adding
random anisotropic stretch. Direct stretch did not beat the formal candidate:

| Model | LetterBox test mAP | Direct-stretch test mAP |
| --- | ---: | ---: |
| NanoDet-Plus-m 320 | 0.5184 | 0.4514 |
| NanoDet-Plus-m1.5 320 | 0.5292 | 0.5171 |
| PicoDet-L 320 | 0.5189 | 0.4994 |
| PicoDet-M 320 | 0.4731 | 0.4789 |

The direct-stretch comparison is retained as experiment evidence only. Random
anisotropic stretch is deliberately excluded from the primary comparison,
because direct resize already changes horizontal and vertical scale according
to source aspect ratio.

## Setup

```bash
conda env create -f environment.yml
conda activate palm-nanodet
./scripts/fetch_nanodet.sh
./scripts/fetch_paddledetection.sh
./scripts/setup_unified_env.sh palm-nanodet
./scripts/acquire_nanodet_pretrained.sh
PYTHONPATH=src python scripts/acquire_picodet_pretrained.py
```

`palm-nanodet` is the single verified environment for NanoDet, PicoDet, ONNX,
and later YOLO work. It validates both PyTorch and Paddle GPU arithmetic.

## Palm/Reject ROI Classifier

The palm/reject classifier is a separate ROI gate. It does not alter the
detector's `palm`, `back`, and `side` output contract. The pinned dataset uses
`0=palm` and `1=reject`; `reject` covers back, side, unsupported gestures,
occlusion, and uncertain hand regions.

Download the pinned dataset with the CUDA-enabled `torch` environment:

```bash
conda run -n torch python scripts/download_palm_reject_roi.py
```

The classifier supports three comparable backbones:

```text
nanodet_shufflenetv2_x1
lcnet_100
efficientnet_b0
```

Run one model:

```bash
conda run -n torch python scripts/train_palm_reject_classifier.py \
  --backbone lcnet_100 \
  --dataset data/palm-reject-roi-v005 \
  --output artifacts/palm-reject-lcnet-100
```

Run the hyperparameter search and final three-model matrix detached:

```bash
nohup bash scripts/start_palm_reject_optimization_ace2.sh \
  > artifacts/palm-reject-optimization-launcher.log 2>&1 &
```

The search evaluates four learning-rate, weight-decay, and label-smoothing
configurations per backbone. It selects by validation ROC-AUC without evaluating
the test split. The final run then evaluates test once. The operational
calibration maximizes palm recall subject to a configurable false-accept limit
for reject samples; the default is `reject -> palm <= 2%`.

The `realistic` augmentation profile combines mild affine and perspective
changes, photometric distortion, Gaussian blur, sharpness changes, JPEG
compression, grayscale, and horizontal flip. It does not randomly remove the
palm region so the positive class remains consistent with the annotation rule.

TensorBoard is served on port `6010`. The latest result snapshot and downloadable
files are served on port `6011`; snapshots are refreshed every 600 seconds.

The dataset is stored in `data/palm-reject-roi-v005`. Each model run writes its
best/last checkpoint, epoch status, TensorBoard events, frozen validation/test
metrics, and inference contract below
`artifacts/20261123-palm-reject-aug-optimization/`.

Export and numerically verify the selected checkpoint against real test ROIs:

```bash
conda run -n torch python scripts/export_palm_reject_classifier_onnx.py \
  --checkpoint artifacts/20261123-palm-reject-aug-optimization/final/efficientnet-b0/best.pt \
  --output artifacts/20261123-palm-reject-aug-optimization/onnx/palm-reject-efficientnet-b0.onnx \
  --metadata artifacts/20261123-palm-reject-aug-optimization/onnx/palm-reject-efficientnet-b0.metadata.json \
  --metrics artifacts/20261123-palm-reject-aug-optimization/final/efficientnet-b0/metrics.json \
  --dataset data/palm-reject-roi-v005
```

The exporter checks the ONNX graph, compares PyTorch and ONNX Runtime outputs on
32 real test ROIs, and records argmax and threshold-decision agreement. It does
not run ATC or claim Ascend 310B admission.

## Formal Workflows

```bash
./scripts/run_surface3_v003.sh
./scripts/run_surface3_320_v004.sh
./scripts/run_surface3_320_stretch_v005.sh
```

The controllers are resumable. Each completed model receives checkpoint
selection on development mAP, static ONNX export, framework/ONNX numerical
checks, and frozen test evaluation.

TensorBoard is available at `http://127.0.0.1:6006` and
`http://127.0.0.1:6007` when its local servers are running.

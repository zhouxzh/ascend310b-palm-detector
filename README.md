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

The formal release candidate is NanoDet-Plus-m with a static `416x416` ONNX
input and centered zero-padded LetterBox preprocessing:

```text
data: float32 [1, 3, 416, 416]
output: float32 [1, 3598, 35]
classes: palm, back, side
```

Its frozen subject-disjoint test result is mAP50-95 `0.5321` and AP50
`0.8317`. The exact artifact, hash, preprocessing contract, and ONNX checks
are recorded in [docs/12-surface3-training-v003.md](docs/12-surface3-training-v003.md).

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

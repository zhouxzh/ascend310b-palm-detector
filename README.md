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

## Preprocessing Contract

All supported training, validation, ONNX export, and inference use centered
zero-padded LetterBox preprocessing. Direct-resize model configurations and
artifacts have been removed. Training-time augmentation retains only
label-preserving transforms compatible with this deployment contract.

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
```

The controllers are resumable. Each completed model receives checkpoint
selection on development mAP, static ONNX export, framework/ONNX numerical
checks, and frozen test evaluation.

TensorBoard is available at `http://127.0.0.1:6006` and
`http://127.0.0.1:6007` when its local servers are running.

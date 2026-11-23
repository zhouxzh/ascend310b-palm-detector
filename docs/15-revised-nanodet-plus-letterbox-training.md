# Revised NanoDet-Plus LetterBox Training

## Scope

The initial full-variant experiment established a consistent advantage for
LetterBox preprocessing over direct stretch on the completed matched pairs.
To keep the remaining work focused, no additional direct-stretch jobs and no
additional non-NanoDet-Plus jobs are scheduled. Existing completed artifacts
are retained as historical baselines; they are not retrained or discarded.

The required active training target is every independent NanoDet-Plus recipe
under LetterBox preprocessing, with the audited one-class `palm_surface` COCO
dataset, seed 42, subject-disjoint splits, development-set checkpoint
selection, and frozen-test evaluation:

| Variant | Input | Required geometry | Training status rule |
| --- | ---: | --- | --- |
| `plus-m-320` | 320x320 | LetterBox | Required |
| `plus-m15-320` | 320x320 | LetterBox | Required |
| `plus-m-416` | 416x416 | LetterBox | Required |
| `plus-m15-416` | 416x416 | LetterBox | Required |
| `plus-convnext-nano-640` | 640x640 | LetterBox | Required |

`nanodet-plus-m_416-yolo.yml` is not an additional experiment: it has the
same model and schedule as the COCO `plus-m-416` recipe but reads Darknet TXT.
The audited dataset is COCO, so including it would duplicate the experiment.

## Acceptance

Each required model is counted only when all of the following artifacts exist
and pass validation:

1. `training-summary.json` with early-stopped or max-epoch completion.
2. Static ONNX model and `onnx-check.json` with `valid: true`.
3. `onnx-comparison.json` with `valid: true` on the deterministic sample set.
4. Frozen test-set `coco-metrics.json` from the held-out test annotations.

The final model selection report ranks these five frozen-test mAP50-95 values.
The earlier completed matched direct-stretch runs remain evidence for the
preprocessing decision, but are not a reason to schedule further direct-stretch
training.

Generate the scoped report with:

```bash
PYTHONPATH=src conda run --no-capture-output -n palm-nanodet \
  python scripts/summarize_single_palm_models.py \
  --experiment-root artifacts/single-palm-v001 \
  --scope configs/nanodet-plus-letterbox-scope-v001.json \
  --output artifacts/single-palm-v001/final-nanodet-plus-letterbox-v001.json
```

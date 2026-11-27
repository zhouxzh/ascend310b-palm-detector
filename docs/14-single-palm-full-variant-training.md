# Single-Palm Training Scope

The single-palm task has one class, `palm_surface`, for direct palm-region
detection. It is separate from the formal three-class `palm`/`back`/`side`
surface model.

The source labels are human-confirmed. `palm` and `back` boxes become
`palm_surface`; images containing a `side` box are excluded. This prevents the
single-class detector from learning that a visible side hand is background.

All active training uses seed 42, subject-disjoint splits, development-set
checkpoint selection, early stopping after eight validation checks without a
0.001 mAP improvement, static ONNX export, framework/ONNX agreement, and one
frozen-test evaluation. Centered zero-padded LetterBox is mandatory for every
training, validation, export, and inference path.

The active NanoDet-Plus scope is recorded in
[nanodet-plus-letterbox-scope-v001.json](../configs/nanodet-plus-letterbox-scope-v001.json):

| Variant | Input |
| --- | ---: |
| `plus-m-320` | 320x320 |
| `plus-m15-320` | 320x320 |
| `plus-m-416` | 416x416 |
| `plus-m15-416` | 416x416 |
| `plus-convnext-nano-640` | 640x640 |

Launch the resumable LetterBox-only controller with:

```bash
nohup ./scripts/run_single_palm_all_variants.sh \
  > artifacts/single-palm-v001/controller.log 2>&1 &
./scripts/start_single_palm_tensorboard.sh 6008
```

The controller records status in `artifacts/single-palm-v001/training-state.json`.
The scoped rank-ordered report is written to
`artifacts/single-palm-v001/final-nanodet-plus-letterbox-v001.json` and `.md`.

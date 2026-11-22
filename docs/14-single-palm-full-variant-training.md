# Single-Palm Full Variant Training

This experiment defines one class, `palm_surface`, for direct palm-region
detection. It is separate from the formal three-class palm-surface model.

The source labels are the human-confirmed `palm`/`back`/`side` annotations.
`palm` and `back` boxes become `palm_surface`; every image that contains a
`side` box is excluded. This ensures that the single-class detector is not
trained to treat a visible side hand as background.

The matrix contains 50 tasks, all using seed 42, fixed subject-disjoint splits,
development-set checkpoint selection, early stopping after eight validation
checks without a 0.001 mAP improvement, ONNX export, framework/ONNX agreement,
and one final frozen-test evaluation:

| Family | Variants | Geometry | Tasks |
| --- | ---: | --- | ---: |
| NanoDet | 16 upstream configurations | LetterBox and direct stretch | 32 |
| PicoDet-LCNet | XS/S/M/L at supported input sizes | LetterBox and direct stretch | 18 |

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
`final-comparison.md`.

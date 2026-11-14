# Training records

Every run directory is immutable and named like:

```text
nanodet-plus-m-320-portable-handed__data-v001__seed-42__YYYYMMDD-HHMM
```

`scripts/train_nanodet_portable_handed.sh` first writes `run.json` and a copied `config.yml`.
The manifest records the project revision and dirty state, fixed NanoDet commit,
base/config SHA-256 values, audited split hashes, pretrained weight URL/release/
license/bytes/hash, seed, OS, Python, and key package versions. It refuses to
start if a run ID already exists, data audit failed, or the checkpoint does not
match its manifest.

Use validation `mAP`/AP50 to select the best checkpoint. Do not select the last
epoch or lowest training loss. Preserve training log, data audit, exported
metrics, error samples, and the exact failure reason in the run directory.

The current baseline uses 150 epochs. Keep the normal experimental sequence
interpretable: alter a small, justified set of variables per run, select using
validation evidence, and retain an independent test set. After test results are
used for tuning, freeze a new independent test set.

`scripts/evaluate_nanodet.sh ... test` writes an evaluation-only config under
`evaluations/test/` that changes NanoDet's otherwise validation-only annotation
path to `instances_test.json`; it never changes the training config or
checkpoint selection.

COCO AP and per-class AP come from NanoDet's evaluator. Run
`scripts/evaluate_task_metrics.py` on its COCO prediction JSON and the matching
frozen annotations to separately report correct anatomical-handedness rate,
miss rate, and duplicate rate. It retains groups for device, lighting,
occlusion, and subject IDs whenever those image metadata fields are supplied.

The NanoDet data transform has no facility to exchange category IDs on
horizontal flip. Its flip probability is consequently hard-set to zero. Do not
enable flip until an augmentation implementation swaps category 1/2 and has
dedicated automated tests.

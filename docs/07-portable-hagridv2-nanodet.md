# Portable HaGRIDv2 NanoDet Training

This workflow includes the original single-class palm experiment and a
two-class anatomical left/right palm detector. Both locate palm regions rather
than the full hand from wrist through fingertips.

## Source

- Dataset: `zhouxzh/portable-hagridv2-mediapipe-hand`
- Mirror: `https://hf-mirror.com`
- Pinned revision: `e343025bd4fd37719dbc269201d6b24124f0c432`
- License: HaGRIDv2 public license with attribution and conditions reserved.
- Rows: 7,246 train, 845 validation, 1,663 test.
- Labels: MediaPipe legacy graph palm detection boxes. The dataset does not
  contain subject IDs or handedness labels.

Run the resumable verified download:

```bash
./scripts/download_portable_hagridv2.sh
```

It fetches from the mirror, locks the revision, and verifies the four Parquet
LFS SHA-256 values before accepting them. To use more concurrent HTTP ranges on
a faster network, set `HF_MIRROR_RANGE_WORKERS`; the default is `4`.

If the local proxy is slow for Xet CDN objects, set `HF_MIRROR_CDN_IP` only
after measuring a valid `us.aws.cdn.hf.co` address. The script still requests
the signed object URL from `hf-mirror.com`, preserves the CDN TLS hostname with
`--resolve`, and validates the expected LFS SHA-256 after assembly.

The mirror currently redirects LFS batch requests to the official Hub. If that
redirect resolves unreliably on the host, pin a reachable Hub address for that
request while retaining the mirror URL as the entry point:

```bash
HF_HUB_IP=18.65.14.85 ./scripts/download_portable_hagridv2.sh
```

## Preparation

For a fully self-contained copy, use the verified Parquet download above, then
convert it:

```bash
PYTHONPATH=src conda run -n palm-nanodet python scripts/prepare_portable_hagridv2_coco.py \
  --dataset-dir data/external/portable-hagridv2-mediapipe-hand \
  --output-dir data/portable-hagridv2-mediapipe-hand/coco-dedup-v001
```

On a host that already has the matching HaGRIDv2 512px images, the metadata-only
path avoids re-downloading the Parquet-embedded JPEG bytes. It still retrieves
the fixed-revision label columns from `hf-mirror.com`, verifies every local image
by source path and decoded dimensions, and records that image provenance:

```bash
PYTHONPATH=src conda run -n palm-nanodet python scripts/fetch_portable_hagridv2_metadata.py \
  --source-image-root /data2/hand_datasets/HaGRIDv2/HaGRIDv2_dataset_512 \
  --output-dir artifacts/portable-hagridv2-mirror-metadata-all \
  --allow-local-source-images

PYTHONPATH=src conda run -n palm-nanodet python scripts/prepare_portable_hagridv2_records_coco.py \
  --metadata-manifest artifacts/portable-hagridv2-mirror-metadata-all/metadata-manifest.json \
  --records-dir artifacts/portable-hagridv2-mirror-metadata-all/records \
  --source-image-root /data2/hand_datasets/HaGRIDv2/HaGRIDv2_dataset_512 \
  --output-dir data/portable-hagridv2-mediapipe-hand/coco-dedup-v001
```

The converted COCO contract requires image hashes and source-session uniqueness;
it explicitly reports that subject leakage cannot be checked because the public
dataset omits `user_id`. Content duplicates found across the upstream split are
dropped from the later split before the audit, so train/validation/test do not
share an image hash.

## Train and Export

```bash
./scripts/train_nanodet_portable_palm.sh \
  nanodet-plus-m-320-portable-hagridv2-palm__data-v001__seed-42__YYYYMMDD-HHMM \
  artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt \
  data/manifests/nanodet-plus-m_416_checkpoint.json

./scripts/export_onnx.sh artifacts/runs/<run-id>
```

The resulting static ONNX output is `1x2125x33`: one sigmoid `palm` score and
32 DFL values. This differs intentionally from the two-class left/right model's
`1x2125x34` output.

## Anatomical Left/Right Conversion

The portable release includes MediaPipe palm boxes and 21-point hand landmarks,
but does not include a handedness field. Generate a separate two-class COCO
dataset by matching MediaPipe Tasks handedness to each saved landmark set:

```bash
PYTHONPATH=src conda run -n palm-nanodet python scripts/prepare_portable_hagridv2_handed_coco.py \
  --metadata-manifest artifacts/portable-hagridv2-mirror-metadata-all/metadata-manifest.json \
  --records-dir artifacts/portable-hagridv2-mirror-metadata-all/records \
  --source-image-root /data2/hand_datasets/HaGRIDv2/HaGRIDv2_dataset_512 \
  --source-annotations-root /data2/hand_datasets/HaGRIDv2/annotations \
  --model-path /home/zhong/Documents/piano/upstream/hagridv2-512px-mediapipe-palm/models/hand_landmarker.task \
  --calibration-record data/manifests/mediapipe-tasks-calibration-v001.json \
  --output-dir data/portable-hagridv2-mediapipe-hand/coco-handed-v001
```

The new COCO categories are `1=left_palm` and `2=right_palm`. The conversion
recovers subject IDs from the original HaGRIDv2 annotation files, records the
model and source hashes, and places images without a matched handedness or a
source subject ID in `handedness-review.json`. The original single-class COCO
dataset and its training runs are separate from this conversion. MediaPipe Tasks
labels are recorded as their own annotation provenance; keypoint orientation is
not used to override them.

Train this two-class dataset with its own configuration and run ID:

```bash
./scripts/train_nanodet_portable_handed.sh \
  nanodet-plus-m-320-portable-handed__data-v001__seed-42__YYYYMMDD-HHMM \
  artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt \
  data/manifests/nanodet-plus-m_416_checkpoint.json
```

This run uses modest scale, stretch, translation, and color augmentation, with
horizontal flipping disabled for anatomical classes. AdamW starts with 500
linear warm-up steps and uses a 150-epoch cosine annealing schedule. Validation
runs every five epochs; NanoDet selects the best validation mAP checkpoint.
Training loss, learning rate, and validation metrics are written as TensorBoard
events inside the run directory. Start the dashboard with:

```bash
tensorboard --logdir artifacts/runs/<run-id> --host 127.0.0.1 --port 6006
```

## Hugging Face Release Package

Build an annotations-only package referencing the pinned source Parquet images:

```bash
PYTHONPATH=src python scripts/build_hf_dataset_release.py
```

The local build replaces source subject/session IDs with release-specific
pseudonyms. The upload folder contains COCO JSON and source-image references,
with no image bytes. Its README identifies the pinned source dataset and image
hash contract. `artifacts/hf-release/subject-key.bin` stays local
and must not be uploaded. `surface_annotations/` contains automatic view
candidates, not human-verified palmprint labels. The dataset card records
HaGRIDv2 attribution and its license link. Upload to a *new* dataset repository,
not the upstream source repository. The Hub's official endpoint authenticates
the token; the configured mirror is intended for downloads:

```bash
HF_ENDPOINT=https://huggingface.co hf upload-large-folder \
  --repo-type dataset --private <username/new-dataset-repo> \
  artifacts/hf-release/portable-handed-v001-annotations
```

# Hand Surface Annotation Candidates

The left/right NanoDet classes describe anatomical handedness. They do not
distinguish the visible palm surface from the back of the hand. The portable
HaGRIDv2 data has no source field for this distinction.

`scripts/annotate_hand_surface.py` creates a separate COCO release with one
surface attribute per existing hand annotation. It reruns the pinned MediaPipe
Tasks Hand Landmarker, matches its landmarks to each saved portable instance,
and estimates the 3D palm-plane normal. The normal is signed using the existing
anatomical left/right label. Strongly facing samples become `palm` or `back`;
intermediate angles become `unknown`. Apparent side views are suggested as
`side` but remain `unknown` until visually confirmed, because curled fingers
can also produce a near-edge-on normal.

```bash
PYTHONPATH=src conda run -n palm-nanodet python scripts/annotate_hand_surface.py \
  --source-dir data/portable-hagridv2-mediapipe-hand/coco-handed-v001 \
  --records-dir artifacts/portable-hagridv2-mirror-metadata-all/records \
  --model-path /home/zhong/Documents/piano/upstream/hagridv2-512px-mediapipe-palm/models/hand_landmarker.task \
  --output-dir data/portable-hagridv2-mediapipe-hand/coco-handed-surface-v001
```

The new release reuses the existing images through a directory link. Its COCO
annotations keep the original boxes and left/right category IDs unchanged, and
add these fields:

| Field | Meaning |
| --- | --- |
| `surface_view` | Provisional `palm`, `back`, or `unknown` |
| `surface_candidate_view` | Raw normal-based suggestion, including `side` |
| `surface_facing` | Signed unit-normal camera component, or `null` if unmatched |
| `surface_provenance` | Method that produced the suggestion |
| `surface_reviewed` | `false` until a visual review is recorded |
| `palmprint_usable` | `null`; visible surface alone does not establish palmprint quality |

`surface-review.json` lists all `unknown` annotations, including side-view
candidates. The output manifest records counts, thresholds, source annotation
hashes, and the exact MediaPipe model hash. These automatic suggestions are not
human ground truth. Review a subject-disjoint sample of each class, correct
errors, and evaluate the final surface classifier on independently reviewed
images before using it to admit palmprint captures.

The current `coco-handed-surface-v001` release contains 9,437 hand annotations:
7,748 provisional palm-facing, 1,316 provisional back-facing, and 373 unknown.
Among the unknown items, 49 have a side-view suggestion. All original COCO
hand boxes and anatomical left/right categories are unchanged, and the full
image/hash/subject/session audit passes at
`reports/portable-hagridv2-handed-surface-data-audit-v001.json`.

## Browser Spot Review

Start the local review desk from the repository root:

```bash
python scripts/surface_review_server.py --port 8765
```

Open `http://127.0.0.1:8765/`. Choose a split, candidate view, sample size,
and seed. Balanced sampling draws from each candidate group; random sampling
draws in proportion to the data. The seed and selected item are kept in the URL
so a sample can be revisited. Each decision records a reviewer ID, the chosen
surface view, an optional note, and the time. Later corrections append to the
same annotation's history. Reviews are stored separately at
`artifacts/reviews/hand-surface-v001.json` and can be exported as CSV. The
original COCO files remain unchanged. The server binds only to `127.0.0.1`.

The image panel can show up to two validated NanoDet ONNX exports at once.
The server discovers exports in `artifacts/runs/`, checks each ONNX hash and
static input/output shape against its `onnx-check.json`, and uses the repository's
BGR preprocessing and DFL/NMS decoder. Model A is drawn in teal and model B in
rose, while the source annotation remains yellow. The confidence slider applies
to both selected models. Single-class exports display `palm`; two-class exports
display anatomical left/right. Inference always uses the original full image;
the centered hand crop is only a browser inspection view and does not modify
the dataset or training coordinates.

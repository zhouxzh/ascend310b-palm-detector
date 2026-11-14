# Data contract

This repository accepts only COCO detection data with this immutable category
mapping:

| COCO ID | Model index | Name | Meaning |
| ---: | ---: | --- | --- |
| 1 | 0 | `left_palm` | The person's anatomical left hand |
| 2 | 1 | `right_palm` | The person's anatomical right hand |

The box is a pixel `xywh` rectangle around the palm region from the MediaPipe
palm detector. It does not intentionally include all fingertips or unrelated
forearm, torso, or held objects. This project does not infer left/right from
whether a hand appears on the image's left or right side.

## Required image metadata

Every COCO `images[]` item needs the standard `id`, `file_name`, `width`, and
`height`, plus:

```json
{
  "subject_id": "p0042",
  "session_id": "capture-20261110-a",
  "sha256": "lowercase-64-character-content-hash"
}
```

`subject_id` and `session_id` are non-identifying dataset-local IDs. The audit
requires each field and rejects intersections across train, validation, and
test. Split preparation joins images connected by either ID, so a subject or a
video/session cannot be split across sets. It also rejects duplicated content
hashes and, when requested, checks the hash against the actual image file.

## Annotation rules

- Include reliably identifiable left or right hands; use two instances for
  overlapping hands.
- For partly occluded hands, box only the visible region and use `iscrowd: 0`.
- Do not force a class for extreme blur, tiny ambiguous hands, or unknown
  handedness. Track them as an external review/ignore list instead.
- Do not include mirror reflections, displayed photos, posters, or screens
  unless a new versioned annotation policy explicitly adds them.
- COCO `area` must equal `bbox[2] * bbox[3]`; boxes must be positive and fully
  within the decoded image.

Before a baseline, conduct a documented human review of each class covering at
least 10% and at least 100 instances, and inspect at least 100 images after the
actual static 320 resize. Record swapped-handedness, missed labels, loose boxes,
and ambiguity counts in the data report.

MediaPipe or another model may generate a *weak-label candidate* only when its
model version, output score, source-box match IoU, calibration examples, and
rejections are preserved. It remains weak supervision until a human review
record confirms the sampled instances and left/right mapping. A model's
handedness output must never be silently rewritten as source ground truth.

For palm/back/side labels, use the browser review desk in
`docs/08-hand-surface-annotations.md`. It stores reviewer IDs, timestamps,
corrections, and notes separately from the original COCO data.

## Privacy and source provenance

Keep one source manifest per dataset release using
`data/manifests/source-manifest.template.json`. It records approval, license or
consent reference, retention policy, source ID, and annotation version without
committing personal data or images. Raw media, consent records, private URLs,
and unconfirmed rights stay outside Git.

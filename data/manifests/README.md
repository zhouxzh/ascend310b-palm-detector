# Manifest templates

Copy the templates in this directory to a non-template JSON file for every
actual source, annotation release, split, and pretrained checkpoint. Do not
commit source manifests that contain personal data, private URLs, consent
records, or secrets.

The COCO image records used by this project separately carry `subject_id`,
`session_id`, and image-content `sha256`. Those fields are mandatory because
the split/audit scripts use them to detect leakage and duplicate frames.

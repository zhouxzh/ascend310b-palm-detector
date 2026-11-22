# Retired Handedness Experiment

An earlier two-class `left_palm`/`right_palm` detector was retired after
human review found that its apparent accuracy was substantially explained by
image-space position: most left-hand examples appeared on the left side of an
image and most right-hand examples appeared on the right side. The detector
therefore learned a framing shortcut rather than reliable anatomical
handedness.

The labels, training configurations, models, ONNX exports, benchmark results,
and Hugging Face annotation release for this experiment have been removed from
the active project. They are not valid for deployment, model comparison, or
palmprint admission.

The formal task is now only `palm`/`back`/`side`. It uses human-confirmed
surface labels, and downstream palmprint recognition accepts only the `palm`
class. This document is intentionally retained as an audit record of the
rejected hypothesis and the reason for its removal.

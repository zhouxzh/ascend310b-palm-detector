"""Task metrics that keep left/right mistakes visible beside COCO AP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _iou_xywh(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_width, left_height = left
    right_x1, right_y1, right_width, right_height = right
    x1, y1 = max(left_x1, right_x1), max(left_y1, right_y1)
    x2, y2 = min(left_x1 + left_width, right_x1 + right_width), min(left_y1 + left_height, right_y1 + right_height)
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0 else 0.0


def _empty_counts() -> dict[str, int]:
    return {"ground_truth": 0, "predictions": 0, "matched": 0, "correct_handedness": 0, "missed": 0, "duplicate": 0, "wrong_handedness": 0}


def _rates(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "correct_handedness_rate": counts["correct_handedness"] / counts["matched"] if counts["matched"] else None,
        "miss_rate": counts["missed"] / counts["ground_truth"] if counts["ground_truth"] else None,
        "duplicate_rate": counts["duplicate"] / counts["ground_truth"] if counts["ground_truth"] else None,
    }


def task_metrics(
    ground_truth: dict[str, Any],
    predictions: Iterable[dict[str, Any]],
    *,
    score_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    group_fields: tuple[str, ...] = ("device", "lighting", "occlusion", "subject_id"),
) -> dict[str, Any]:
    """Greedily match by spatial IoU, then report whether class anatomy agrees."""
    image_info = {image["id"]: image for image in ground_truth["images"]}
    ground_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        if annotation.get("iscrowd", 0) == 0:
            ground_by_image[annotation["image_id"]].append(annotation)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction.get("score", 0.0) >= score_threshold:
            predictions_by_image[prediction["image_id"]].append(prediction)
    group_counts: dict[str, dict[str, dict[str, int]]] = {"all": {"all": _empty_counts()}}
    for field in group_fields:
        group_counts[field] = defaultdict(_empty_counts)

    def target_groups(image_id: int) -> list[dict[str, int]]:
        image = image_info[image_id]
        targets = [group_counts["all"]["all"]]
        for field in group_fields:
            value = str(image.get(field, "unknown"))
            targets.append(group_counts[field][value])
        return targets

    for image_id, image in image_info.items():
        targets = target_groups(image_id)
        ground = ground_by_image[image_id]
        predicted = sorted(predictions_by_image[image_id], key=lambda item: item["score"], reverse=True)
        for counts in targets:
            counts["ground_truth"] += len(ground)
            counts["predictions"] += len(predicted)
        unmatched = set(range(len(ground)))
        matched = set()
        for prediction in predicted:
            candidates = [(index, _iou_xywh(prediction["bbox"], ground[index]["bbox"])) for index in unmatched]
            candidates = [candidate for candidate in candidates if candidate[1] >= iou_threshold]
            if candidates:
                index, _ = max(candidates, key=lambda item: item[1])
                unmatched.remove(index)
                matched.add(index)
                correct = prediction["category_id"] == ground[index]["category_id"]
                for counts in targets:
                    counts["matched"] += 1
                    if correct:
                        counts["correct_handedness"] += 1
                    else:
                        counts["wrong_handedness"] += 1
            elif any(_iou_xywh(prediction["bbox"], ground[index]["bbox"]) >= iou_threshold for index in matched):
                for counts in targets:
                    counts["duplicate"] += 1
        for counts in targets:
            counts["missed"] += len(unmatched)
    groups: dict[str, dict[str, dict[str, object]]] = {}
    for field, values in group_counts.items():
        groups[field] = {value: {**counts, **_rates(counts)} for value, counts in values.items()}
    return {"score_threshold": score_threshold, "iou_threshold": iou_threshold, "groups": groups}

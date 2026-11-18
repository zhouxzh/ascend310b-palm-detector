"""The category mapping is an externally visible model contract."""

from __future__ import annotations

CLASS_NAMES = ("left_palm", "right_palm")
COCO_CATEGORY_IDS = {1: "left_palm", 2: "right_palm"}
CLASS_INDEX_BY_COCO_ID = {category_id: index for index, category_id in enumerate(COCO_CATEGORY_IDS)}
SINGLE_PALM_CATEGORY_IDS = {1: "palm"}
SURFACE_CATEGORY_IDS = {1: "palm", 2: "back", 3: "side"}


def validate_categories(
    categories: list[dict], expected_categories: dict[int, str] = COCO_CATEGORY_IDS
) -> list[str]:
    """Return contract violations without relying on category-list ordering."""
    observed = {category.get("id"): category.get("name") for category in categories}
    errors: list[str] = []
    if len(observed) != len(categories):
        errors.append("COCO categories contain duplicate ids")
    if observed != expected_categories:
        contract = ", ".join(
            f"{{id: {category_id}, name: {name}}}"
            for category_id, name in expected_categories.items()
        )
        errors.append(f"COCO categories must be exactly [{contract}]")
    return errors

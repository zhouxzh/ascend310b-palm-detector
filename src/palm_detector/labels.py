"""The category mapping is an externally visible model contract."""

from __future__ import annotations

SURFACE_CATEGORY_IDS = {1: "palm", 2: "back", 3: "side"}
CLASS_NAMES = tuple(SURFACE_CATEGORY_IDS.values())
COCO_CATEGORY_IDS = SURFACE_CATEGORY_IDS
CLASS_INDEX_BY_COCO_ID = {category_id: index for index, category_id in enumerate(COCO_CATEGORY_IDS)}


def validate_categories(
    categories: list[dict], expected_categories: dict[int, str] = SURFACE_CATEGORY_IDS
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

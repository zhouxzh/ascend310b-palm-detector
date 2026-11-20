from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_nanodet_surface3_resolution_ablation_changes_only_input_size() -> None:
    pairs = (
        ("nanodet-plus-m-416-surface3-letterbox-v003.yml", "nanodet-plus-m-320-surface3-letterbox-v004.yml"),
        ("nanodet-plus-m15-416-surface3-letterbox-v003.yml", "nanodet-plus-m15-320-surface3-letterbox-v004.yml"),
    )
    for source_name, candidate_name in pairs:
        source = yaml.safe_load((ROOT / "configs" / source_name).read_text(encoding="utf-8"))
        candidate = yaml.safe_load((ROOT / "configs" / candidate_name).read_text(encoding="utf-8"))
        assert candidate["class_names"] == source["class_names"] == ["palm", "back", "side"]
        assert candidate["data"]["train"]["input_size"] == candidate["data"]["val"]["input_size"] == [320, 320]
        assert candidate["data"]["train"]["pipeline"] == source["data"]["train"]["pipeline"]
        assert candidate["data"]["val"]["pipeline"] == source["data"]["val"]["pipeline"]
        assert candidate["schedule"] == source["schedule"]
        assert candidate["model"] == source["model"]


def test_picodet_surface3_resolution_ablation_reuses_416_training_recipes() -> None:
    candidate = yaml.safe_load((ROOT / "configs/picodet-surface3-320-letterbox-v004.yml").read_text(encoding="utf-8"))
    l416 = yaml.safe_load((ROOT / "configs/picodet-surface3-letterbox-v003.yml").read_text(encoding="utf-8"))
    m416 = yaml.safe_load((ROOT / "configs/picodet-surface3-m-416-letterbox-v003.yml").read_text(encoding="utf-8"))
    expected = {item["id"].replace("-320-", "-416-"): item for item in candidate["variants"]}
    assert candidate["classes"] == l416["classes"] == m416["classes"] == ["palm", "back", "side"]
    assert candidate["preprocessing"] == l416["preprocessing"] == m416["preprocessing"]
    assert candidate["sampling"] == l416["sampling"] == m416["sampling"]
    for source in (l416, m416):
        original = source["variants"][0]
        ablation = expected[original["id"]]
        assert ablation["input_size"] == 320
        assert ablation["batch_size"] == original["batch_size"]
        assert ablation["reference_lr"] == original["reference_lr"]
        assert ablation["epochs"] == original["epochs"]

#!/usr/bin/env python3
"""Materialize immutable NanoDet direct-stretch configs for the surface3 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "configs/surface3-320-direct-stretch-v005.yml"
DEFAULT_OUTPUT = ROOT / "artifacts/surface3-320-stretch-v005/configs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(source: dict, *, input_size: int) -> dict:
    config = json.loads(json.dumps(source))
    if config.get("class_names") != ["palm", "back", "side"]:
        raise ValueError("direct-stretch surface config requires palm/back/side classes")
    for split in ("train", "val"):
        dataset = config["data"][split]
        if dataset["input_size"] != [input_size, input_size]:
            raise ValueError(f"{split} input must already be {input_size} square")
        dataset["keep_ratio"] = False
        dataset["pipeline"]["palm_letterbox"] = False
    pipeline = config["data"]["train"]["pipeline"]
    if pipeline.get("stretch") != [[1.0, 1.0], [1.0, 1.0]]:
        raise ValueError("random anisotropic stretch must remain disabled in the primary ablation")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    recipe_bytes = args.recipe.read_bytes()
    recipe = yaml.safe_load(recipe_bytes)
    if recipe.get("geometry") != "direct_stretch" or recipe.get("random_anisotropic_stretch") is not False:
        raise ValueError("recipe must define a direct-stretch ablation without random anisotropic stretch")
    input_size = int(recipe["input_size"])
    args.output.mkdir(parents=True, exist_ok=True)
    configs = []
    for item in recipe["nanodet"]:
        source_path = ROOT / item["source_config"]
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        generated = build_config(source, input_size=input_size)
        path = args.output / f"{item['id']}.yml"
        path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
        configs.append({
            "id": item["id"], "source_config": str(source_path), "source_sha256": sha256(source_path),
            "config": str(path), "config_sha256": sha256(path), "pretrained_checkpoint": item["pretrained_checkpoint"],
            "pretrained_manifest": item["pretrained_manifest"],
        })
    manifest = {
        "schema_version": 1, "recipe": str(args.recipe), "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "geometry": recipe["geometry"], "input_size": input_size, "random_anisotropic_stretch": False, "configs": configs,
    }
    report = args.output.parent / "nanodet-config-inventory.json"
    report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

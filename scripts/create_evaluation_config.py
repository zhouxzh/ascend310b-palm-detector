#!/usr/bin/env python3
"""Derive an evaluation-only config without mutating an immutable training run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--task", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.run_config.read_text(encoding="utf-8"))
    if args.task == "test":
        val_path = str(config["data"]["val"]["ann_path"])
        if "instances_val.json" not in val_path:
            raise ValueError(f"cannot derive test annotations from val path: {val_path}")
        config["data"]["val"]["ann_path"] = val_path.replace("instances_val.json", "instances_test.json")
    config["save_dir"] = str(args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

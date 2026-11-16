#!/usr/bin/env python3
"""Convert a pinned TIMM classification checkpoint to NanoDet backbone init."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/nanodet"))
from nanodet.model.arch import build_model
from nanodet.util.yacs import CfgNode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if _sha256(args.source) != args.expected_source_sha256:
        raise ValueError("TIMM source checkpoint SHA-256 differs from the pinned filename hash")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_model(CfgNode(config["model"]))
    target = model.state_dict()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = payload.get("state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint does not contain a state dictionary")
    state = {}
    for key, value in source_state.items():
        if key.startswith(("head.", "classifier.", "norm.")):
            continue
        converted_key = key.replace("stem.", "stem_", 1).replace("stages.", "stages_", 1)
        converted = "backbone.timm." + converted_key
        if converted not in target or target[converted].shape != value.shape:
            continue
        state[converted] = value
    backbone_keys = [key for key in target if key.startswith("backbone.")]
    coverage = len(state) / len(backbone_keys)
    if coverage < 0.98:
        raise ValueError(f"TIMM weights only cover {coverage:.1%} of the NanoDet backbone")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b"")
    torch.save(
        {
            "epoch": 0,
            "iter": 0,
            "state_dict": state,
            "backbone_only": True,
        },
        args.output,
    )
    record = {
        "model_name": "convnext-nano-d1h-in1k-backbone-init",
        "upstream_url": args.source_url,
        "upstream_revision_or_release": "pytorch-image-models release v0.1-rsb-weights; convnext_nano.d1h_in1k",
        "license": "Apache-2.0",
        "local_path": str(args.output.resolve().relative_to(ROOT)),
        "bytes": args.output.stat().st_size,
        "sha256": _sha256(args.output),
        "source_checkpoint_sha256": _sha256(args.source),
        "backbone_tensor_coverage": coverage,
        "loaded_tensors": len(state),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"converted {len(state)}/{len(backbone_keys)} backbone tensors ({coverage:.1%}): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

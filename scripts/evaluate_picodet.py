#!/usr/bin/env python3
"""Evaluate a completed PicoDet run on an explicitly selected COCO split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/PaddleDetection")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yml"
    checkpoint = run_dir / "paddle-output/best_model"
    if not config_path.is_file() or not checkpoint.with_suffix(".pdparams").is_file() or not args.annotations.is_file():
        raise FileNotFoundError("PicoDet config, best checkpoint, or test annotations are missing")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    classes = tuple(metadata.get("classes", ()))
    if classes not in {("left_palm", "right_palm"), ("palm", "back", "side")}:
        raise ValueError("PicoDet run has an unsupported class contract")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(args.upstream.resolve()))
    from palm_detector.paddledetection_extensions import install_paddledetection_extensions

    install_paddledetection_extensions()
    import paddle
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer

    cfg = load_config(str(config_path))
    cfg.EvalDataset["anno_path"] = str(args.annotations.resolve())
    cfg.output_eval = str(args.output_dir.resolve())
    cfg.use_gpu = True
    cfg.use_vdl = False
    paddle.set_device("gpu:0")
    trainer = Trainer(cfg, mode="eval")
    trainer.load_weights(str(checkpoint))
    trainer.evaluate()
    prediction = args.output_dir / "bbox.json"
    if not prediction.is_file():
        raise FileNotFoundError("PicoDet evaluation did not emit bbox.json")
    print(prediction)
    # Paddle 3.5 may spin indefinitely during interpreter teardown after a
    # completed GPU evaluation. All artifacts are durable at this point; end
    # this one-shot subprocess without entering that teardown path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

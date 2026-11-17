#!/usr/bin/env python3
"""Run one GPU train step, checkpoint save, decode, and COCO evaluation.

This is a pipeline check, not a quality measurement. Its output records the
source label quality supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

from palm_detector.dataset_contract import sha256_file
from palm_detector.nanodet_extensions import install_nanodet_extensions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-quality", choices=("human_reviewed", "mediapipe_reviewed"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def _move_images(batch: dict) -> dict:
    batch["img"] = torch.stack(batch["img"], dim=0).cuda(non_blocking=True)
    return batch


def _git(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verified_pretrained(manifest_path: Path, checkpoint_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != sha256_file(checkpoint_path):
        raise ValueError("pretrained manifest SHA-256 does not match checkpoint")
    if int(manifest.get("bytes", -1)) != checkpoint_path.stat().st_size:
        raise ValueError("pretrained manifest byte count does not match checkpoint")
    return {**manifest, "manifest_path": str(manifest_path), "verified_sha256": manifest["sha256"]}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU pipeline smoke test")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory exists and is non-empty: {args.output_dir}")
    for path in (args.base_config, args.train_annotations, args.val_annotations, args.images_root, args.pretrained_checkpoint, args.pretrained_manifest, args.data_audit):
        if not path.exists():
            raise FileNotFoundError(path)
    data_audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    if not data_audit.get("valid"):
        raise ValueError("pipeline smoke requires a passing data-contract audit")
    pretrained = _verified_pretrained(args.pretrained_manifest, args.pretrained_checkpoint)

    install_nanodet_extensions()
    load_config(cfg, str(args.base_config))
    cfg.defrost()
    cfg.save_dir = str(args.output_dir)
    cfg.data.train.img_path = str(args.images_root)
    cfg.data.train.ann_path = str(args.train_annotations)
    cfg.data.val.img_path = str(args.images_root)
    cfg.data.val.ann_path = str(args.val_annotations)
    cfg.device.gpu_ids = [0]
    cfg.device.workers_per_gpu = 0
    cfg.device.batchsize_per_gpu = args.batch_size
    cfg.freeze()
    args.output_dir.mkdir(parents=True)
    config_path = args.output_dir / "config.yml"
    config_path.write_text(cfg.dump(), encoding="utf-8")
    run = {
        "run_id": args.output_dir.name,
        "purpose": "pipeline_smoke_not_quality_measurement",
        "label_quality": args.label_quality,
        "project": {"git_commit": _git("git", "rev-parse", "HEAD")},
        "upstream_nanodet": {"url": "https://github.com/RangiLyu/nanodet.git", "commit": _git("git", "-C", "third_party/nanodet", "rev-parse", "HEAD")},
        "base_config": {"path": str(args.base_config), "sha256": sha256_file(args.base_config)},
        "run_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "data_audit": {"path": str(args.data_audit), "sha256": sha256_file(args.data_audit)},
        "pretrained": pretrained,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "val")
    train_batch = _move_images(next(iter(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=naive_collate))))
    val_batch = _move_images(next(iter(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=naive_collate))))
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    load_model_weight(model, checkpoint, Logger(-1, str(args.output_dir), False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    _, loss, loss_states = model.forward_train(train_batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    model.eval()
    with torch.no_grad():
        detections = model.head.post_process(model(val_batch["img"]), val_batch)
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    evaluation = evaluator.evaluate(detections, str(args.output_dir), rank=-1)
    checkpoint_path = args.output_dir / "model_best" / "model_best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"epoch": 0, "state_dict": model.state_dict(), "label_quality": args.label_quality}, checkpoint_path)
    report = {
        "purpose": "pipeline_smoke_not_quality_measurement",
        "label_quality": args.label_quality,
        "cuda_device": torch.cuda.get_device_name(0),
        "train_annotations": {"path": str(args.train_annotations), "sha256": sha256_file(args.train_annotations)},
        "val_annotations": {"path": str(args.val_annotations), "sha256": sha256_file(args.val_annotations)},
        "pretrained_checkpoint": {"path": str(args.pretrained_checkpoint), "sha256": sha256_file(args.pretrained_checkpoint)},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "train_batch_shape": list(train_batch["img"].shape),
        "train_loss": float(loss.detach().cpu()),
        "loss_states": {name: float(value.detach().cpu()) for name, value in loss_states.items()},
        "decoded_images": len(detections),
        "decoded_detections": sum(len(boxes) for result in detections.values() for boxes in result.values()),
        "coco_evaluation": evaluation,
    }
    (args.output_dir / "smoke-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

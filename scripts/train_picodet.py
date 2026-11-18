#!/usr/bin/env python3
"""Train one pinned PicoDet experiment with reproducible early stopping."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class EarlyStop(RuntimeError):
    """Internal control flow raised after the upstream callback saves a checkpoint."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incoming")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def newest_resume_prefix(save_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for optimizer_path in save_dir.glob("*.pdopt"):
        try:
            epoch = int(optimizer_path.stem)
        except ValueError:
            continue
        prefix = optimizer_path.with_suffix("")
        if prefix.with_suffix(".pdparams").is_file():
            candidates.append((epoch, prefix))
    return max(candidates, default=(0, None))[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    classes = tuple(manifest["classes"])
    if classes not in {("left_palm", "right_palm"), ("palm", "back", "side")}:
        raise ValueError("PicoDet class contract must be anatomical hands or palm/back/side")
    config_path = Path(manifest["training_config_path"])
    if not config_path.is_file() or not (args.upstream / "tools/train.py").is_file():
        raise FileNotFoundError("generated config or pinned PaddleDetection checkout is missing")
    weights_path = ROOT / "artifacts/picodet-v001/pretrained/weights.json"
    if not weights_path.is_file():
        raise FileNotFoundError("run scripts/acquire_picodet_pretrained.py before training")
    weight_record = json.loads(weights_path.read_text(encoding="utf-8"))["weights"].get(manifest["pretrained"]["id"])
    if weight_record is None or not Path(weight_record["path"]).is_file():
        raise FileNotFoundError("verified pretraining weight is missing")
    if sha256(Path(weight_record["path"])) != weight_record["sha256"] or Path(weight_record["path"]).stat().st_size != weight_record["bytes"]:
        raise ValueError("verified pretraining weight checksum or byte count changed")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(args.upstream))
    from palm_detector.paddledetection_extensions import install_paddledetection_extensions

    install_paddledetection_extensions()
    import paddle
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer
    from ppdet.engine.callbacks import Callback

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
        raise RuntimeError("PicoDet sweep requires exactly one usable CUDA device")
    paddle.set_device("gpu:0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_dir = args.run_dir / "paddle-output"
    tensorboard_dir = args.run_dir / "tensorboard"
    summary_path = args.run_dir / "training-summary.json"
    progress_path = args.run_dir / "training-progress.json"
    copied_config = args.run_dir / "config.yml"
    if not copied_config.exists():
        shutil.copy2(config_path, copied_config)

    cfg = load_config(str(config_path))
    cfg.save_dir = str(save_dir)
    cfg.output_eval = str(args.run_dir / "dev-eval")
    cfg.vdl_log_dir = str(args.run_dir / "visualdl")
    cfg.use_vdl = False
    cfg.use_gpu = True
    cfg.log_iter = 50
    cfg.seed = args.seed
    if cfg.num_classes != len(classes):
        raise ValueError("generated PicoDet config class count does not match its manifest")
    transforms = cfg["TrainReader"]["sample_transforms"]
    if classes == ("left_palm", "right_palm") and any(item.get("RandomFlip", {}).get("prob") not in (None, 0.0) for item in transforms):
        raise ValueError("horizontal flip must remain disabled for anatomical classes")
    letterbox_mode = any("PalmLetterBox" in item for item in transforms)
    if letterbox_mode:
        if classes == ("left_palm", "right_palm") and not any(item.get("PalmHandednessFlip", {}).get("prob", 0.0) > 0.0 for item in transforms):
            raise ValueError("MediaPipe-style PicoDet training requires PalmHandednessFlip")
        if classes == ("palm", "back", "side") and not any(item.get("RandomFlip", {}).get("prob", 0.0) > 0.0 for item in transforms):
            raise ValueError("surface PicoDet training requires label-preserving horizontal flip")
        if not any("PalmLetterBox" in item for item in transforms):
            raise ValueError("MediaPipe-style PicoDet training requires PalmLetterBox")

    progress = {"schema_version": 1, "variant": manifest["id"], "best_dev_map": float("-inf"), "bad_checks": 0, "history": []}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    try:
        from tensorboardX import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("tensorboardX is required for PicoDet TensorBoard logs") from exc
    writer = SummaryWriter(str(tensorboard_dir))

    class SweepCallback(Callback):
        def __init__(self, model) -> None:
            super().__init__(model)
            self.best = float(progress["best_dev_map"])
            self.best_observed = float(progress.get("best_observed_dev_map", self.best))
            self.bad_checks = int(progress["bad_checks"])

        def on_step_end(self, status) -> None:
            if status.get("mode") != "train":
                return
            step = int(status["epoch_id"]) * int(status["steps_per_epoch"]) + int(status["step_id"])
            for key, value in status["training_staus"].get().items():
                writer.add_scalar(f"train/{key}", float(value), step)
            writer.add_scalar("train/lr", float(status["learning_rate"]), step)

        def on_epoch_end(self, status) -> None:
            if status.get("mode") != "eval":
                return
            results = self.model._metrics[0].get_results()
            if "bbox" not in results:
                raise RuntimeError("COCO bbox mAP is missing from validation results")
            map_value = float(results["bbox"][0])
            epoch = int(status["epoch_id"]) + 1
            writer.add_scalar("dev/mAP", map_value, epoch)
            if len(results["bbox"]) > 1:
                writer.add_scalar("dev/AP50", float(results["bbox"][1]), epoch)
            if len(results["bbox"]) > 2:
                writer.add_scalar("dev/AP75", float(results["bbox"][2]), epoch)
            improved = map_value > self.best + float(manifest["early_stop"]["min_delta"])
            self.best_observed = max(self.best_observed, map_value)
            if improved:
                self.best = map_value
                self.bad_checks = 0
            else:
                self.bad_checks += 1
            progress.update({
                "best_dev_map": self.best,
                "best_observed_dev_map": self.best_observed,
                "bad_checks": self.bad_checks,
                "history": [*progress["history"], {"epoch": epoch, "mAP": map_value, "improved": improved}],
            })
            atomic_json(progress_path, progress)
            if self.bad_checks >= int(manifest["early_stop"]["patience"]):
                raise EarlyStop("PicoDet early stopping criterion reached")

    run_metadata = {
        "run_id": args.run_dir.name,
        "variant": manifest["id"],
        "classes": list(classes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "config": str(copied_config),
        "config_sha256": sha256(copied_config),
        "upstream": str(args.upstream),
        "upstream_ref": "b25522a0f4bde8c80603f3ba5e3472059972e3b5",
        "project_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pretrained": weight_record,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "paddlepaddle_gpu": importlib.metadata.version("paddlepaddle-gpu"),
            "cuda_device": paddle.device.cuda.get_device_name(0),
        },
    }
    atomic_json(args.run_dir / "run.json", run_metadata)

    trainer = Trainer(cfg, mode="train")
    resume_prefix = newest_resume_prefix(save_dir) if args.resume else None
    if resume_prefix is not None:
        trainer.resume_weights(str(resume_prefix))
    elif cfg.get("pretrain_weights"):
        trainer.load_weights(cfg.pretrain_weights)
    callback = SweepCallback(trainer)
    trainer.register_callbacks([callback])
    status = "max_epochs"
    try:
        trainer.train(validate=True)
    except EarlyStop:
        status = "early_stopped"
    finally:
        writer.close()
    atomic_json(summary_path, {
        "schema_version": 1,
        "status": status,
        "best_dev_map": callback.best,
        "best_observed_dev_map": callback.best_observed,
        "early_stop": {"bad_checks": callback.bad_checks, **manifest["early_stop"]},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

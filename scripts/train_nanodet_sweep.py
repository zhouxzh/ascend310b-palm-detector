#!/usr/bin/env python3
"""Train one pinned NanoDet sweep config with a fixed effective batch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, TQDMProgressBar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/nanodet"))

from palm_detector.nanodet_extensions import install_nanodet_extensions
from palm_detector.labels import SURFACE_CATEGORY_IDS

install_nanodet_extensions()

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import NanoDetLightningLogger, cfg, convert_old_model, load_config, load_model_weight


METRIC_PATTERN = re.compile(r"Val_metrics: .*?'mAP': np\.float64\(([^)]+)\)")


def _best_completed_metric(save_dir: Path) -> float:
    result_path = save_dir / "model_best" / "eval_results.txt"
    if not result_path.is_file():
        return float("-inf")
    values = []
    for line in result_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("mAP:"):
            values.append(float(line.split(":", 1)[1].strip()))
    return max(values, default=float("-inf"))


def _historical_bad_checks(train_log: Path, best_metric: float, min_delta: float) -> int:
    if not train_log.is_file() or best_metric == float("-inf"):
        return 0
    values = [float(match.group(1)) for match in METRIC_PATTERN.finditer(train_log.read_text(encoding="utf-8", errors="replace"))]
    bad_checks = 0
    for value in reversed(values):
        if value > best_metric + min_delta:
            break
        bad_checks += 1
    return bad_checks


class SweepEarlyStopping(Callback):
    """Stop a sweep trial after validation mAP remains below its best result."""

    def __init__(self, *, initial_best: float, initial_bad_checks: int, patience: int, min_delta: float) -> None:
        self.best_metric = initial_best
        self.bad_checks = initial_bad_checks
        self.patience = patience
        self.min_delta = min_delta
        self.stopped = False

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metric = float(pl_module.save_flag)
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.bad_checks = 0
            return
        self.bad_checks += 1
        if self.bad_checks >= self.patience:
            self.stopped = True
            trainer.should_stop = True
            pl_module.logger.info(
                "Early stopping sweep trial after "
                f"{self.bad_checks} validation checks without a {self.min_delta:g} mAP improvement."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    args = parser.parse_args()
    if args.early_stop_patience < 1 or args.early_stop_min_delta < 0:
        raise ValueError("early-stop patience must be positive and min-delta must be nonnegative")
    load_config(cfg, str(args.config))
    classes = tuple(cfg.class_names)
    surface_classes = tuple(SURFACE_CATEGORY_IDS.values())
    if cfg.model.arch.head.num_classes != len(surface_classes) or classes != surface_classes:
        raise ValueError("the detector class contract must be palm/back/side")
    if cfg.data.train.pipeline.get("palm_letterbox", False) and not cfg.data.val.pipeline.get("palm_letterbox", False):
        raise ValueError("NanoDet LetterBox must be used for both train and validation")
    if cfg.device.gpu_ids != [0] or not torch.cuda.is_available():
        raise RuntimeError("the pinned sweep requires one usable CUDA GPU")
    accumulation = int(cfg.device.accumulate_grad_batches)
    if accumulation < 1:
        raise ValueError("accumulate_grad_batches must be positive")
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)
    train_data = build_dataset(cfg.data.train, "train")
    val_data = build_dataset(cfg.data.val, "test")
    evaluator = build_evaluator(cfg.evaluator, val_data)
    workers = int(cfg.device.workers_per_gpu)
    dataloader = dict(
        batch_size=cfg.device.batchsize_per_gpu,
        num_workers=workers,
        pin_memory=True,
        collate_fn=naive_collate,
    )
    if workers > 0:
        dataloader.update(persistent_workers=True, prefetch_factor=2)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, drop_last=True, **dataloader)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, drop_last=False, **dataloader)
    task = TrainingTask(cfg, evaluator)
    save_dir = Path(cfg.save_dir)
    previous_best = _best_completed_metric(save_dir)
    if previous_best != float("-inf"):
        task.save_flag = previous_best
    if "load_model" not in cfg.schedule:
        raise ValueError("a verified pretrained checkpoint is required")
    checkpoint = torch.load(cfg.schedule.load_model, map_location="cpu", weights_only=False)
    backbone_only = bool(checkpoint.get("backbone_only", False))
    if "pytorch-lightning_version" not in checkpoint:
        checkpoint = convert_old_model(checkpoint)
        checkpoint["backbone_only"] = backbone_only
    state = checkpoint["state_dict"].copy()
    for key in list(state):
        if key.startswith("avg_model."):
            state[key[4:]] = state.pop(key)
    if state and next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    if state and next(iter(state)).startswith("model."):
        state = {key[6:]: value for key, value in state.items()}
    target = task.model.state_dict()
    coverage: dict[str, list[int]] = {}
    for key, value in target.items():
        group = key.split(".", 1)[0]
        counts = coverage.setdefault(group, [0, 0])
        counts[1] += 1
        if key in state and tuple(value.shape) == tuple(state[key].shape):
            counts[0] += 1
    critical_groups = ("backbone",) if checkpoint.get("backbone_only", False) else ("backbone", "fpn")
    critical = [group for group in critical_groups if group in coverage]
    if not critical or any(coverage[group][0] / coverage[group][1] < 0.98 for group in critical):
        raise ValueError(f"pretrained checkpoint does not cover the critical backbone/FPN tensors: {coverage}")
    logger.info("Pretrained tensor coverage: " + json.dumps({key: round(value[0] / value[1], 4) for key, value in coverage.items()}, sort_keys=True))
    load_model_weight(task.model, checkpoint, logger)
    logger.info(f"Loaded verified pretrained checkpoint: {cfg.schedule.load_model}")
    resume_path = Path(cfg.save_dir) / "model_last.ckpt"
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"cannot resume without {resume_path}")
    resume = str(resume_path) if args.resume else None
    early_stopping = SweepEarlyStopping(
        initial_best=task.save_flag,
        initial_bad_checks=_historical_bad_checks(save_dir / "train.log", task.save_flag, args.early_stop_min_delta),
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator="gpu", devices=cfg.device.gpu_ids,
        accumulate_grad_batches=accumulation,
        precision=cfg.device.precision,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0), early_stopping],
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
    )
    trainer.fit(task, train_loader, val_loader, ckpt_path=resume)
    summary = {
        "schema_version": 1,
        "status": "early_stopped" if early_stopping.stopped else "max_epochs",
        "best_dev_map": None if task.save_flag == float("-inf") else task.save_flag,
        "early_stop": {
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "bad_checks": early_stopping.bad_checks,
        },
    }
    (save_dir / "training-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

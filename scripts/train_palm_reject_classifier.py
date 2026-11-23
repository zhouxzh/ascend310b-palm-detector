#!/usr/bin/env python3
"""Train a palm/reject ROI classifier from the pinned local Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
from datasets import Dataset, load_from_disk
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party" / "nanodet")]

from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2  # noqa: E402
from palm_detector.letterbox import letterbox_bgr  # noqa: E402


LABELS = ("palm", "reject")
IMAGE_SIZE = 416
NANODET_NORMALIZE = {"mean": [103.53, 116.28, 123.675], "std": [57.375, 57.12, 58.395]}
BACKBONES = ("nanodet_shufflenetv2_x1", "lcnet_100", "efficientnet_b0")
PRETRAINED_FILES = {
    "lcnet_100": ROOT / "artifacts/pretrained/lcnet_100-a929038c.pth",
    "efficientnet_b0": ROOT / "artifacts/pretrained/efficientnet_b0_ra-3dd342df.pth",
}


class NanoDetTransform:
    """Apply the detector's BGR LetterBox and normalization contract to an ROI."""

    def __init__(self, train: bool) -> None:
        self.augment = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            v2.RandomHorizontalFlip(),
        ]) if train else None

    def __call__(self, image: Image.Image) -> Tensor:
        if self.augment is not None:
            image = self.augment(image)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        canvas, _ = letterbox_bgr(bgr, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        normalized = (canvas.astype(np.float32) - np.asarray(NANODET_NORMALIZE["mean"], dtype=np.float32)) / np.asarray(NANODET_NORMALIZE["std"], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


class NanoDetShuffleNetClassifier(nn.Module):
    """Binary head over the exact ShuffleNetV2 backbone used by NanoDet-Plus-m."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ShuffleNetV2(
            model_size="1.0x",
            out_stages=(4,),
            with_last_conv=False,
            activation="LeakyReLU",
            pretrain=False,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(464, len(LABELS))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.backbone(inputs)[0]
        return self.classifier(self.pool(features).flatten(1))


def load_timm_pretrained(model: nn.Module, checkpoint_path: Path, *, model_name: str) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{model_name} pretrained checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    backbone_state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    missing_backbone = [key for key in missing if not key.startswith("classifier.")]
    if unexpected or missing_backbone:
        raise ValueError(f"{model_name} checkpoint mismatch: missing={missing_backbone}, unexpected={unexpected}")


def build_model(backbone: str, backbone_checkpoint: Path) -> tuple[nn.Module, str, str]:
    if backbone == "nanodet_shufflenetv2_x1":
        model = NanoDetShuffleNetClassifier()
        initialize_nanodet_backbone(model, backbone_checkpoint.resolve())
        return model, "nanodet_plus_m_shufflenetv2_1.0x_leakyrelu", str(backbone_checkpoint.resolve())
    if backbone == "lcnet_100":
        model = timm.create_model("lcnet_100", pretrained=False, num_classes=len(LABELS))
    elif backbone == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(LABELS))
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    checkpoint = PRETRAINED_FILES[backbone]
    load_timm_pretrained(model, checkpoint, model_name=backbone)
    return model, backbone, str(checkpoint)


class RoiDataset(TorchDataset[tuple[Tensor, int]]):
    def __init__(self, table: Dataset, transform: nn.Module) -> None:
        self.table = table
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.table[index]
        label = int(row["label_id"])
        if label not in (0, 1) or row["label"] != LABELS[label]:
            raise ValueError(f"invalid label at index {index}: {row['label']!r}/{label}")
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected a decoded PIL image, got {type(image)}")
        return self.transform(image.convert("RGB")), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/palm-reject-roi-v005")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/palm-reject-nanodet-shufflenetv2-x1-v001")
    parser.add_argument("--backbone", choices=BACKBONES, default="nanodet_shufflenetv2_x1")
    parser.add_argument("--backbone-checkpoint", type=Path, default=ROOT / "artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt")
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(dataset: TorchDataset[tuple[Tensor, int]], *, batch_size: int, workers: int, shuffle: bool) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def probabilities(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, target in loader:
            logits = model(inputs.to(device, non_blocking=True))
            scores.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (scores < threshold).astype(np.int64)  # 0=palm, 1=reject
    palm_true = labels == 0
    palm_pred = prediction == 0
    reject_true = ~palm_true
    reject_pred = ~palm_pred
    tp_palm = int(np.sum(palm_true & palm_pred))
    fp_palm = int(np.sum(~palm_true & palm_pred))
    fn_palm = int(np.sum(palm_true & ~palm_pred))
    tp_reject = int(np.sum(reject_true & reject_pred))
    fp_reject = int(np.sum(~reject_true & reject_pred))
    fn_reject = int(np.sum(reject_true & ~reject_pred))

    def f1(tp: int, fp: int, fn: int) -> float:
        denominator = 2 * tp + fp + fn
        return 0.0 if denominator == 0 else 2 * tp / denominator

    palm_f1 = f1(tp_palm, fp_palm, fn_palm)
    reject_f1 = f1(tp_reject, fp_reject, fn_reject)
    return {
        "threshold": threshold,
        "accuracy": float(np.mean(prediction == labels)),
        "palm_precision": 0.0 if tp_palm + fp_palm == 0 else tp_palm / (tp_palm + fp_palm),
        "palm_recall": 0.0 if tp_palm + fn_palm == 0 else tp_palm / (tp_palm + fn_palm),
        "palm_f1": palm_f1,
        "reject_precision": 0.0 if tp_reject + fp_reject == 0 else tp_reject / (tp_reject + fp_reject),
        "reject_recall": 0.0 if tp_reject + fn_reject == 0 else tp_reject / (tp_reject + fn_reject),
        "reject_f1": reject_f1,
        "macro_f1": (palm_f1 + reject_f1) / 2,
        "palm_true_positive": tp_palm,
        "palm_false_positive": fp_palm,
        "palm_false_negative": fn_palm,
        "reject_true_positive": tp_reject,
        "reject_false_positive": fp_reject,
        "reject_false_negative": fn_reject,
    }


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    candidates = np.linspace(0.05, 0.95, 181)
    return max(
        (metrics_at_threshold(scores, labels, float(value)) for value in candidates),
        key=lambda item: (item["macro_f1"], item["threshold"]),
    )


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_nanodet_backbone(model: NanoDetShuffleNetClassifier, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NanoDet checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    backbone_state = {
        key.removeprefix("model.backbone."): value
        for key, value in state.items()
        if key.startswith("model.backbone.")
    }
    if not backbone_state:
        raise ValueError("checkpoint does not contain NanoDet backbone parameters")
    model.backbone.load_state_dict(backbone_state, strict=True)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0 or args.patience <= 0:
        raise ValueError("epochs, batch size, and patience must be positive; workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entry point")
    dataset_path = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    provenance = json.loads((dataset_path / "provenance.json").read_text(encoding="utf-8"))
    if provenance["class_labels"] != {"0": "palm", "1": "reject"}:
        raise ValueError("dataset provenance does not match the expected palm/reject label mapping")
    dataset = load_from_disk(str(dataset_path))
    if set(dataset) != {"train", "valid", "test"}:
        raise ValueError(f"unexpected dataset splits: {sorted(dataset)}")
    set_seed(args.seed)
    output.mkdir(parents=True)
    device = torch.device("cuda")
    train_transform = NanoDetTransform(train=True)
    eval_transform = NanoDetTransform(train=False)
    train_loader = make_loader(RoiDataset(dataset["train"], train_transform), batch_size=args.batch_size, workers=args.workers, shuffle=True)
    valid_loader = make_loader(RoiDataset(dataset["valid"], eval_transform), batch_size=args.batch_size, workers=args.workers, shuffle=False)
    test_loader = make_loader(RoiDataset(dataset["test"], eval_transform), batch_size=args.batch_size, workers=args.workers, shuffle=False)
    model, model_name, initialization = build_model(args.backbone, args.backbone_checkpoint)
    model.to(device)
    tensorboard_dir = (args.tensorboard_dir or output / "tensorboard").resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for inputs, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(inputs.to(device, non_blocking=True))
                loss = criterion(logits, target.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            total_rows += len(target)
        scheduler.step()
        valid_scores, valid_labels = probabilities(model, valid_loader, device)
        valid_metrics = select_threshold(valid_scores, valid_labels)
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(epoch_record)
        writer.add_scalar("train/loss", epoch_record["train_loss"], epoch)
        writer.add_scalar("train/learning_rate", epoch_record["learning_rate"], epoch)
        for key, value in valid_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"valid/{key}", value, epoch)
        writer.flush()
        improved = float(valid_metrics["macro_f1"]) > best_score
        if improved:
            best_score = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"epoch": epoch, "model_name": model_name, "backbone": args.backbone, "model_state": model.state_dict(), "valid_metrics": valid_metrics}, output / "best.pt")
        else:
            stale_epochs += 1
        torch.save({"epoch": epoch, "model_name": model_name, "backbone": args.backbone, "model_state": model.state_dict(), "valid_metrics": valid_metrics, "history": history}, output / "last.pt.incoming")
        (output / "last.pt.incoming").replace(output / "last.pt")
        atomic_json(output / "status.json", {
            "state": "running",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "backbone": args.backbone,
            "model": model_name,
            "epoch": epoch,
            "epochs": args.epochs,
            "improved": improved,
            "best_epoch": best_epoch,
            "best_macro_f1": best_score,
            "stale_epochs": stale_epochs,
            "valid": valid_metrics,
        })
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if stale_epochs >= args.patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    valid_scores, valid_labels = probabilities(model, valid_loader, device)
    valid_metrics = select_threshold(valid_scores, valid_labels)
    test_scores, test_labels = probabilities(model, test_loader, device)
    test_metrics = metrics_at_threshold(test_scores, test_labels, float(valid_metrics["threshold"]))
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        git_revision = None
    report = {
        "task": "palm_reject_roi_classification",
        "labels": {str(index): label for index, label in enumerate(LABELS)},
        "model": model_name,
        "backbone": args.backbone,
        "initialization": initialization,
        "image_size": IMAGE_SIZE,
        "normalization": NANODET_NORMALIZE,
        "dataset_provenance": provenance,
        "best_epoch": best_epoch,
        "valid": valid_metrics,
        "test": test_metrics,
        "git_revision": git_revision,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "history": history,
    }
    atomic_json(output / "metrics.json", report)
    atomic_json(output / "inference-contract.json", {
        "input": "RGB uint8 ROI converted to BGR, centered zero-padded LetterBox to 416x416, then normalized with the NanoDet-Plus-m BGR mean/std",
        "output": {"0": "palm", "1": "reject"},
        "palm_acceptance": f"softmax(logits)[palm] >= {float(valid_metrics['threshold']):.3f}",
        "threshold_source": "validation macro-F1 calibration",
        "tensorboard_dir": str(tensorboard_dir),
    })
    atomic_json(output / "status.json", {
        "state": "completed",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": args.backbone,
        "model": model_name,
        "epoch": len(history),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_score,
        "valid": valid_metrics,
        "test": test_metrics,
    })
    writer.add_hparams(
        {"backbone": args.backbone, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "epochs": args.epochs},
        {"test_macro_f1": float(test_metrics["macro_f1"]), "valid_macro_f1": float(valid_metrics["macro_f1"])},
    )
    writer.close()
    print(json.dumps({"best_epoch": best_epoch, "valid": valid_metrics, "test": test_metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

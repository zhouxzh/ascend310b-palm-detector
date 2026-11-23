#!/usr/bin/env python3
"""Tune and train palm/reject classifiers with realistic augmentation."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/zhong/Documents/ascend310b-palm-detector")
PYTHON = Path("/home/zhong/anaconda3/envs/torch/bin/python")
RUN_ID = "20261123-palm-reject-aug-optimization"
RUN_ROOT = ROOT / "artifacts" / RUN_ID
DATASET = ROOT / "data/palm-reject-roi-v005"
BACKBONE_CHECKPOINT = ROOT / "artifacts/pretrained/nanodet-plus-m_416_checkpoint.ckpt"
MAX_FALSE_ACCEPT_RATE = 0.02
BACKBONES = (
    ("nanodet_shufflenetv2_x1", "nanodet-shufflenetv2-x1"),
    ("lcnet_100", "lcnet-100"),
    ("efficientnet_b0", "efficientnet-b0"),
)
TRIALS = (
    ("lr1e4-wd1e4-ls005", 0.0001, 0.0001, 0.05),
    ("lr3e4-wd1e4-ls005", 0.0003, 0.0001, 0.05),
    ("lr3e4-wd5e4-ls010", 0.0003, 0.0005, 0.10),
    ("lr7e4-wd5e4-ls010", 0.0007, 0.0005, 0.10),
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.incoming")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def event(kind: str, **payload: object) -> None:
    record = {"event": kind, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with (RUN_ROOT / "matrix-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return process.returncode


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    allowed_entries = {
        "logs",
        "tensorboard",
        "snapshots",
        "supervisor.json",
        "runner.pid",
        "tensorboard.pid",
        "snapshot.pid",
        "results-http.pid",
        "matrix-events.jsonl",
    }
    unexpected = {path.name for path in RUN_ROOT.iterdir()} - allowed_entries
    if unexpected:
        raise FileExistsError(f"output already contains training artifacts: {sorted(unexpected)}")
    event("started", run_id=RUN_ID, max_false_accept_rate=MAX_FALSE_ACCEPT_RATE)
    trial_results: dict[str, list[dict]] = {}
    for backbone, slug in BACKBONES:
        trial_results[slug] = []
        for trial_name, learning_rate, weight_decay, label_smoothing in TRIALS:
            output = RUN_ROOT / "search" / slug / trial_name
            command = [
                str(PYTHON), "scripts/train_palm_reject_classifier.py",
                "--dataset", str(DATASET),
                "--output", str(output),
                "--backbone", backbone,
                "--backbone-checkpoint", str(BACKBONE_CHECKPOINT),
                "--tensorboard-dir", str(RUN_ROOT / "tensorboard/search" / slug / trial_name),
                "--augmentation-profile", "realistic",
                "--selection-metric", "roc_auc",
                "--max-false-accept-rate", str(MAX_FALSE_ACCEPT_RATE),
                "--label-smoothing", str(label_smoothing),
                "--skip-test",
                "--epochs", "12",
                "--batch-size", "64",
                "--workers", "8",
                "--learning-rate", str(learning_rate),
                "--weight-decay", str(weight_decay),
                "--patience", "4",
                "--seed", "43",
            ]
            event("trial-started", backbone=backbone, trial=trial_name)
            code = run(command, RUN_ROOT / "logs" / "search" / slug / f"{trial_name}.log")
            metrics_path = output / "metrics.json"
            if code != 0 or not metrics_path.is_file():
                trial_results[slug].append({"trial": trial_name, "status": "failed", "exit_code": code})
                event("trial-failed", backbone=backbone, trial=trial_name, exit_code=code)
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = {
                "trial": trial_name,
                "status": "completed",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "best_epoch": metrics["best_epoch"],
                "valid_roc_auc": metrics["valid"]["roc_auc"],
                "valid_macro_f1": metrics["valid"]["macro_f1"],
                "valid_palm_recall": metrics["valid"]["palm_recall"],
                "valid_palm_false_reject_rate": metrics["valid"]["palm_false_reject_rate"],
                "valid_reject_false_accept_rate": metrics["valid"]["reject_false_accept_rate"],
            }
            trial_results[slug].append(result)
            event("trial-completed", backbone=backbone, trial=trial_name, metrics=result)
    selected = {}
    for _, slug in BACKBONES:
        candidates = [item for item in trial_results[slug] if item["status"] == "completed"]
        if not candidates:
            raise RuntimeError(f"no successful tuning trials for {slug}")
        selected[slug] = max(candidates, key=lambda item: (item["valid_roc_auc"], item["valid_macro_f1"], item["valid_palm_recall"]))
        event("trial-selected", backbone=slug, trial=selected[slug]["trial"], valid_roc_auc=selected[slug]["valid_roc_auc"])
    atomic_json(RUN_ROOT / "tuning-summary.json", {
        "run_id": RUN_ID,
        "augmentation_profile": "realistic",
        "selection_metric": "validation_roc_auc",
        "max_false_accept_rate": MAX_FALSE_ACCEPT_RATE,
        "search_epochs": 12,
        "patience": 4,
        "trials": trial_results,
        "selected": selected,
    })
    final_results = {}
    for backbone, slug in BACKBONES:
        best = selected[slug]
        output = RUN_ROOT / "final" / slug
        command = [
            str(PYTHON), "scripts/train_palm_reject_classifier.py",
            "--dataset", str(DATASET),
            "--output", str(output),
            "--backbone", backbone,
            "--backbone-checkpoint", str(BACKBONE_CHECKPOINT),
            "--tensorboard-dir", str(RUN_ROOT / "tensorboard/final" / slug),
            "--augmentation-profile", "realistic",
            "--selection-metric", "roc_auc",
            "--max-false-accept-rate", str(MAX_FALSE_ACCEPT_RATE),
            "--label-smoothing", str(best["label_smoothing"]),
            "--epochs", "30",
            "--batch-size", "64",
            "--workers", "8",
            "--learning-rate", str(best["learning_rate"]),
            "--weight-decay", str(best["weight_decay"]),
            "--patience", "8",
            "--seed", "43",
        ]
        event("final-started", backbone=backbone, trial=best["trial"])
        code = run(command, RUN_ROOT / "logs" / "final" / f"{slug}.log")
        if code == 0 and (output / "metrics.json").is_file():
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            final_results[slug] = {"status": "completed", "trial": best["trial"], "metrics": metrics}
            event("final-completed", backbone=backbone, trial=best["trial"], test=metrics.get("test"))
        else:
            final_results[slug] = {"status": "failed", "trial": best["trial"], "exit_code": code}
            event("final-failed", backbone=backbone, trial=best["trial"], exit_code=code)
    atomic_json(RUN_ROOT / "final-summary.json", {
        "run_id": RUN_ID,
        "dataset": "palm-reject-roi-v005",
        "augmentation_profile": "realistic",
        "selection_metric": "validation_roc_auc",
        "max_false_accept_rate": MAX_FALSE_ACCEPT_RATE,
        "results": final_results,
    })
    (RUN_ROOT / "MATRIX_COMPLETE").touch()
    event("finished", run_id=RUN_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

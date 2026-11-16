#!/usr/bin/env python3
"""Run verified NanoDet variants serially and record exact status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file
from prepare_nanodet_sweep import MATRIX, OUTPUT, ROOT


WEIGHTS = ROOT / "configs/nanodet-sweep-weights-v001.yml"
AUDIT = ROOT / "artifacts/sweeps/palm-v001/sweep-data-audit.json"
STATE = ROOT / "artifacts/sweeps/palm-v001/sweep-state.json"


def completed_run(run_dir: Path) -> bool:
    summary_path = run_dir / "training-summary.json"
    if summary_path.is_file():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8")).get("status") in {"early_stopped", "max_epochs"}
        except json.JSONDecodeError:
            return False
    train_log = run_dir / "train.log"
    return train_log.is_file() and "`Trainer.fit` stopped: `max_epochs=" in train_log.read_text(encoding="utf-8", errors="replace")


def verified_weight(item: dict) -> tuple[Path | None, Path | None, str]:
    if not item.get("checkpoint") or not item.get("manifest"):
        return None, None, "matching official checkpoint is not locally verified"
    checkpoint = ROOT / item["checkpoint"]
    manifest = ROOT / item["manifest"]
    if not checkpoint.is_file() or not manifest.is_file():
        return None, None, "checkpoint or manifest is missing"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("upstream_url") != item.get("source_url"):
        return None, None, "manifest source URL does not match the registry"
    if record.get("sha256") != sha256_file(checkpoint) or record.get("bytes") != checkpoint.stat().st_size:
        return None, None, "checkpoint SHA-256 or byte count does not match manifest"
    return checkpoint, manifest, "verified"


def matching_existing_run(variant: str, seed: int, config_path: Path, audit_sha: str, checkpoint_sha: str) -> tuple[str, Path] | None:
    candidates = (
        f"sweep-v001-{variant}-seed-{seed}",
        f"sweep-v001-{variant}-dev-v001-seed-{seed}",
    )
    for run_id in candidates:
        run_dir = ROOT / "artifacts/runs" / run_id
        manifest_path = run_dir / "run.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_data_and_weight = (
            manifest.get("data_audit", {}).get("sha256") == audit_sha
            and manifest.get("pretrained", {}).get("verified_sha256") == checkpoint_sha
        )
        if same_data_and_weight and manifest.get("base_config", {}).get("sha256") == sha256_file(config_path):
            return run_id, run_dir
        if same_data_and_weight:
            train_log = run_dir / "train.log"
            if completed_run(run_dir):
                return run_id, run_dir
    return None


def write_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incoming")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--configs-dir", type=Path, default=OUTPUT)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue to later variants after a failed training process",
    )
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    weights = yaml.safe_load(args.weights.read_text(encoding="utf-8"))["variants"]
    IDs = {item["id"] for item in matrix["variants"]}
    if set(weights) != IDs:
        raise ValueError("weight registry must cover the exact variant matrix")
    if args.only is not None and set(args.only) - IDs:
        raise ValueError(f"unknown variants: {sorted(set(args.only) - IDs)}")
    if not json.loads(AUDIT.read_text(encoding="utf-8")).get("valid"):
        raise ValueError("full data audit must pass before running the sweep")
    audit_sha = sha256_file(AUDIT)
    state = json.loads(args.state.read_text(encoding="utf-8")) if args.state.is_file() else {"schema_version": 1, "variants": {}}
    for item in matrix["variants"]:
        variant = item["id"]
        if args.only is not None and variant not in args.only:
            continue
        config_path = args.configs_dir / f"{variant}.yml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        checkpoint, manifest, reason = verified_weight(weights[variant])
        if checkpoint is None:
            state["variants"][variant] = {"status": "paused_missing_weight", "reason": reason, "source_url": weights[variant].get("source_url")}
            continue
        matching = matching_existing_run(variant, args.seed, config_path, audit_sha, sha256_file(checkpoint))
        if matching:
            run_id, run_dir = matching
        else:
            run_id = f"sweep-v001-{variant}-dev-v001-seed-{args.seed}"
            run_dir = ROOT / "artifacts/runs" / run_id
        resume = False
        if run_dir.exists():
            if completed_run(run_dir):
                state["variants"][variant] = {"status": "trained", "run_id": run_id, "run_dir": str(run_dir), "ended_at_utc": datetime.now(timezone.utc).isoformat()}
                write_state(args.state, state)
                continue
            existing = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").is_file() else {}
            if existing.get("run_id") != run_id or existing.get("pretrained", {}).get("verified_sha256") != sha256_file(checkpoint):
                state["variants"][variant] = {"status": "blocked_existing_run_mismatch", "run_id": run_id, "run_dir": str(run_dir)}
                write_state(args.state, state)
                continue
            if (run_dir / "model_last.ckpt").is_file():
                resume = True
            elif not (run_dir / "config.yml").is_file():
                state["variants"][variant] = {"status": "blocked_existing_run_incomplete", "run_id": run_id, "run_dir": str(run_dir)}
                write_state(args.state, state)
                continue
        state["variants"][variant] = {"status": "resuming" if resume else "ready", "run_id": run_id, "run_dir": str(run_dir), "weight_sha256": sha256_file(checkpoint)}
        if args.dry_run:
            continue
        write_state(args.state, state)
        try:
            if not run_dir.exists():
                preflight = [
                    sys.executable, str(ROOT / "scripts/create_run.py"),
                    "--base-config", str(config_path), "--run-id", run_id,
                    "--pretrained-checkpoint", str(checkpoint),
                    "--pretrained-manifest", str(manifest),
                    "--data-report", str(AUDIT), "--seed", str(args.seed),
                ]
                subprocess.run(preflight, cwd=ROOT, check=True)
            state["variants"][variant]["status"] = "running"
            state["variants"][variant]["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_state(args.state, state)
            command = [
                "conda", "run", "--no-capture-output", "-n", "palm-nanodet",
                str(ROOT / "scripts/run_with_host_nvidia.sh"),
                "env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'third_party/nanodet'}",
                "python", "scripts/train_nanodet_sweep.py", str(run_dir / "config.yml"), "--seed", str(args.seed),
            ]
            if resume:
                command.append("--resume")
            with (run_dir / "train.log").open("a" if resume else "w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["variants"][variant]["status"] = "trained"
        except subprocess.CalledProcessError as exc:
            state["variants"][variant]["status"] = "failed"
            state["variants"][variant]["exit_code"] = exc.returncode
        state["variants"][variant]["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(args.state, state)
        if state["variants"][variant]["status"] == "failed" and not args.continue_on_error:
            print(f"stopping after {variant} failed; inspect {run_dir / 'train.log'}", file=sys.stderr)
            return state["variants"][variant]["exit_code"] or 1
    if args.dry_run:
        print(json.dumps(state, indent=2, sort_keys=True))
    else:
        write_state(args.state, state)
        print(args.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

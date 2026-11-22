#!/usr/bin/env python3
"""Run standard PicoDet variants serially with resumable state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifacts/picodet-surface3-320-v004/variant-inventory.json"
STATE = ROOT / "artifacts/picodet-surface3-320-v004/sweep-state.json"
RUNS_ROOT = ROOT / "artifacts/picodet-surface3-320-v004/runs"
RUN_PREFIX = "picodet-surface3-320-v004"
UPSTREAM = ROOT / "third_party/PaddleDetection"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def completed(run_dir: Path) -> bool:
    summary = run_dir / "training-summary.json"
    if not summary.is_file():
        return False
    return json.loads(summary.read_text(encoding="utf-8")).get("status") in {"early_stopped", "max_epochs"}


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".incoming")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-prefix", default=RUN_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.inventory.read_text(encoding="utf-8"))
    variants = {item["id"]: Path(item["path"]) for item in matrix["variants"]}
    if args.only is not None and set(args.only) - set(variants):
        raise ValueError(f"unknown variants: {sorted(set(args.only) - set(variants))}")
    state = json.loads(args.state.read_text(encoding="utf-8")) if args.state.is_file() else {"schema_version": 1, "variants": {}}
    for variant, manifest_path in variants.items():
        if args.only is not None and variant not in args.only:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = f"{args.run_prefix}-{variant}-seed-{args.seed}"
        run_dir = args.runs_root / run_id
        if completed(run_dir):
            state["variants"][variant] = {"status": "trained", "run_id": run_id, "run_dir": str(run_dir), "ended_at_utc": now()}
            write_state(args.state, state)
            continue
        resume = run_dir.exists() and any((run_dir / "paddle-output").glob("*.pdopt"))
        state["variants"][variant] = {"status": "resuming" if resume else "ready", "run_id": run_id, "run_dir": str(run_dir), "manifest": str(manifest_path)}
        write_state(args.state, state)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        state["variants"][variant]["status"] = "running"
        state["variants"][variant]["started_at_utc"] = now()
        write_state(args.state, state)
        command = [
            sys.executable, str(ROOT / "scripts/train_picodet.py"),
            "--manifest", str(manifest_path), "--run-dir", str(run_dir),
            "--upstream", str(args.upstream), "--seed", str(args.seed),
        ]
        if resume:
            command.append("--resume")
        try:
            with (run_dir / "train.log").open("a" if resume else "w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["variants"][variant]["status"] = "trained"
        except subprocess.CalledProcessError as exc:
            state["variants"][variant]["status"] = "failed"
            state["variants"][variant]["exit_code"] = exc.returncode
            state["variants"][variant]["ended_at_utc"] = now()
            write_state(args.state, state)
            return exc.returncode or 1
        state["variants"][variant]["ended_at_utc"] = now()
        write_state(args.state, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

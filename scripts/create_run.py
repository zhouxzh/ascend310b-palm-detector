#!/usr/bin/env python3
"""Create an immutable NanoDet run directory and provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from palm_detector.dataset_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    return {name: _version(name) for name in ("torch", "torchvision", "pytorch-lightning", "onnx", "onnxruntime", "pycocotools", "opencv-python")}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_pretrained(manifest_path: Path, checkpoint_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("upstream_url", "upstream_revision_or_release", "license", "sha256", "bytes")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"pretrained manifest is incomplete: {missing}")
    digest = sha256_file(checkpoint_path)
    if manifest["sha256"] != digest:
        raise ValueError("pretrained checkpoint SHA-256 does not match its manifest")
    if int(manifest["bytes"]) != checkpoint_path.stat().st_size:
        raise ValueError("pretrained checkpoint byte count does not match its manifest")
    return {**manifest, "manifest_path": str(manifest_path), "resolved_checkpoint_path": str(checkpoint_path.resolve()), "verified_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in args.run_id):
        parser.error("run id may contain only letters, digits, '-', '_', and '.'")
    run_dir = ROOT / "artifacts/runs" / args.run_id
    if run_dir.exists():
        parser.error(f"run directory already exists: {run_dir}")
    if not args.pretrained_checkpoint.is_file() or not args.pretrained_manifest.is_file() or not args.data_report.is_file():
        parser.error("pretrained checkpoint, pretrained manifest, and data report must exist")
    pretrained = _validate_pretrained(args.pretrained_manifest, args.pretrained_checkpoint)
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if not data_report.get("valid"):
        parser.error("data report is not valid")
    base_bytes = args.base_config.read_bytes()
    config = yaml.safe_load(base_bytes)
    config["save_dir"] = str(run_dir.relative_to(ROOT))
    config.setdefault("schedule", {})["load_model"] = str(args.pretrained_checkpoint.resolve())
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    annotation_hashes = {}
    for split in data_report.get("splits", []):
        path = Path(split["annotation_path"])
        if path.is_file():
            annotation_hashes[split["split"]] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "project": {"git_commit": _git(["git", "rev-parse", "HEAD"]), "git_status_porcelain": _git(["git", "status", "--porcelain"]), "root": str(ROOT)},
        "upstream_nanodet": {"url": "https://github.com/RangiLyu/nanodet.git", "commit": _git(["git", "-C", str(ROOT / "third_party/nanodet"), "rev-parse", "HEAD"])},
        "base_config": {"path": str(args.base_config), "sha256": hashlib.sha256(base_bytes).hexdigest()},
        "run_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "data_audit": {"path": str(args.data_report), "sha256": sha256_file(args.data_report), "split_annotation_sha256": annotation_hashes},
        "pretrained": pretrained,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": _package_versions()},
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

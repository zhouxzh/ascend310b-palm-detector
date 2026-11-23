#!/usr/bin/env python3
"""Publish a 10-minute tarball snapshot of palm/reject classifier results."""
from __future__ import annotations

import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/zhong/Documents/ascend310b-palm-detector")
RUN_ID = os.environ.get("PALMPRINT_CLASSIFIER_RUN_ID", "20261123-palm-reject-three-backbones")
RUN_ROOT = ROOT / "artifacts" / RUN_ID
SNAPSHOT_ROOT = RUN_ROOT / "snapshots"
INTERVAL_SECONDS = 600
PATTERNS = (
    "summary.json",
    "matrix-events.jsonl",
    "MATRIX_COMPLETE",
    "supervisor.json",
    "logs/*.log",
    "*/status.json",
    "*/metrics.json",
    "*/inference-contract.json",
    "*/best.pt",
    "*/last.pt",
    "tensorboard/**/*",
)


def collect() -> list[Path]:
    files: list[Path] = []
    for pattern in PATTERNS:
        files.extend(path for path in RUN_ROOT.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: str(path))


def write_state(payload: dict) -> None:
    target = SNAPSHOT_ROOT / "snapshot-state.json"
    temporary = target.with_suffix(".json.incoming")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> int:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(timezone.utc)
        files = collect()
        if files:
            name = timestamp.strftime("%Y%m%dT%H%M%SZ") + ".tar.gz"
            archive = SNAPSHOT_ROOT / name
            incoming = archive.with_suffix(".tar.gz.incoming")
            with tarfile.open(incoming, "w:gz") as bundle:
                for path in files:
                    bundle.add(path, arcname=str(path.relative_to(RUN_ROOT)))
            os.replace(incoming, archive)
            latest = SNAPSHOT_ROOT / "latest.tar.gz"
            os.replace(archive, latest)
            write_state({
                "updated_at_utc": timestamp.isoformat(),
                "latest": str(latest),
                "bytes": latest.stat().st_size,
                "file_count": len(files),
                "interval_seconds": INTERVAL_SECONDS,
            })
        else:
            write_state({"updated_at_utc": timestamp.isoformat(), "state": "waiting-for-results", "interval_seconds": INTERVAL_SECONDS})
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())

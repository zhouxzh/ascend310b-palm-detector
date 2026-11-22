"""Process-scoped locks for resumable experiment directories."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO


def acquire_training_lock(run_dir: Path) -> IO[str]:
    """Wait for exclusive ownership of one run directory's training phase.

    The returned handle must remain referenced until the protected process is
    finished. Closing it, or process exit, releases the advisory lock.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / ".training.lock").open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle

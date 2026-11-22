from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
import subprocess
import sys
import time

from palm_detector.run_lock import acquire_training_lock


class RunLockTest(unittest.TestCase):
    def test_lock_file_is_created_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = acquire_training_lock(run_dir)
            self.assertTrue((run_dir / ".training.lock").is_file())
            first.close()
            second = acquire_training_lock(run_dir)
            second.close()

    def test_second_process_waits_until_lock_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            marker = Path(temporary) / "acquired"
            first = acquire_training_lock(run_dir)
            environment = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            command = (
                "from pathlib import Path; "
                "from palm_detector.run_lock import acquire_training_lock; "
                f"handle=acquire_training_lock(Path({str(run_dir)!r})); "
                f"Path({str(marker)!r}).write_text('acquired', encoding='utf-8'); "
                "handle.close()"
            )
            waiting = subprocess.Popen([sys.executable, "-c", command], env=environment)
            time.sleep(0.1)
            self.assertFalse(marker.exists())
            first.close()
            self.assertEqual(waiting.wait(timeout=5), 0)
            self.assertTrue(marker.is_file())

    def test_controller_creates_a_run_when_only_lock_files_exist(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_single_palm_all_variants.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ ! -f "$run_dir/config.yml" ]]; then', source)
        self.assertIn('local lock_root="$EXPERIMENT_ROOT/.run-locks"', source)
        self.assertNotIn('exec {run_lock_fd}>"$run_dir/.run.lock"', source)

    def test_controller_retries_failed_runs_from_their_existing_artifacts(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_single_palm_all_variants.sh").read_text(encoding="utf-8")
        self.assertIn('readonly MAX_RUN_ATTEMPTS="${SINGLE_PALM_MAX_RUN_ATTEMPTS:-3}"', source)
        self.assertIn('run_with_retries()', source)
        self.assertIn('record_state "$family" "$geometry" "$variant" retrying', source)
        self.assertIn('failed after ${attempt}/${MAX_RUN_ATTEMPTS} attempts', source)

#!/usr/bin/env python3
"""Run NanoDet's test entry point after loading project preprocess extensions."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/nanodet"))

from palm_detector.nanodet_extensions import install_nanodet_extensions

install_nanodet_extensions()
runpy.run_path(str(ROOT / "third_party/nanodet/tools/test.py"), run_name="__main__")

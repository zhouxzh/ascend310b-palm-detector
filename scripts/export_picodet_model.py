#!/usr/bin/env python3
"""Run PaddleDetection's exporter after registering project extensions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/PaddleDetection"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))

from palm_detector.paddledetection_extensions import install_paddledetection_extensions

install_paddledetection_extensions()

from tools.export_model import main


if __name__ == "__main__":
    main()

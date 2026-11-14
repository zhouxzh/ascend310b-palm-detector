#!/usr/bin/env python3
"""Compare an ACL raw output tensor with its ONNX reference tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-abs-error", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = np.load(args.reference)
    candidate = np.load(args.candidate)
    errors: list[str] = []
    if reference.shape != candidate.shape:
        errors.append(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
        max_error = mean_error = None
    else:
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        max_error = float(difference.max())
        mean_error = float(difference.mean())
        if not np.isfinite(difference).all():
            errors.append("candidate or reference contains non-finite values")
        elif max_error > args.max_abs_error:
            errors.append(f"max absolute error {max_error} exceeds {args.max_abs_error}")
    report = {"valid": not errors, "reference": str(args.reference), "candidate": str(args.candidate), "shape": list(reference.shape), "max_abs_error_limit": args.max_abs_error, "max_abs_error": max_error, "mean_abs_error": mean_error, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {'PASS' if report['valid'] else 'FAIL'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

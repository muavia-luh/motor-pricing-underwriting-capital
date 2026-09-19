"""Entry point: run the whole pipeline (frequency GLM + ML, severity GLM + tail +
ML, pure premium, capital) and write results/figures to reports/.

Usage:
    python -m motor_pricing.pipeline            # from src/, or with the package installed
    python scripts/run_all.py                   # convenience wrapper from the repo root
"""
from __future__ import annotations
import os
from . import reporting


def main(root=None):
    if root is None:
        # repo root = two levels up from this file (src/motor_pricing/pipeline.py)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reporting.run(root)


if __name__ == "__main__":
    main()

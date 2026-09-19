"""Convenience wrapper: run the full pipeline from the repo root without installing."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from motor_pricing.pipeline import main
main(ROOT)

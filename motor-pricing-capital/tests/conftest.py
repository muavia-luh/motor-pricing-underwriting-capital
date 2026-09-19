"""Make `src/` importable when running the tests without installing the package."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

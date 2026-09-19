"""Fetch the CASdatasets freMTPL2 data into data/raw/ (clones the package and parses
the .rda with the pure-Python reader). Run once before the pipeline."""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from rda_reader import load_rda
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
REPO = "https://github.com/dutangc/CASdatasets.git"
# Pin an exact commit so the data never shifts under the results.
COMMIT = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"   # CASdatasets, 2026-06-10

def main():
    os.makedirs(RAW, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "CASdatasets")
        # Clone then check out the pinned commit (works even without shallow fetch).
        subprocess.run(["git", "clone", REPO, dst], check=True)
        subprocess.run(["git", "-C", dst, "checkout", "--quiet", COMMIT], check=True)
        for obj in ["freMTPL2freq", "freMTPL2sev"]:
            _, df = load_rda(os.path.join(dst, "data", f"{obj}.rda"))
            df.to_csv(os.path.join(RAW, f"{obj}.csv"), index=False)
            print("wrote", obj, df.shape)

if __name__ == "__main__":
    main()

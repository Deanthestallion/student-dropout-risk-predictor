"""
Download the UCI Student Performance dataset into this folder.

    python data/download_data.py

Fetches student-por.csv and student-mat.csv from the UCI Machine Learning Repository.
notebook.ipynb calls the same logic automatically, so running this by hand is optional.

Source: Cortez, P. and Silva, A. (2008). "Using Data Mining to Predict Secondary School
Student Performance". UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/320/student+performance
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEST = Path(__file__).parent
WANTED = ["student-por.csv", "student-mat.csv"]

# Row counts documented by UCI, used as an integrity check after download.
EXPECTED_ROWS = {"student-por.csv": 650, "student-mat.csv": 396}  # includes the header row


def download(force: bool = False) -> int:
    if not force and all((DEST / f).exists() for f in WANTED):
        print("Both CSVs are already present. Pass --force to re-download.")
    else:
        print(f"Downloading {UCI_ZIP}")
        try:
            raw = urllib.request.urlopen(UCI_ZIP, timeout=60).read()
        except Exception as exc:
            print(f"FAILED: could not reach UCI ({exc}).")
            print("Download the archive manually from the URL above and unzip it into data/.")
            return 1
        print(f"  received {len(raw) / 1024:.0f} KB, sha256 {hashlib.sha256(raw).hexdigest()[:16]}")

        outer = zipfile.ZipFile(io.BytesIO(raw))
        # The UCI archive nests a second zip that holds the CSVs.
        source = (zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))
                  if "student.zip" in outer.namelist() else outer)
        for name in WANTED:
            (DEST / name).write_bytes(source.read(name))
            print(f"  extracted {name}")

    ok = True
    for name, expected in EXPECTED_ROWS.items():
        path = DEST / name
        if not path.exists():
            print(f"  [FAIL] {name} is missing")
            ok = False
            continue
        rows = sum(1 for _ in path.open(encoding="utf-8"))
        status = "PASS" if rows == expected else "FAIL"
        print(f"  [{status}] {name}: {rows} lines (expected {expected})")
        ok &= rows == expected

    print("\nData ready." if ok else "\nIntegrity check failed. Re-run with --force.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(download(force="--force" in sys.argv))

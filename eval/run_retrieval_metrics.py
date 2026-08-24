#!/usr/bin/env python3
"""One-command GS-T3 retrieval measurement.

Downloads the movie corpus and evaluation fixtures if they are missing,
then runs the evaluation CLI for BM25, semantic, RRF, and RRF plus
cross-encoder against data/golden_dataset.json.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MOVIES_URL = (
    "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/movies.json"
)
OUTPUT_PATH = REPO_ROOT / "eval" / "results.json"


def _copy_fixture(name: str, dest: Path) -> None:
    src = FIXTURES_DIR / name
    if not src.exists():
        raise FileNotFoundError(f"Missing fixture {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden = DATA_DIR / "golden_dataset.json"
    stopwords = DATA_DIR / "stopwords.txt"
    movies = DATA_DIR / "movies.json"

    if not golden.exists():
        _copy_fixture("golden_dataset.json", golden)
    if not stopwords.exists():
        _copy_fixture("stopwords.txt", stopwords)
    if not movies.exists():
        print(f"Downloading {MOVIES_URL} -> {movies}")
        urllib.request.urlretrieve(MOVIES_URL, movies)


def main() -> int:
    ensure_data()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cli" / "evaluation_cli.py"),
        "--limit",
        "5",
        "--output",
        str(OUTPUT_PATH),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

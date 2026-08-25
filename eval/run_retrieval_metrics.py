#!/usr/bin/env python3
"""One-command GS-T3 retrieval measurement.

If data/golden_dataset.json or data/stopwords.txt are missing, copies them
from eval/fixtures. If they exist and differ from those fixtures, exits
non-zero and does not run evaluation or write eval/results.json.
Downloads the movie corpus if it is missing, then runs the evaluation CLI
for BM25, semantic, RRF, and RRF plus cross-encoder.
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
FIXTURE_NAMES = ("golden_dataset.json", "stopwords.txt")
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
    mismatches: list[str] = []
    for name in FIXTURE_NAMES:
        dest = DATA_DIR / name
        src = FIXTURES_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing fixture {src}")
        if not dest.exists():
            _copy_fixture(name, dest)
        elif dest.read_bytes() != src.read_bytes():
            mismatches.append(name)
    if mismatches:
        names = ", ".join(mismatches)
        print(
            f"data/ does not match eval/fixtures for: {names}. "
            "Refusing to evaluate a dirty data/ directory.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    movies = DATA_DIR / "movies.json"
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

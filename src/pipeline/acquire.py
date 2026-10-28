"""S1 Acquire: copy the labelled Kaggle file to data/raw/delivery_raw.csv and checksum it.

Sources (first match wins):
  1. ``--source PATH``: a local train.csv or the Kaggle zip (manual download)
  2. ``kagglehub.dataset_download`` (public dataset; uses ~/.kaggle credentials if present)

Only ``train.csv`` is used; Kaggle's ``test.csv`` has no target column, so it can't be scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import CITATION, KAGGLE_DATASET, KAGGLE_URL, Paths

LABELLED_MEMBER = "train.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _from_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if Path(n).name == LABELLED_MEMBER]
        if not names:
            raise FileNotFoundError(f"{LABELLED_MEMBER} not found inside {zip_path}")
        # Stream the single member rather than extractall (avoids path traversal).
        with zf.open(names[0]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)


def _from_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "kagglehub is not installed. Either `pip install kagglehub` or download the zip "
            f"manually from {KAGGLE_URL} and pass --source path/to/archive.zip"
        ) from e
    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return folder / LABELLED_MEMBER


def acquire(paths: Paths, source: Path | None = None) -> dict:
    paths.ensure_dirs()
    dest = paths.raw_csv
    if source is not None:
        source = Path(source)
        if source.suffix == ".zip":
            _from_zip(source, dest)
        else:
            shutil.copyfile(source, dest)
        origin = f"local file: {source.name}"
    else:
        src = _from_kagglehub()
        shutil.copyfile(src, dest)
        origin = f"kagglehub: {KAGGLE_DATASET}"

    with open(dest, encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    meta = {
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": origin,
        "file": LABELLED_MEMBER,
        "sha256": sha256_file(dest),
        "n_rows": n_rows,
        "citation": CITATION,
        "licence_note": (
            "Public Kaggle dataset. Licence terms are those shown on the Kaggle dataset page; "
            "check them before any redistribution. Raw data is git-ignored and not redistributed "
            "by this repo. No new personal data is collected; no re-identification is attempted."
        ),
    }
    paths.raw_meta.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=None, help="local train.csv or Kaggle zip")
    args = ap.parse_args()
    meta = acquire(Paths.default(), args.source)
    print(f"[S1] {meta['n_rows']} rows -> data/raw/delivery_raw.csv sha256={meta['sha256'][:12]}…")


if __name__ == "__main__":
    main()

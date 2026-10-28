"""Run the S1–S6 pipeline end to end.

    python -m src.pipeline.run [--source PATH] [--skip-acquire] [--refreeze]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Paths
from src.pipeline import acquire, clean, document, engineer, freeze, profile


def run(paths: Paths, source: Path | None = None, skip_acquire: bool = False,
        refreeze: bool = False) -> dict:
    if not skip_acquire:
        meta = acquire.acquire(paths, source)
        print(f"[S1] {meta['n_rows']:,} raw rows (sha256 {meta['sha256'][:12]}…)")
    elif not paths.raw_csv.exists():
        raise FileNotFoundError(f"{paths.raw_csv} missing; run without --skip-acquire")
    profile.profile(paths)
    print("[S2] profile -> reports/eda/profile_notes.md")
    df = clean.clean(paths)
    print(f"[S3] {len(df):,} clean rows")
    feats = engineer.engineer(paths)
    print(f"[S4] features {feats.shape}")
    document.document(paths)
    print("[S5] data dictionary + transform log")
    split = freeze.freeze(paths, refreeze=refreeze)
    print(f"[S6] train={len(split['train']):,} holdout={len(split['holdout']):,}")
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--skip-acquire", action="store_true")
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args()
    run(Paths.default(), a.source, a.skip_acquire, a.refreeze)


if __name__ == "__main__":
    main()

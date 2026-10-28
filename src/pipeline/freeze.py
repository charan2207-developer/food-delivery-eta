"""S6 Freeze: seeded ~80/20 train/hold-out split on row_id -> data/splits/row_ids.json.

Change control (risk R8): if a split file already exists and a fresh split would
differ (features changed), we refuse unless ``--refreeze`` is passed. After a
refreeze, models must be retrained and metrics republished (run_all.sh does both).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.config import HOLDOUT_FRACTION, ROW_ID, SEED, Paths
from src.pipeline.engineer import load_features


class SplitDriftError(RuntimeError):
    pass


def make_split(row_ids, seed: int = SEED, holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    ids = np.sort(np.asarray(row_ids, dtype=np.int64))
    perm = np.random.default_rng(seed).permutation(ids)
    n_hold = int(round(len(ids) * holdout_fraction))
    return {"train": sorted(map(int, perm[n_hold:])), "holdout": sorted(map(int, perm[:n_hold]))}


def freeze(paths: Paths, refreeze: bool = False) -> dict:
    split = make_split(load_features(paths)[ROW_ID])
    if paths.row_ids.exists() and not refreeze:
        existing = json.loads(paths.row_ids.read_text())
        if existing != split:
            raise SplitDriftError(
                "Existing data/splits/row_ids.json differs from a fresh split (features changed?). "
                "Re-run with --refreeze, then retrain models and republish metrics."
            )
        return existing
    paths.row_ids.parent.mkdir(parents=True, exist_ok=True)
    paths.row_ids.write_text(json.dumps(split))
    return split


def load_split(paths: Paths) -> dict:
    return json.loads(paths.row_ids.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refreeze", action="store_true")
    split = freeze(Paths.default(), ap.parse_args().refreeze)
    print(f"[S6] train={len(split['train'])} holdout={len(split['holdout'])} (seed={SEED})")


if __name__ == "__main__":
    main()

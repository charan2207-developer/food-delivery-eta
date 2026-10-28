"""Shared configuration: paths (interface contract), seed, and governance constants.

Every stage takes a ``Paths`` object so the whole system can be re-rooted
(e.g. into a pytest tmp dir with a synthetic fixture) without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
HOLDOUT_FRACTION = 0.20
SUBGROUP_MIN_N = 50
TARGET = "time_taken_min"
ROW_ID = "row_id"

KAGGLE_DATASET = "gauravmalik26/food-delivery-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset"
CITATION = "Malik, G. (2022). Food delivery dataset [Data set]. Kaggle. " + KAGGLE_URL

INTENDED_USE_NOTE = "ops-proxy demo; not production ETA"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(Path(os.environ.get("FDE_ROOT", PROJECT_ROOT)).resolve())

    # --- data (pipeline) ---
    @property
    def raw_csv(self) -> Path:
        return self.root / "data/raw/delivery_raw.csv"

    @property
    def raw_meta(self) -> Path:
        return self.root / "data/raw/acquisition.json"

    @property
    def interim_csv(self) -> Path:
        return self.root / "data/interim/delivery_clean.csv"

    @property
    def clean_stats(self) -> Path:
        return self.root / "data/interim/clean_stats.json"

    @property
    def features(self) -> Path:
        return self.root / "data/features/delivery_features.parquet"

    @property
    def row_ids(self) -> Path:
        return self.root / "data/splits/row_ids.json"

    @property
    def profile_notes(self) -> Path:
        return self.root / "reports/eda/profile_notes.md"

    # --- docs ---
    @property
    def data_dictionary(self) -> Path:
        return self.root / "docs/data_dictionary.md"

    @property
    def transform_log(self) -> Path:
        return self.root / "docs/transform_log.md"

    @property
    def model_card(self) -> Path:
        return self.root / "docs/model_card.md"

    @property
    def datasheet(self) -> Path:
        return self.root / "docs/datasheet_lite.md"

    # --- artifacts (models + evaluation) ---
    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def baseline_metrics(self) -> Path:
        return self.artifacts / "baseline_metrics.json"

    @property
    def leaderboard(self) -> Path:
        return self.artifacts / "leaderboard.csv"

    @property
    def y_pred_holdout(self) -> Path:
        return self.artifacts / "y_pred_holdout.csv"

    @property
    def training_meta(self) -> Path:
        return self.artifacts / "training_meta.json"

    @property
    def metrics(self) -> Path:
        return self.artifacts / "metrics.json"

    @property
    def subgroup_metrics(self) -> Path:
        return self.artifacts / "subgroup_metrics.csv"

    @property
    def feature_importance(self) -> Path:
        return self.artifacts / "feature_importance.csv"

    @property
    def metrics_history(self) -> Path:
        return self.artifacts / "history"

    def ensure_dirs(self) -> None:
        for p in [
            self.raw_csv, self.interim_csv, self.features, self.row_ids,
            self.profile_notes, self.data_dictionary, self.leaderboard,
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history.mkdir(parents=True, exist_ok=True)

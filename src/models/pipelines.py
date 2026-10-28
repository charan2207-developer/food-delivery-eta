"""Candidate model definitions (M1–M3) as sklearn Pipelines.

All preprocessing (imputation, scaling, one-hot) lives inside the pipeline so
it is fit on whatever rows ``fit`` sees — in practice, the training split only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.config import SEED


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer(
        [("num", make_pipeline(*num_steps), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        verbose_feature_names_out=False,
    )


@dataclass
class Candidate:
    name: str
    family: str          # linear | rf | boost  (drives which M-step artefact it competes for)
    step: str            # M1 / M2 / M3
    estimator: object
    scale: bool
    param_grid: dict = field(default_factory=dict)
    fast_grid: dict = field(default_factory=dict)
    n_iter: int = 0      # 0 = exhaustive grid; >0 = randomized search

    def pipeline(self, numeric: list[str], categorical: list[str]) -> Pipeline:
        from sklearn.base import clone
        return Pipeline([("prep", make_preprocessor(numeric, categorical, self.scale)),
                         ("model", clone(self.estimator))])


def candidates(fast: bool = False) -> list[Candidate]:
    rf_trees, boost_trees = (30, 60) if fast else (200, 600)
    return [
        Candidate("m1_ols", "linear", "M1", LinearRegression(), scale=True),
        Candidate(
            "m1_elasticnet", "linear", "M1", ElasticNet(max_iter=5000, random_state=SEED),
            scale=True,
            param_grid={"model__alpha": [0.001, 0.01, 0.1, 1.0],
                        "model__l1_ratio": [0.1, 0.5, 0.9]},
            fast_grid={"model__alpha": [0.01, 0.1], "model__l1_ratio": [0.5]},
        ),
        Candidate(
            "m2_random_forest", "rf", "M2",
            RandomForestRegressor(n_estimators=rf_trees, random_state=SEED, n_jobs=-1),
            scale=False,
            param_grid={"model__max_depth": [None, 20, 12],
                        "model__min_samples_leaf": [1, 3, 8],
                        "model__max_features": [0.33, 0.6, 1.0]},
            fast_grid={"model__max_depth": [8], "model__min_samples_leaf": [3],
                       "model__max_features": [0.6]},
            n_iter=6,
        ),
        Candidate(
            "m3_xgboost", "boost", "M3",
            XGBRegressor(n_estimators=boost_trees, tree_method="hist", random_state=SEED,
                         n_jobs=-1, verbosity=0),
            scale=False,
            param_grid={"model__learning_rate": [0.03, 0.05, 0.1],
                        "model__max_depth": [4, 6, 8],
                        "model__min_child_weight": [1, 5, 10],
                        "model__subsample": [0.7, 0.9, 1.0],
                        "model__colsample_bytree": [0.6, 0.8, 1.0]},
            fast_grid={"model__learning_rate": [0.1], "model__max_depth": [4]},
            n_iter=10,
        ),
        Candidate(
            "m3_lightgbm", "boost", "M3",
            # n_jobs=1: on ~36k rows threading overhead makes LightGBM ~3x slower on macOS
            LGBMRegressor(n_estimators=boost_trees, random_state=SEED, n_jobs=1,
                          deterministic=True, force_col_wise=True, verbose=-1),
            scale=False,
            param_grid={"model__learning_rate": [0.03, 0.05, 0.1],
                        "model__num_leaves": [15, 31, 63],
                        "model__min_child_samples": [10, 20, 50],
                        "model__subsample": [0.7, 0.9, 1.0],
                        "model__subsample_freq": [1],
                        "model__colsample_bytree": [0.6, 0.8, 1.0]},
            fast_grid={"model__learning_rate": [0.1], "model__num_leaves": [15]},
            n_iter=10,
        ),
    ]


def candidate_by_name(name: str, fast: bool = False) -> Candidate:
    return next(c for c in candidates(fast) if c.name == name)

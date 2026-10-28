"""M0–M5 model ladder under the frozen split.

    python -m src.models.train [--fast]

Selection protocol (no hold-out peeping):
  * hyper-parameters chosen by 5-fold CV on the TRAIN split only (MAE objective)
  * preferred model = lowest train-CV MAE among M1–M3 candidates
  * the hold-out is scored once per candidate after fitting, for reporting only
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate

from src.config import ROW_ID, SEED, TARGET, Paths
from src.evaluation.metrics import score
from src.models.pipelines import Candidate, candidates
from src.pipeline.acquire import sha256_file
from src.pipeline.engineer import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PREDICTORS, load_features
from src.pipeline.freeze import load_split

CV_FOLDS = 5
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
FAMILY_ARTIFACT = {"linear": "linear_model.joblib", "rf": "rf_model.joblib",
                   "boost": "boost_model.joblib"}


def load_frames(paths: Paths):
    feats = load_features(paths).set_index(ROW_ID)
    split = load_split(paths)
    train_ids, hold_ids = split["train"], split["holdout"]
    assert set(train_ids).isdisjoint(hold_ids), "train/holdout overlap"
    missing = (set(train_ids) | set(hold_ids)) - set(feats.index)
    if missing:
        raise RuntimeError(f"{len(missing)} split row_ids not in feature table: re-freeze (S6) "
                           "and retrain")
    train, hold = feats.loc[train_ids], feats.loc[hold_ids]
    return train[PREDICTORS], train[TARGET], hold[PREDICTORS], hold[TARGET]


def search(c: Candidate, X, y, cv, fast: bool, numeric, categorical) -> dict:
    pipe = c.pipeline(numeric, categorical)
    grid = c.fast_grid if fast else c.param_grid
    t0 = time.perf_counter()
    if not grid:
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
        best = {"params": {}, "cv_mae": -res["test_mae"].mean(), "cv_rmse": -res["test_rmse"].mean()}
        model = pipe.fit(X, y)
    else:
        kw = dict(scoring=SCORING, refit="mae", cv=cv, n_jobs=1)
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        if c.n_iter and c.n_iter < n_combos:
            s = RandomizedSearchCV(pipe, grid, n_iter=c.n_iter, random_state=SEED, **kw)
        else:
            s = GridSearchCV(pipe, grid, **kw)
        s.fit(X, y)
        i = s.best_index_
        best = {"params": {k.replace("model__", ""): v for k, v in s.best_params_.items()},
                "cv_mae": -s.cv_results_["mean_test_mae"][i],
                "cv_rmse": -s.cv_results_["mean_test_rmse"][i]}
        model = s.best_estimator_
    best["fit_seconds"] = round(time.perf_counter() - t0, 1)
    best["model"] = model
    return best


def train_all(paths: Paths, fast: bool = False,
              numeric: list[str] | None = None, categorical: list[str] | None = None) -> dict:
    numeric = numeric or NUMERIC_FEATURES
    categorical = categorical or CATEGORICAL_FEATURES
    X_tr, y_tr, X_ho, y_ho = load_frames(paths)
    X_tr, X_ho = X_tr[numeric + categorical], X_ho[numeric + categorical]
    cv = KFold(n_splits=3 if fast else CV_FOLDS, shuffle=True, random_state=SEED)
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    # M0: naive training-mean baseline (CV'd the same way for a like-for-like leaderboard)
    train_mean = float(y_tr.mean())
    fold_scores = [score(y_tr.iloc[v], np.full(len(v), y_tr.iloc[t].mean()))
                   for t, v in cv.split(X_tr)]
    naive = {"model": "m0_naive", "train_mean": round(train_mean, 4),
             "cv_mae": float(np.mean([s["mae"] for s in fold_scores])),
             "cv_rmse": float(np.mean([s["rmse"] for s in fold_scores])),
             "holdout": score(y_ho, np.full(len(y_ho), train_mean)),
             "n_train": int(len(y_tr)), "n_holdout": int(len(y_ho))}
    paths.baseline_metrics.write_text(json.dumps(naive, indent=2))
    rows = [{"model": "m0_naive", "step": "M0", "family": "naive",
             "cv_mae": naive["cv_mae"], "cv_rmse": naive["cv_rmse"],
             "holdout_mae": naive["holdout"]["mae"], "holdout_rmse": naive["holdout"]["rmse"],
             "fit_seconds": 0.0, "best_params": "{}"}]
    print(f"[M0] naive mean={train_mean:.2f}  holdout {naive['holdout']}")

    fitted, preds = {}, {}
    for c in candidates(fast):
        r = search(c, X_tr, y_tr, cv, fast, numeric, categorical)
        preds[c.name] = r["model"].predict(X_ho)  # scored for reporting; not used for selection
        ho = score(y_ho, preds[c.name])
        fitted[c.name] = (c, r["model"])
        rows.append({"model": c.name, "step": c.step, "family": c.family,
                     "cv_mae": r["cv_mae"], "cv_rmse": r["cv_rmse"],
                     "holdout_mae": ho["mae"], "holdout_rmse": ho["rmse"],
                     "fit_seconds": r["fit_seconds"], "best_params": json.dumps(r["params"])})
        print(f"[{c.step}] {c.name:<18} cv_mae={r['cv_mae']:.3f}  holdout {ho}  "
              f"({r['fit_seconds']}s)")

    board = pd.DataFrame(rows)
    for col in ["cv_mae", "cv_rmse"]:
        board[col] = board[col].round(4)
    cand = board[board["family"] != "naive"].sort_values(["cv_mae", "cv_rmse"])
    preferred = cand.iloc[0]["model"]
    board["selected"] = board["model"].eq(preferred)
    board = board.sort_values("cv_mae").reset_index(drop=True)

    # M1–M3 family artefacts: best of each family by train-CV MAE
    for family, fname in FAMILY_ARTIFACT.items():
        name = cand[cand["family"] == family].iloc[0]["model"]
        joblib.dump(fitted[name][1], paths.models_dir / fname)
    # M5: preferred model + hold-out predictions
    joblib.dump(fitted[preferred][1], paths.models_dir / "preferred_model.joblib")
    board.to_csv(paths.leaderboard, index=False)
    pd.DataFrame({ROW_ID: X_ho.index.astype("int64"), "y_true": y_ho.to_numpy(),
                  "y_pred": np.round(preds[preferred], 4)}).to_csv(paths.y_pred_holdout, index=False)

    meta = {
        "seed": SEED, "cv_folds": cv.get_n_splits(), "fast_mode": fast,
        "selection_rule": "lowest train-CV MAE among M1-M3 (hold-out not used for selection)",
        "preferred_model": preferred,
        "preferred_params": json.loads(board.loc[board["selected"], "best_params"].iloc[0]),
        "numeric_features": numeric, "categorical_features": categorical,
        "n_train": int(len(y_tr)), "n_holdout": int(len(y_ho)),
        "input_sha256": {"features": sha256_file(paths.features),
                         "row_ids": sha256_file(paths.row_ids)},
        "python": platform.python_version(),
    }
    paths.training_meta.write_text(json.dumps(meta, indent=2))
    print(f"[M5] preferred = {preferred}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="tiny grids (smoke runs only)")
    train_all(Paths.default(), fast=ap.parse_args().fast)


if __name__ == "__main__":
    main()

"""C1–C6 evaluation + governance artefacts.

    python -m src.evaluation.evaluate [--fast]

Reads model outputs (never re-selects models) and writes:
  artifacts/metrics.json            C1 + C6 (dated, hashed; the UI's ONLY KPI source)
  artifacts/subgroup_metrics.csv    C2 (traffic / weather / distance + traffic x weather, low_n flag)
  artifacts/feature_importance.csv  research question: which features explain duration
  artifacts/history/metrics_<ts>.json  C6 audit trail
  docs/model_card.md, docs/datasheet_lite.md  C3 / C4 / C5
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import INTENDED_USE_NOTE, ROW_ID, SEED, SUBGROUP_MIN_N, TARGET, Paths
from src.evaluation.metrics import score
from src.models.pipelines import candidate_by_name
from src.models.train import load_frames
from src.pipeline.acquire import sha256_file
from src.pipeline.engineer import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, SUBGROUP_COLUMNS,
                                   load_features)

CROSS = ("traffic_band", "weather_band")
SENSITIVITY_DROP = "courier_rating"


def holdout_frame(paths: Paths) -> pd.DataFrame:
    """y_true / y_pred joined to the band columns (the UI's D3/D4 join, too)."""
    preds = pd.read_csv(paths.y_pred_holdout)
    feats = load_features(paths)[[ROW_ID, TARGET, *SUBGROUP_COLUMNS]]
    df = preds.merge(feats, on=ROW_ID, how="left", validate="1:1")
    assert (df["y_true"] == df[TARGET]).all(), "y_pred_holdout out of sync with features"
    return df.drop(columns=TARGET)


def subgroup_table(df: pd.DataFrame, naive_pred: float, min_n: int = SUBGROUP_MIN_N) -> pd.DataFrame:
    rows = []
    groupings = [(c, [c]) for c in SUBGROUP_COLUMNS] + [(" x ".join(CROSS), list(CROSS))]
    for name, cols in groupings:
        for key, g in df.groupby(cols, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            s, nv = score(g["y_true"], g["y_pred"]), score(g["y_true"], np.full(len(g), naive_pred))
            rows.append({"subgroup": name, "level": " / ".join(map(str, key)), "n": len(g),
                         "mae": s["mae"], "rmse": s["rmse"],
                         "naive_mae": nv["mae"], "naive_rmse": nv["rmse"],
                         "mean_y_true": round(g["y_true"].mean(), 2),
                         "low_n": len(g) < min_n})
    return pd.DataFrame(rows)


def feature_importance(paths: Paths, fast: bool) -> pd.DataFrame:
    model = joblib.load(paths.models_dir / "preferred_model.joblib")
    _, _, X_ho, y_ho = load_frames(paths)
    X_ho = X_ho[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    r = permutation_importance(model, X_ho, y_ho, scoring="neg_mean_absolute_error",
                               n_repeats=2 if fast else 5, random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"feature": X_ho.columns,
                          "mae_increase": r.importances_mean.round(4),
                          "std": r.importances_std.round(4)})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def sensitivity_without(paths: Paths, meta: dict, drop: str, fast: bool) -> dict:
    """Refit the preferred spec without one ambiguous field (train only), score hold-out."""
    c = candidate_by_name(meta["preferred_model"], fast)
    num = [f for f in meta["numeric_features"] if f != drop]
    cat = [f for f in meta["categorical_features"] if f != drop]
    X_tr, y_tr, X_ho, y_ho = load_frames(paths)
    params = {f"model__{k}": v for k, v in meta["preferred_params"].items()}
    pipe = c.pipeline(num, cat).set_params(**params).fit(X_tr[num + cat], y_tr)
    return {"dropped_feature": drop, **score(y_ho, pipe.predict(X_ho[num + cat]))}


def evaluate(paths: Paths, fast: bool = False) -> dict:
    meta = json.loads(paths.training_meta.read_text())
    baseline = json.loads(paths.baseline_metrics.read_text())
    board = pd.read_csv(paths.leaderboard)
    df = holdout_frame(paths)

    # C1
    preferred = score(df["y_true"], df["y_pred"])
    naive = score(df["y_true"], np.full(len(df), baseline["train_mean"]))
    assert naive == baseline["holdout"], "naive baseline mismatch between M0 and C1"
    sel = board[board["selected"]].iloc[0]
    assert abs(sel["holdout_mae"] - preferred["mae"]) < 1e-3, "leaderboard != y_pred_holdout"

    # C2
    sub = subgroup_table(df, baseline["train_mean"])
    sub.to_csv(paths.subgroup_metrics, index=False)
    fi = feature_importance(paths, fast)
    fi.to_csv(paths.feature_importance, index=False)
    sens = sensitivity_without(paths, meta, SENSITIVITY_DROP, fast)

    # C6
    now = datetime.now(timezone.utc)
    hashed = {k: paths.root / v for k, v in {
        "raw_csv": "data/raw/delivery_raw.csv",
        "features_parquet": "data/features/delivery_features.parquet",
        "row_ids_json": "data/splits/row_ids.json",
        "preferred_model_joblib": "artifacts/models/preferred_model.joblib",
        "y_pred_holdout_csv": "artifacts/y_pred_holdout.csv",
        "leaderboard_csv": "artifacts/leaderboard.csv",
        "subgroup_metrics_csv": "artifacts/subgroup_metrics.csv"}.items()}
    metrics = {
        "generated_at": now.isoformat(timespec="seconds"),
        "seed": SEED,
        "split": "holdout",
        "n_train": meta["n_train"],
        "n_holdout": int(len(df)),
        "naive": naive,
        "naive_train_mean": baseline["train_mean"],
        "preferred_model": meta["preferred_model"],
        "preferred": preferred,
        "preferred_params": meta["preferred_params"],
        "selection_rule": meta["selection_rule"],
        "beats_naive_mae": preferred["mae"] < naive["mae"],
        "beats_naive_rmse": preferred["rmse"] < naive["rmse"],
        "improvement_vs_naive_pct": {
            k: round(100 * (naive[k] - preferred[k]) / naive[k], 2) for k in ("mae", "rmse")},
        "leaderboard": [{"model": r.model, "mae": r.holdout_mae, "rmse": r.holdout_rmse,
                         "cv_mae": r.cv_mae, "cv_rmse": r.cv_rmse, "selected": bool(r.selected)}
                        for r in board.itertuples()],
        "subgroup_min_n": SUBGROUP_MIN_N,
        "subgroup_low_n_cells": int(sub["low_n"].sum()),
        "top_features": fi.head(5)["feature"].tolist(),
        "sensitivity": sens,
        "fast_mode": bool(fast or meta.get("fast_mode")),
        "notes": INTENDED_USE_NOTE,
        "sha256": {k: sha256_file(p) for k, p in hashed.items() if p.exists()},
    }
    metrics["success_criteria_met"] = metrics["beats_naive_mae"] and metrics["beats_naive_rmse"]
    paths.metrics.write_text(json.dumps(metrics, indent=2))
    paths.metrics_history.mkdir(parents=True, exist_ok=True)
    (paths.metrics_history / f"metrics_{now:%Y%m%dT%H%M%SZ}.json").write_text(
        json.dumps(metrics, indent=2))

    from src.evaluation.docs import write_datasheet, write_model_card
    write_model_card(paths, metrics, sub, fi)
    write_datasheet(paths, metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    m = evaluate(Paths.default(), fast=ap.parse_args().fast)
    n, p = m["naive"], m["preferred"]
    print(f"[C1] naive    MAE={n['mae']:.3f}  RMSE={n['rmse']:.3f}")
    print(f"[C1] {m['preferred_model']:<9} MAE={p['mae']:.3f}  RMSE={p['rmse']:.3f}")
    print(f"[C1] success criteria (beats naive on BOTH MAE and RMSE): "
          f"{'MET' if m['success_criteria_met'] else 'NOT MET'}")
    print(f"[C2] low-n subgroup cells (n<{m['subgroup_min_n']}): {m['subgroup_low_n_cells']}")


if __name__ == "__main__":
    main()

import json
import shutil

import joblib
import pandas as pd

from src.config import ROW_ID, TARGET, Paths
from src.models.train import train_all
from src.pipeline.engineer import PREDICTORS


def test_model_artifacts_exist(trained_paths: Paths):
    for name in ["linear_model", "rf_model", "boost_model", "preferred_model"]:
        assert (trained_paths.models_dir / f"{name}.joblib").exists()
    for p in [trained_paths.baseline_metrics, trained_paths.leaderboard,
              trained_paths.y_pred_holdout, trained_paths.training_meta]:
        assert p.exists()


def test_holdout_predictions_align_with_split(trained_paths: Paths):
    preds = pd.read_csv(trained_paths.y_pred_holdout)
    split = json.loads(trained_paths.row_ids.read_text())
    assert list(preds.columns) == [ROW_ID, "y_true", "y_pred"]
    assert sorted(preds[ROW_ID]) == sorted(split["holdout"])
    feats = pd.read_parquet(trained_paths.features).set_index(ROW_ID)
    assert (feats.loc[preds[ROW_ID], TARGET].to_numpy() == preds["y_true"].to_numpy()).all()
    assert preds["y_pred"].notna().all()


def test_preferred_model_reloads_and_reproduces(trained_paths: Paths):
    model = joblib.load(trained_paths.models_dir / "preferred_model.joblib")
    preds = pd.read_csv(trained_paths.y_pred_holdout)
    feats = pd.read_parquet(trained_paths.features).set_index(ROW_ID)
    again = model.predict(feats.loc[preds[ROW_ID], PREDICTORS])
    assert abs(again - preds["y_pred"].to_numpy()).max() < 1e-3


def test_leaderboard_has_ladder_and_one_selection(trained_paths: Paths):
    board = pd.read_csv(trained_paths.leaderboard)
    assert {"M0", "M1", "M2", "M3"} <= set(board["step"])
    assert board["selected"].sum() == 1
    sel = board[board["selected"]].iloc[0]
    assert sel["family"] != "naive"
    # selection is by train-CV MAE, not hold-out
    assert sel["cv_mae"] == board[board["family"] != "naive"]["cv_mae"].min()


def test_no_holdout_peeping(trained_paths: Paths, tmp_path):
    """Scrambling hold-out targets must not change CV scores or the selected model."""
    copy = Paths(tmp_path)
    copy.ensure_dirs()
    shutil.copy(trained_paths.row_ids, copy.row_ids)
    feats = pd.read_parquet(trained_paths.features)
    hold = set(json.loads(trained_paths.row_ids.read_text())["holdout"])
    m = feats[ROW_ID].isin(hold)
    feats.loc[m, TARGET] = feats.loc[m, TARGET].sample(frac=1, random_state=0).to_numpy() * 3
    feats.to_parquet(copy.features, index=False)

    meta = train_all(copy, fast=True)
    a = pd.read_csv(trained_paths.leaderboard).set_index("model")
    b = pd.read_csv(copy.leaderboard).set_index("model")
    assert meta["preferred_model"] == a.index[a["selected"]][0]
    pd.testing.assert_series_equal(a["cv_mae"].sort_index(), b["cv_mae"].sort_index(),
                                   atol=1e-3)

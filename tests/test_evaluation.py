import json

import pandas as pd

from src.config import SUBGROUP_MIN_N, Paths
from src.evaluation.evaluate import subgroup_table

# Keys the Streamlit UI reads from metrics.json (D1 relies on these existing).
UI_KEYS = {"generated_at", "seed", "split", "naive", "preferred_model", "preferred",
           "beats_naive_mae", "beats_naive_rmse", "leaderboard", "subgroup_min_n", "notes",
           "success_criteria_met", "improvement_vs_naive_pct", "n_holdout"}


def test_metrics_json_has_ui_keys(evaluated_paths: Paths):
    m = json.loads(evaluated_paths.metrics.read_text())
    assert UI_KEYS <= set(m)
    for k in ("naive", "preferred"):
        assert set(m[k]) == {"mae", "rmse"}
    assert m["seed"] == 42 and m["split"] == "holdout"
    assert m["subgroup_min_n"] == SUBGROUP_MIN_N
    assert all({"model", "mae", "rmse"} <= set(r) for r in m["leaderboard"])
    assert m["beats_naive_mae"] == (m["preferred"]["mae"] < m["naive"]["mae"])
    assert m["fast_mode"] is True  # fixture runs must self-identify as non-results
    assert "preferred_model_joblib" in m["sha256"]


def test_metrics_consistent_with_predictions(evaluated_paths: Paths):
    m = json.loads(evaluated_paths.metrics.read_text())
    p = pd.read_csv(evaluated_paths.y_pred_holdout)
    assert abs((p.y_true - p.y_pred).abs().mean() - m["preferred"]["mae"]) < 1e-3
    assert m["n_holdout"] == len(p)


def test_subgroup_csv_and_low_n_flag(evaluated_paths: Paths):
    sub = pd.read_csv(evaluated_paths.subgroup_metrics)
    assert {"subgroup", "level", "n", "mae", "rmse", "low_n"} <= set(sub.columns)
    assert {"traffic_band", "weather_band", "distance_band"} <= set(sub["subgroup"])
    assert (sub["low_n"] == (sub["n"] < SUBGROUP_MIN_N)).all()


def test_subgroup_table_flags_small_cells():
    df = pd.DataFrame({"y_true": [10.0] * 60 + [20.0] * 5, "y_pred": [11.0] * 65,
                       "traffic_band": ["Low"] * 60 + ["Jam"] * 5,
                       "weather_band": ["Sunny"] * 65, "distance_band": ["0-5 km"] * 65})
    t = subgroup_table(df, naive_pred=15.0).set_index(["subgroup", "level"])
    assert not t.loc[("traffic_band", "Low"), "low_n"]
    assert t.loc[("traffic_band", "Jam"), "low_n"]
    assert t.loc[("traffic_band", "Low"), "mae"] == 1.0


def test_docs_written(evaluated_paths: Paths):
    card = evaluated_paths.model_card.read_text()
    assert "Intended use" in card and "not a production ETA" in card
    assert "fast/smoke mode" in card
    assert "Licence" in evaluated_paths.datasheet.read_text()
    assert list(evaluated_paths.metrics_history.glob("metrics_*.json"))

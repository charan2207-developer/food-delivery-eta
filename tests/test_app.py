"""Dashboard smoke test: renders all tabs against fixture artefacts without exceptions,
shows the limitations banner, and takes KPI numbers from metrics.json. The Predict tab
scores one row with the preferred model and never touches metrics.json."""
import json
import math
import re
import shutil

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT, Paths
from src.pipeline.engineer import PREDICTORS

APP = str(PROJECT_ROOT / "app/streamlit_app.py")


def _predicted_minutes(at: AppTest) -> float:
    msg = next(s.value for s in at.success if "Predicted duration" in s.value)
    return float(re.search(r"Predicted duration: ([-\d.]+) minutes", msg).group(1))


def test_app_renders_from_artifacts(evaluated_paths: Paths, monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(evaluated_paths.root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("not a production" in w.value for w in at.warning)  # D6 banner on landing
    m = json.loads(evaluated_paths.metrics.read_text())
    values = [x.value for x in at.metric]
    assert f"{m['preferred']['mae']:.2f}" in values
    assert f"{m['naive']['rmse']:.2f}" in values
    assert [t.label for t in at.tabs][-1] == "Predict"
    assert len(at.tabs) == 7


def test_app_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("run_all.sh" in e.value for e in at.error)


def test_predict_tab_returns_finite_duration(evaluated_paths: Paths, monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(evaluated_paths.root))
    before = evaluated_paths.metrics.read_bytes()
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="pred_go").click().run()
    assert not at.exception, at.exception
    assert math.isfinite(_predicted_minutes(at))
    assert any("not a production" in c.value for c in at.caption)
    assert evaluated_paths.metrics.read_bytes() == before  # interactive only


def test_predict_fill_from_holdout_is_labelled_illustration(evaluated_paths: Paths,
                                                            monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(evaluated_paths.root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="pred_fill").click().run()
    assert not at.exception, at.exception
    assert math.isfinite(_predicted_minutes(at))
    assert any("Illustration only, not evaluation" in i.value for i in at.info)


def test_predict_reports_missing_model(evaluated_paths: Paths, tmp_path, monkeypatch):
    root = tmp_path / "root"
    shutil.copytree(evaluated_paths.root, root)
    (Paths(root).models_dir / "preferred_model.joblib").unlink()
    monkeypatch.setenv("FDE_ROOT", str(root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("preferred_model.joblib" in e.value and "run_all.sh" in e.value for e in at.error)
    assert not any("Predicted duration" in s.value for s in at.success)


def test_preferred_pipeline_scores_one_row(trained_paths: Paths):
    model = joblib.load(trained_paths.models_dir / "preferred_model.joblib")
    feats = pd.read_parquet(trained_paths.features)
    row = feats[PREDICTORS].iloc[[0]].copy()
    for c in row.columns:  # the form sends plain floats / strings, not categoricals
        row[c] = row[c].astype(float if pd.api.types.is_numeric_dtype(row[c]) else str)
    assert math.isfinite(float(model.predict(row)[0]))

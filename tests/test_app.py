"""Dashboard smoke test: renders all tabs against fixture artefacts without exceptions,
shows the limitations banner, and takes KPI numbers from metrics.json."""
import json

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT, Paths

APP = str(PROJECT_ROOT / "app/streamlit_app.py")


def test_app_renders_from_artifacts(evaluated_paths: Paths, monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(evaluated_paths.root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("not a production" in w.value for w in at.warning)  # D6 banner on landing
    m = json.loads(evaluated_paths.metrics.read_text())
    values = [x.value for x in at.metric]
    assert f"{m['preferred']['mae']:.2f}" in values
    assert f"{m['naive']['rmse']:.2f}" in values
    assert len(at.tabs) == 6


def test_app_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("FDE_ROOT", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("run_all.sh" in e.value for e in at.error)

from __future__ import annotations

import pytest

from src.config import Paths
from src.pipeline.run import run as run_pipeline
from tests.fixtures import make_raw_fixture


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory) -> Paths:
    """A full S1–S6 run over the synthetic fixture in an isolated root."""
    root = tmp_path_factory.mktemp("fde")
    src = make_raw_fixture(root / "fixture_train.csv")
    paths = Paths(root)
    run_pipeline(paths, source=src)
    return paths


@pytest.fixture(scope="session")
def trained_paths(fixture_paths) -> Paths:
    from src.models.train import train_all
    train_all(fixture_paths, fast=True)
    return fixture_paths


@pytest.fixture(scope="session")
def evaluated_paths(trained_paths) -> Paths:
    from src.evaluation.evaluate import evaluate
    evaluate(trained_paths, fast=True)
    return trained_paths

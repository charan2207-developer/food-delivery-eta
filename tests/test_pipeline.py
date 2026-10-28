import json

import pandas as pd
import pytest

from src.config import ROW_ID, TARGET, Paths
from src.pipeline.engineer import FEATURE_SPEC, PREDICTORS, SUBGROUP_COLUMNS
from src.pipeline.freeze import SplitDriftError, freeze, make_split


def test_contract_files_exist(fixture_paths: Paths):
    for p in [fixture_paths.raw_csv, fixture_paths.interim_csv, fixture_paths.features,
              fixture_paths.row_ids, fixture_paths.profile_notes,
              fixture_paths.data_dictionary, fixture_paths.transform_log]:
        assert p.exists(), p


def test_features_schema_and_no_leaky_columns(fixture_paths: Paths):
    f = pd.read_parquet(fixture_paths.features)
    assert list(f.columns) == list(FEATURE_SPEC)
    assert TARGET not in PREDICTORS and ROW_ID not in PREDICTORS
    forbidden = {"Delivery_person_ID", "ID", "restaurant_lat", "restaurant_lon",
                 "delivery_lat", "delivery_lon", "order_id"}
    assert forbidden.isdisjoint(f.columns)
    assert f[ROW_ID].is_unique
    assert f[TARGET].between(10, 60).all()
    for col in SUBGROUP_COLUMNS:
        assert f[col].notna().all()


def test_cleaning_rules(fixture_paths: Paths):
    f = pd.read_parquet(fixture_paths.features)
    assert (f["distance_km"] >= 0).all()
    assert f["courier_rating"].dropna().between(1, 5).all()  # fixture injects "6"
    assert (f["traffic_band"] == "Unknown").sum() == 10
    assert (f["weather_band"] == "Unknown").sum() == 5
    assert f["prep_time_min"].isna().sum() == 10
    assert f["coords_placeholder"].sum() == 5


def test_split_disjoint_complete(fixture_paths: Paths):
    split = json.loads(fixture_paths.row_ids.read_text())
    assert set(split) == {"train", "holdout"}
    tr, ho = set(split["train"]), set(split["holdout"])
    ids = set(pd.read_parquet(fixture_paths.features)[ROW_ID])
    assert tr.isdisjoint(ho) and tr | ho == ids
    assert abs(len(ho) / len(ids) - 0.2) < 0.01


def test_freeze_deterministic(fixture_paths: Paths):
    ids = pd.read_parquet(fixture_paths.features)[ROW_ID]
    assert make_split(ids) == make_split(ids)
    assert make_split(ids, seed=1) != make_split(ids)
    assert freeze(fixture_paths) == json.loads(fixture_paths.row_ids.read_text())


def test_freeze_refuses_silent_drift(fixture_paths: Paths, tmp_path):
    copy = Paths(tmp_path)
    copy.ensure_dirs()
    copy.features.write_bytes(fixture_paths.features.read_bytes())
    copy.row_ids.write_text(json.dumps({"train": [0], "holdout": [1]}))
    with pytest.raises(SplitDriftError):
        freeze(copy)
    freeze(copy, refreeze=True)
    assert json.loads(copy.row_ids.read_text()) == json.loads(fixture_paths.row_ids.read_text())

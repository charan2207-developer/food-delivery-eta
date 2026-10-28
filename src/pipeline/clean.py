"""S3 Clean: fix encodings and invalid values. No imputation here.

Imputation is deferred to the model pipelines (fit on the training split only),
so no hold-out statistics leak into training. Every rule is counted and written
to ``clean_stats.json`` which S5 turns into docs/transform_log.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ROW_ID, TARGET, Paths
from src.pipeline.profile import RAW_TARGET, read_raw

# Raw columns dropped at clean time and why (also rendered into the transform log).
DROPPED_COLUMNS = {
    "ID": "Order identifier; carries no predictive meaning. Replaced by positional row_id.",
    "Delivery_person_ID": "Pseudonymous courier identifier. Excluded for privacy (no "
                          "per-person profiling / re-identification) and to avoid memorisation.",
}

RENAME = {
    "Delivery_person_Age": "courier_age",
    "Delivery_person_Ratings": "courier_rating",
    "Restaurant_latitude": "restaurant_lat",
    "Restaurant_longitude": "restaurant_lon",
    "Delivery_location_latitude": "delivery_lat",
    "Delivery_location_longitude": "delivery_lon",
    "Order_Date": "order_date",
    "Time_Orderd": "time_ordered",
    "Time_Order_picked": "time_picked",
    "Weatherconditions": "weather",
    "Road_traffic_density": "traffic",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle_type",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city_type",
    RAW_TARGET: TARGET,
}

NUMERIC = ["courier_age", "courier_rating", "restaurant_lat", "restaurant_lon",
           "delivery_lat", "delivery_lon", "vehicle_condition", "multiple_deliveries"]
CATEGORICAL = ["weather", "traffic", "order_type", "vehicle_type", "festival", "city_type"]
RATING_RANGE = (1.0, 5.0)


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"rows_in": int(len(raw))}
    df = raw.apply(lambda s: s.str.strip())
    df.insert(0, ROW_ID, np.arange(len(df), dtype=np.int64))
    df = df.drop(columns=list(DROPPED_COLUMNS)).rename(columns=RENAME)

    stats["nan_literal_to_null"] = {}
    for col in df.columns:
        if col == ROW_ID:
            continue
        mask = df[col].isin(["NaN", ""])
        stats["nan_literal_to_null"][col] = int(mask.sum())
        df.loc[mask, col] = None

    # Labels: "conditions Sunny" -> "Sunny"; "(min) 24" -> 24
    df["weather"] = df["weather"].str.replace("conditions ", "", regex=False)
    weather_nan = df["weather"].eq("NaN")
    stats["nan_literal_to_null"]["weather"] += int(weather_nan.sum())
    df.loc[weather_nan, "weather"] = None
    df[TARGET] = df[TARGET].str.extract(r"(\d+)")[0].astype(float)
    stats["target_unparseable"] = int(df[TARGET].isna().sum())
    df = df[df[TARGET].notna()].copy()

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coordinates: sign errors (all orders are in India, so lat/lon must be positive).
    coord_cols = ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon"]
    stats["negative_coords_abs"] = {c: int((df[c] < 0).sum()) for c in coord_cols}
    df[coord_cols] = df[coord_cols].abs()
    df["coords_placeholder"] = (df["restaurant_lat"] < 1) | (df["restaurant_lon"] < 1)
    stats["coords_placeholder_rows"] = int(df["coords_placeholder"].sum())

    lo, hi = RATING_RANGE
    bad_rating = df["courier_rating"].notna() & ~df["courier_rating"].between(lo, hi)
    stats["rating_out_of_range_to_null"] = int(bad_rating.sum())
    df.loc[bad_rating, "courier_rating"] = np.nan

    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce")
    stats["order_date_unparseable"] = int(df["order_date"].isna().sum())
    for col in ["time_ordered", "time_picked"]:
        parsed = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
        stats[f"{col}_unparseable"] = int((parsed.isna() & df[col].notna()).sum())
        df[col] = parsed.dt.strftime("%H:%M:%S")

    stats["duplicates_dropped"] = int(df.drop(columns=[ROW_ID]).duplicated().sum())
    df = df[~df.drop(columns=[ROW_ID]).duplicated()]
    stats["target_range"] = [float(df[TARGET].min()), float(df[TARGET].max())]
    stats["rows_out"] = int(len(df))
    stats["dropped_columns"] = DROPPED_COLUMNS
    return df.reset_index(drop=True), stats


def clean(paths: Paths) -> pd.DataFrame:
    df, stats = clean_frame(read_raw(paths))
    paths.interim_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.interim_csv, index=False, date_format="%Y-%m-%d")
    paths.clean_stats.write_text(json.dumps(stats, indent=2))
    return df


def read_clean(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.interim_csv, parse_dates=["order_date"])


def main() -> None:
    df = clean(Paths.default())
    print(f"[S3] {len(df)} clean rows -> data/interim/delivery_clean.csv")


if __name__ == "__main__":
    main()

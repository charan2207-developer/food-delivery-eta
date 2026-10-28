"""S4 Engineer: derived predictors + distance / traffic / weather bands.

``FEATURE_SPEC`` is the single source of truth for which columns are model
predictors; the model layer and the data dictionary both read it.
Raw coordinates are used only to compute distance and are NOT written to the
feature table (aggregate-first / no geo dump).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ROW_ID, TARGET, Paths
from src.pipeline.clean import read_clean

UNKNOWN = "Unknown"
DISTANCE_BINS = [0, 5, 10, 15, np.inf]
DISTANCE_LABELS = ["0-5 km", "5-10 km", "10-15 km", "15+ km"]
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam", UNKNOWN]
WEATHER_ORDER = ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms", UNKNOWN]

# name -> (role, description, source)
FEATURE_SPEC: dict[str, tuple[str, str, str]] = {
    ROW_ID: ("id", "Positional row index in the raw file; join key for splits/predictions. "
                   "Not a predictor.", "derived (S3)"),
    TARGET: ("target", "Delivery duration in minutes.", "Time_taken(min)"),
    "courier_age": ("numeric", "Courier age in years (NaN imputed in model pipeline).",
                    "Delivery_person_Age"),
    "courier_rating": ("numeric", "Courier profile rating 1–5; >5 set NaN in S3. See leakage "
                       "note in transform log.", "Delivery_person_Ratings"),
    "vehicle_condition": ("numeric", "Vehicle condition score 0–3.", "Vehicle_condition"),
    "multiple_deliveries": ("numeric", "Number of orders carried together (0–3).",
                            "multiple_deliveries"),
    "distance_km": ("numeric", "Great-circle (haversine) restaurant→delivery distance, km.",
                    "lat/lon (S4)"),
    "coords_placeholder": ("numeric", "1 if restaurant coordinates were ≈0 placeholders "
                           "(distance less reliable).", "derived (S3)"),
    "prep_time_min": ("numeric", "Minutes between order and pickup (midnight wrap handled); "
                      "NaN where order time missing.", "Time_Orderd, Time_Order_picked"),
    "pickup_hour": ("numeric", "Hour of day of pickup (0–23).", "Time_Order_picked"),
    "day_of_week": ("numeric", "Day of week of order, Monday=0.", "Order_Date"),
    "is_weekend": ("numeric", "1 if Saturday/Sunday.", "Order_Date"),
    "weather_band": ("categorical", "Weather condition band; missing -> Unknown.",
                     "Weatherconditions"),
    "traffic_band": ("categorical", "Road traffic density band Low/Medium/High/Jam; "
                     "missing -> Unknown.", "Road_traffic_density"),
    "distance_band": ("categorical", "Distance band: " + ", ".join(DISTANCE_LABELS) + ".",
                      "distance_km"),
    "order_type": ("categorical", "Snack / Meal / Drinks / Buffet.", "Type_of_order"),
    "vehicle_type": ("categorical", "motorcycle / scooter / electric_scooter / bicycle.",
                     "Type_of_vehicle"),
    "festival": ("categorical", "Festival day Yes/No/Unknown.", "Festival"),
    "city_type": ("categorical", "Metropolitian / Urban / Semi-Urban / Unknown "
                  "(source spelling kept).", "City"),
}

NUMERIC_FEATURES = [k for k, v in FEATURE_SPEC.items() if v[0] == "numeric"]
CATEGORICAL_FEATURES = [k for k, v in FEATURE_SPEC.items() if v[0] == "categorical"]
PREDICTORS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SUBGROUP_COLUMNS = ["traffic_band", "weather_band", "distance_band"]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _minutes(hhmmss: pd.Series) -> pd.Series:
    t = pd.to_datetime(hhmmss, format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute + t.dt.second / 60


def engineer_frame(clean: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({ROW_ID: clean[ROW_ID].astype("int64")})
    out[TARGET] = clean[TARGET].astype(float)
    for col in ["courier_age", "courier_rating", "vehicle_condition", "multiple_deliveries"]:
        out[col] = clean[col].astype(float)

    out["distance_km"] = haversine_km(clean["restaurant_lat"], clean["restaurant_lon"],
                                      clean["delivery_lat"], clean["delivery_lon"]).round(3)
    out["coords_placeholder"] = clean["coords_placeholder"].astype(bool).astype(int)

    prep = _minutes(clean["time_picked"]) - _minutes(clean["time_ordered"])
    out["prep_time_min"] = prep.where(prep >= 0, prep + 24 * 60)
    out["pickup_hour"] = pd.to_datetime(clean["time_picked"], format="%H:%M:%S",
                                        errors="coerce").dt.hour.astype(float)
    dow = pd.to_datetime(clean["order_date"]).dt.dayofweek
    out["day_of_week"] = dow.astype(float)
    out["is_weekend"] = (dow >= 5).astype(int)

    out["weather_band"] = pd.Categorical(clean["weather"].fillna(UNKNOWN), WEATHER_ORDER)
    out["traffic_band"] = pd.Categorical(clean["traffic"].fillna(UNKNOWN), TRAFFIC_ORDER)
    out["distance_band"] = pd.cut(out["distance_km"], DISTANCE_BINS, labels=DISTANCE_LABELS,
                                  right=False)
    for col in ["order_type", "vehicle_type", "festival", "city_type"]:
        out[col] = clean[col].fillna(UNKNOWN).astype(str)

    assert list(out.columns) == list(FEATURE_SPEC), "FEATURE_SPEC out of sync with engineer"
    return out


def engineer(paths: Paths) -> pd.DataFrame:
    feats = engineer_frame(read_clean(paths))
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(paths.features, index=False)
    return feats


def load_features(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.features)


def main() -> None:
    f = engineer(Paths.default())
    print(f"[S4] {f.shape} -> data/features/delivery_features.parquet")


if __name__ == "__main__":
    main()

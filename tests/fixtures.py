"""Synthetic smoke fixture in the raw Kaggle format (CI/tests only).

Values are random and NOT Malik (2022) data; metrics computed on this fixture
must never be reported as real results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEATHER = ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]


def make_raw_fixture(path: Path, n: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(10, 30, n)
    lon = rng.uniform(70, 88, n)
    off = rng.choice(np.arange(1, 14) / 100, n)
    traffic = rng.choice(TRAFFIC, n)
    weather = rng.choice(WEATHER, n)
    ordered_min = rng.integers(8 * 60, 23 * 60, n)
    prep = rng.choice([5, 10, 15], n)
    dist_km = off * 150
    y = (12 + 0.9 * dist_km + 4 * np.vectorize(TRAFFIC.index)(traffic)
         + 3 * np.isin(weather, ["Stormy", "Sandstorms", "Fog"]) + rng.normal(0, 3, n))
    y = np.clip(np.round(y), 10, 54).astype(int)

    def hhmm(m):
        m = m % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}:00"

    df = pd.DataFrame({
        "ID": [f"0x{i:04x} " for i in range(n)],
        "Delivery_person_ID": [f"CITYRES{i % 20:02d}DEL0{i % 3} " for i in range(n)],
        "Delivery_person_Age": rng.integers(18, 40, n).astype(str),
        "Delivery_person_Ratings": rng.choice(["4.5", "4.7", "4.9", "5", "6"], n),
        "Restaurant_latitude": lat.round(6),
        "Restaurant_longitude": lon.round(6),
        "Delivery_location_latitude": (lat + off).round(6),
        "Delivery_location_longitude": (lon + off).round(6),
        "Order_Date": pd.to_datetime("2022-03-01")
                      + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
        "Time_Orderd": [hhmm(m) for m in ordered_min],
        "Time_Order_picked": [hhmm(m + p) for m, p in zip(ordered_min, prep)],
        "Weatherconditions": ["conditions " + w for w in weather],
        "Road_traffic_density": [t + " " for t in traffic],
        "Vehicle_condition": rng.integers(0, 4, n),
        "Type_of_order": rng.choice(["Snack ", "Meal ", "Drinks ", "Buffet "], n),
        "Type_of_vehicle": rng.choice(["motorcycle ", "scooter ", "electric_scooter "], n),
        "multiple_deliveries": rng.integers(0, 3, n).astype(str),
        "Festival": rng.choice(["No ", "Yes "], n, p=[0.95, 0.05]),
        "City": rng.choice(["Metropolitian ", "Urban ", "Semi-Urban "], n),
        "Time_taken(min)": [f"(min) {v}" for v in y],
    })
    df["Order_Date"] = df["Order_Date"].dt.strftime("%d-%m-%Y")
    # Inject the same quirks the real file has.
    q = rng.choice(n, 40, replace=False)
    df.loc[q[:10], "Time_Orderd"] = "NaN "
    df.loc[q[10:20], "Road_traffic_density"] = "NaN "
    df.loc[q[20:25], "Weatherconditions"] = "conditions NaN"
    df.loc[q[25:30], "Restaurant_latitude"] *= -1
    df.loc[q[30:35], ["Restaurant_latitude", "Restaurant_longitude"]] = 0.0
    df.loc[q[30:35], "Delivery_location_latitude"] = off[q[30:35]]
    df.loc[q[30:35], "Delivery_location_longitude"] = off[q[30:35]]
    df.loc[q[35:40], "Delivery_person_Age"] = "NaN "
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

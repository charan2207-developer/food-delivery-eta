"""S2 Profile: nulls (incl. literal "NaN" strings), dtypes, cardinality, target summary."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.config import Paths

RAW_TARGET = "Time_taken(min)"


def read_raw(paths: Paths) -> pd.DataFrame:
    # Keep everything as string so the profile sees the raw encoding (e.g. "NaN ", "(min) 24").
    return pd.read_csv(paths.raw_csv, dtype=str, keep_default_na=False)


def profile(paths: Paths) -> pd.DataFrame:
    raw = read_raw(paths)
    stripped = raw.apply(lambda s: s.str.strip())
    rows = []
    for col in raw.columns:
        s = stripped[col]
        missing = s.isin(["", "NaN", "conditions NaN"]).sum()
        numeric = pd.to_numeric(s, errors="coerce")
        rows.append({
            "column": col,
            "missing_like": int(missing),
            "missing_pct": round(100 * missing / len(s), 2),
            "n_unique": int(s.nunique()),
            "numeric_parse_pct": round(100 * numeric.notna().mean(), 1),
            "example": s.iloc[0],
        })
    table = pd.DataFrame(rows)

    y = stripped[RAW_TARGET].str.extract(r"(\d+)")[0].astype(float)
    lat_neg = (pd.to_numeric(stripped["Restaurant_latitude"]) < 0).sum()
    lon_neg = (pd.to_numeric(stripped["Restaurant_longitude"]) < 0).sum()
    zero_coords = (pd.to_numeric(stripped["Restaurant_latitude"]).abs() < 1).sum()
    ratings = pd.to_numeric(stripped["Delivery_person_Ratings"], errors="coerce")

    lines = [
        "# S2 Profile notes",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
        "`data/raw/delivery_raw.csv`._",
        "",
        f"- Rows: **{len(raw)}**, columns: **{raw.shape[1]}**",
        "- All columns are read as strings first: missing values are encoded as the literal "
        "`NaN` (often with trailing spaces), and the target is encoded as `(min) NN`.",
        "",
        "## Column profile",
        "",
        table.to_markdown(index=False),
        "",
        "## Target: delivery duration (minutes)",
        "",
        y.describe().round(2).to_frame("time_taken_min").to_markdown(),
        "",
        "## Data-quality observations (drive S3 rules)",
        "",
        f"- Restaurant latitude < 0 in {lat_neg} rows, longitude < 0 in {lon_neg} rows "
        "(sign errors; all orders are in India).",
        f"- Restaurant coordinates ≈ 0 in {zero_coords} rows (masked/placeholder locations).",
        f"- Courier ratings > 5 in {(ratings > 5).sum()} rows (outside the 1–5 scale).",
        "- `Time_Orderd` missing in "
        f"{(stripped['Time_Orderd'] == 'NaN').sum()} rows → preparation time not computable there.",
        "",
    ]
    paths.profile_notes.parent.mkdir(parents=True, exist_ok=True)
    paths.profile_notes.write_text("\n".join(lines))
    return table


def main() -> None:
    profile(Paths.default())
    print("[S2] wrote reports/eda/profile_notes.md")


if __name__ == "__main__":
    main()

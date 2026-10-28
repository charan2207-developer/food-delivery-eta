"""S5 Document: data dictionary + transform log (incl. leakage checklist) from real run stats."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import Paths
from src.pipeline.engineer import FEATURE_SPEC, load_features

LEAKAGE_CHECKLIST = [
    ("Target `Time_taken(min)`", "Excluded from predictors (it is the label)."),
    ("Prediction point", "Defined as **at pickup**: everything used is known once the courier "
     "has collected the order. Nothing observed during or after the trip is used."),
    ("`Time_Order_picked` → `prep_time_min`, `pickup_hour`", "Known at pickup, so allowed. "
     "If the estimate were needed at *order* time, `prep_time_min` would have to be dropped."),
    ("`Delivery_person_Ratings` → `courier_rating`", "**Ambiguous.** Treated as a courier profile "
     "attribute known before the trip, but the source does not say whether it includes the "
     "rating of this delivery. Kept, with a sensitivity check (model retrained without it) "
     "reported in the model card."),
    ("`Delivery_person_ID`", "Dropped (pseudonymous identifier; privacy + memorisation risk)."),
    ("`ID`", "Dropped (order identifier)."),
    ("Raw lat/lon", "Used only to compute `distance_km`; not stored in the feature table."),
    ("Imputation / scaling / encoding", "Fit inside model pipelines on the **training split "
     "only** (no global statistics computed over the hold-out)."),
    ("Hyper-parameter search & model selection", "K-fold CV on the training split only. The "
     "hold-out is scored once per model, after selection, for reporting."),
]


def write_data_dictionary(paths: Paths, feats: pd.DataFrame) -> None:
    rows = []
    for name, (role, desc, source) in FEATURE_SPEC.items():
        s = feats[name]
        if role == "categorical":
            values = ", ".join(map(str, s.value_counts(sort=False).index))
        elif role in ("numeric", "target"):
            values = f"{s.min():.2f} – {s.max():.2f}"
        else:
            values = f"{s.min()} – {s.max()}"
        rows.append({"column": f"`{name}`", "role": role, "dtype": str(s.dtype),
                     "missing": int(s.isna().sum()), "range / levels": values,
                     "source": source, "description": desc})
    md = [
        "# Data dictionary — `data/features/delivery_features.parquet`",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`src/pipeline/document.py` (S5). Rows: **{:,}**._".format(len(feats)),
        "",
        "Roles: `id` = join key only, `target` = label, `numeric` / `categorical` = model "
        "predictors. Bands used for subgroup evaluation: `traffic_band`, `weather_band`, "
        "`distance_band`.",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "Not present by design: courier ID, order ID, raw coordinates (see transform log).",
        "",
    ]
    paths.data_dictionary.parent.mkdir(parents=True, exist_ok=True)
    paths.data_dictionary.write_text("\n".join(md))


def write_transform_log(paths: Paths, feats: pd.DataFrame) -> None:
    st = json.loads(paths.clean_stats.read_text())
    nulls = {k: v for k, v in st["nan_literal_to_null"].items() if v}
    neg = st["negative_coords_abs"]
    md = [
        "# Transform log (S1–S6)",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
        "`data/interim/clean_stats.json`. Counts are from the actual run._",
        "",
        "| Stage | Rule | Rows affected |",
        "|---|---|---|",
        f"| S1 | Use Kaggle `train.csv` only (test.csv has no target) | {st['rows_in']:,} in |",
        "| S3 | Strip whitespace from all fields | all |",
        f"| S3 | Drop columns: {', '.join('`'+c+'`' for c in st['dropped_columns'])} | all |",
        "| S3 | Literal `NaN` strings → null (per column) | "
        + ", ".join(f"{k}: {v}" for k, v in nulls.items()) + " |",
        "| S3 | `conditions X` → `X` (weather); `(min) NN` → NN (target) | all |",
        f"| S3 | Rows with unparseable target dropped | {st['target_unparseable']} |",
        "| S3 | Negative coordinates → absolute value (sign errors; all orders in India) | "
        + ", ".join(f"{k}: {v}" for k, v in neg.items()) + " |",
        f"| S3 | Flag placeholder (≈0) restaurant coordinates as `coords_placeholder` | "
        f"{st['coords_placeholder_rows']} |",
        f"| S3 | Courier rating outside 1–5 → null | {st['rating_out_of_range_to_null']} |",
        f"| S3 | Exact duplicate rows dropped | {st['duplicates_dropped']} |",
        "| S3 | **No imputation** (deferred to train-only model pipelines) | — |",
        "| S4 | `distance_km` = haversine(restaurant, delivery); raw coords then discarded | all |",
        "| S4 | `prep_time_min` = pickup − order time (+24h if negative) | "
        f"{int(feats['prep_time_min'].isna().sum())} null (order time missing) |",
        "| S4 | Missing categoricals → `Unknown` level | see dictionary |",
        "| S4 | Bands: distance (0-5/5-10/10-15/15+ km), traffic, weather | all |",
        "| S6 | Seeded 80/20 train/hold-out split on `row_id` | see `data/splits/` |",
        "",
        f"Rows out of S3/S4: **{st['rows_out']:,}**. Target range: "
        f"{st['target_range'][0]:.0f}–{st['target_range'][1]:.0f} min (no clipping needed).",
        "",
        "## Leakage checklist",
        "",
        "| Field / step | Decision |",
        "|---|---|",
        *[f"| {a} | {b} |" for a, b in LEAKAGE_CHECKLIST],
        "",
    ]
    paths.transform_log.write_text("\n".join(md))


def document(paths: Paths) -> None:
    feats = load_features(paths)
    write_data_dictionary(paths, feats)
    write_transform_log(paths, feats)


def main() -> None:
    document(Paths.default())
    print("[S5] wrote docs/data_dictionary.md, docs/transform_log.md")


if __name__ == "__main__":
    main()

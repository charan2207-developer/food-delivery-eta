"""C3/C4/C5: model card + datasheet-lite, rendered from the real evaluation outputs."""
from __future__ import annotations

import json

import pandas as pd

from src.config import CITATION, KAGGLE_URL, Paths

LIMITATIONS = [
    "**Offline ops-proxy demo, not a production ETA.** Not validated for customer-facing "
    "promises, courier dispatch, pay, or performance management.",
    "**Single public dataset, short window.** ~45k orders spanning roughly six weeks in "
    "Feb–Apr 2022 across Indian cities. No evidence it transfers to other cities, seasons, "
    "platforms or years.",
    "**Provenance not verifiable.** The source does not document how records were collected; "
    "they may be partly simulated. Treat findings as properties of this dataset, not of "
    "real-world delivery operations.",
    "**Random (not temporal) hold-out.** Hold-out rows share dates with training rows, so the "
    "score is optimistic relative to forecasting future weeks.",
    "**No live GPS, routing, restaurant load or real-time traffic feeds.** Distance is "
    "straight-line (haversine), not road distance; placeholder (≈0) coordinates exist.",
    "**Prediction point is at pickup.** Uses pickup time and preparation time; an estimate "
    "at order-placement time would need those removed and would likely be less accurate.",
    "**Courier rating is ambiguous** (may include post-delivery feedback). See sensitivity check.",
    "**Subgroup cells with n < {min_n} are flagged** and should not be used for conclusions.",
    "Errors are in whole minutes on a 10–54 minute target; individual predictions can be off "
    "by much more than the average error.",
]


def _fmt_board(metrics: dict) -> str:
    rows = [{"model": r["model"] + (" ✅" if r["selected"] else ""),
             "train-CV MAE": r["cv_mae"], "train-CV RMSE": r["cv_rmse"],
             "hold-out MAE": r["mae"], "hold-out RMSE": r["rmse"]} for r in metrics["leaderboard"]]
    return pd.DataFrame(rows).to_markdown(index=False)


def write_model_card(paths: Paths, m: dict, sub: pd.DataFrame, fi: pd.DataFrame) -> None:
    n, p = m["naive"], m["preferred"]
    verdict = ("**MET** — the preferred model improves both MAE and RMSE over the naive baseline "
               "on the hold-out." if m["success_criteria_met"] else
               "**NOT MET** — the preferred model does not improve both MAE and RMSE over the "
               "naive baseline on the hold-out. Reported as-is.")
    single = sub[sub["subgroup"] != "traffic_band x weather_band"]
    ok = single[~single["low_n"]].sort_values("mae")
    if ok.empty:  # tiny datasets: every cell is low-n, so say so rather than fail
        ok = single.sort_values("mae").assign(level=lambda d: d["level"] + " (low n)")
    low = sub[sub["low_n"]]
    sens = m["sensitivity"]
    fast_warn = ("\n> ⚠️ **Generated in fast/smoke mode — numbers are NOT valid results.**\n"
                 if m.get("fast_mode") else "")
    md = f"""# Model card — food-delivery duration regressor
{fast_warn}
_Generated {m['generated_at']} by `src/evaluation/evaluate.py` from `artifacts/metrics.json`._

## Model details
- **Preferred model:** `{m['preferred_model']}` (sklearn Pipeline: train-only median imputation
  + missing indicators, one-hot categoricals, then the regressor).
- **Hyper-parameters:** `{json.dumps(m['preferred_params'])}`
- **Selection rule:** {m['selection_rule']}.
- **Seed:** {m['seed']}. Train rows: {m['n_train']:,}. Hold-out rows: {m['n_holdout']:,}.
- Artefact: `artifacts/models/preferred_model.joblib` (sha256 `{m['sha256'].get('preferred_model_joblib', 'n/a')[:16]}…`).

## Intended use (C3)
- **Users:** analytics / operations reviewers exploring what drives delivery duration in a
  public dataset, and how well tabular regression can estimate it.
- **Use:** offline decision-support and teaching demo (ops-proxy).
- **Out of scope:** live customer-facing ETAs, courier routing/GPS, dispatch, pay or
  individual courier assessment, any use on personal data.

## Metrics (C1) — hold-out, minutes
| | MAE | RMSE |
|---|---|---|
| Naive (train mean = {m['naive_train_mean']:.2f} min) | {n['mae']:.3f} | {n['rmse']:.3f} |
| Preferred `{m['preferred_model']}` | {p['mae']:.3f} | {p['rmse']:.3f} |
| Improvement vs naive | {m['improvement_vs_naive_pct']['mae']:.1f}% | {m['improvement_vs_naive_pct']['rmse']:.1f}% |

**Success criterion:** {verdict}

### Leaderboard (selection used train-CV only; hold-out shown for reporting)
{_fmt_board(m)}

## Subgroup errors (C2)
Full table: `artifacts/subgroup_metrics.csv` (traffic, weather, distance bands and the
traffic × weather cross). Cells with n < {m['subgroup_min_n']} are flagged `low_n`.

- Lowest-error band: **{ok.iloc[0]['subgroup']} = {ok.iloc[0]['level']}** (MAE {ok.iloc[0]['mae']:.2f}, n={ok.iloc[0]['n']:,})
- Highest-error band: **{ok.iloc[-1]['subgroup']} = {ok.iloc[-1]['level']}** (MAE {ok.iloc[-1]['mae']:.2f}, n={ok.iloc[-1]['n']:,})
- Low-n cells flagged: **{len(low)}**{(' — ' + ', '.join(f"{r.level} (n={r.n})" for r in low.itertuples())) if len(low) else ''}

## What explains duration (research question)
Permutation importance on the hold-out (increase in MAE, minutes, when a feature is shuffled):

{fi.head(8).to_markdown(index=False)}

Importance is associative within this dataset, not causal.

## Sensitivity check — ambiguous field
Refit `{m['preferred_model']}` with the same hyper-parameters **without `{sens['dropped_feature']}`**
(train only), scored on the same hold-out: MAE **{sens['mae']:.3f}**, RMSE **{sens['rmse']:.3f}**
(vs {p['mae']:.3f} / {p['rmse']:.3f} with it; naive {n['mae']:.3f} / {n['rmse']:.3f}).

## Limitations
""" + "\n".join(f"- {x.format(min_n=m['subgroup_min_n'])}" for x in LIMITATIONS) + f"""

## Ethical / privacy considerations (C5)
- Public data only; no new data collected. Courier and order IDs and raw coordinates are
  excluded from the feature table and from the dashboard. No re-identification attempted.
- Dashboard is aggregate-first; the case table shows a small sample without IDs or locations.
- The model must not be used to rate or rank individual couriers.

## Reproducibility (C6)
`scripts/run_all.sh` regenerates everything from the raw file. Input/output SHA-256 hashes are in
`artifacts/metrics.json`; every run also writes a dated copy to `artifacts/history/`.

Data: {CITATION}
"""
    paths.model_card.parent.mkdir(parents=True, exist_ok=True)
    paths.model_card.write_text(md)


def write_datasheet(paths: Paths, m: dict) -> None:
    acq = json.loads(paths.raw_meta.read_text()) if paths.raw_meta.exists() else {}
    st = json.loads(paths.clean_stats.read_text())
    md = f"""# Datasheet-lite — Food delivery dataset (Malik, 2022)

_Generated {m['generated_at']}. Pipeline facts come from `data/raw/acquisition.json` and
`data/interim/clean_stats.json`._

## Source & licence (C4)
- **Citation:** {CITATION}
- **File used:** `train.csv` only ({acq.get('n_rows', st['rows_in']):,} rows; Kaggle `test.csv`
  has no target and is not used). SHA-256 `{acq.get('sha256', 'n/a')}`.
- **Licence:** governed by the terms shown on the [Kaggle dataset page]({KAGGLE_URL}).
  Check them before redistributing. This repo **does not redistribute** the data;
  `data/` is git-ignored and is regenerated locally from Kaggle.

## Motivation
Public benchmark for food-delivery duration regression; used here for an offline ops-proxy demo.

## Composition
- One row per delivered order: courier age/rating (pseudonymous courier ID), restaurant &
  delivery coordinates, order date, order/pickup times, weather, traffic density, vehicle
  type/condition, order type, multiple deliveries, festival flag, city type, duration (min).
- Target range {st['target_range'][0]:.0f}–{st['target_range'][1]:.0f} minutes.
- Missingness is encoded as literal `NaN` strings; see `docs/transform_log.md` for counts.

## Collection process
Not documented in detail by the source. Whether records are real, anonymised, or simulated
cannot be verified. Results should be read as properties of this dataset.

## Preprocessing & exclusions
- Dropped: {', '.join('`'+k+'`' for k in st['dropped_columns'])} (identifiers / privacy).
- Raw coordinates used only to derive `distance_km`, then excluded.
- Sign-error coordinates corrected; out-of-range ratings nulled ({st['rating_out_of_range_to_null']});
  placeholder coordinates flagged ({st['coords_placeholder_rows']}).
- Full rule list and leakage checklist: `docs/transform_log.md`; fields: `docs/data_dictionary.md`.

## Privacy & security (C5)
- No new personal data collected; no scraping; no re-identification attempts.
- Courier IDs are pseudonymous in the source and are still removed before modelling.
- UI is aggregate-first and never shows IDs or coordinates.
- Kaggle credentials stay in the user's home directory and are never committed.

## Uses
- **Appropriate:** teaching, offline analytics, method comparison.
- **Inappropriate:** live ETAs, courier surveillance or assessment, claims about real
  operations of any named company.

## Maintenance
Snapshot as downloaded on {acq.get('acquired_at', 'n/a')}. If the upstream file changes,
the SHA-256 changes; re-run `scripts/run_all.sh --refreeze` and republish metrics.
"""
    paths.datasheet.write_text(md)

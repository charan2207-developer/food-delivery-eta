# Model card — food-delivery duration regressor

_Generated 2026-10-26T08:23:34+00:00 by `src/evaluation/evaluate.py` from `artifacts/metrics.json`._

## Model details
- **Preferred model:** `m3_lightgbm` (sklearn Pipeline: train-only median imputation
  + missing indicators, one-hot categoricals, then the regressor).
- **Hyper-parameters:** `{"subsample_freq": 1, "subsample": 0.9, "num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.03, "colsample_bytree": 1.0}`
- **Selection rule:** lowest train-CV MAE among M1-M3 (hold-out not used for selection).
- **Seed:** 42. Train rows: 36,474. Hold-out rows: 9,119.
- Artefact: `artifacts/models/preferred_model.joblib` (sha256 `db00c57f494bd073…`).

## Intended use (C3)
- **Users:** analytics / operations reviewers exploring what drives delivery duration in a
  public dataset, and how well tabular regression can estimate it.
- **Use:** offline decision-support and teaching demo (ops-proxy).
- **Out of scope:** live customer-facing ETAs, courier routing/GPS, dispatch, pay or
  individual courier assessment, any use on personal data.

## Metrics (C1) — hold-out, minutes
| | MAE | RMSE |
|---|---|---|
| Naive (train mean = 26.28 min) | 7.552 | 9.349 |
| Preferred `m3_lightgbm` | 3.104 | 3.913 |
| Improvement vs naive | 58.9% | 58.1% |

**Success criterion:** **MET** — the preferred model improves both MAE and RMSE over the naive baseline on the hold-out.

### Leaderboard (selection used train-CV only; hold-out shown for reporting)
| model            |   train-CV MAE |   train-CV RMSE |   hold-out MAE |   hold-out RMSE |
|:-----------------|---------------:|----------------:|---------------:|----------------:|
| m3_lightgbm ✅    |         3.0885 |          3.8737 |         3.1037 |          3.9131 |
| m2_random_forest |         3.0907 |          3.8652 |         3.1337 |          3.946  |
| m3_xgboost       |         3.099  |          3.893  |         3.1221 |          3.9373 |
| m1_elasticnet    |         4.6643 |          5.8521 |         4.6952 |          5.8895 |
| m1_ols           |         4.6649 |          5.8528 |         4.6957 |          5.8893 |
| m0_naive         |         7.6025 |          9.3925 |         7.5523 |          9.3494 |

## Subgroup errors (C2)
Full table: `artifacts/subgroup_metrics.csv` (traffic, weather, distance bands and the
traffic × weather cross). Cells with n < 50 are flagged `low_n`.

- Lowest-error band: **traffic_band = Low** (MAE 2.67, n=3,099)
- Highest-error band: **weather_band = Unknown** (MAE 6.03, n=120)
- Low-n cells flagged: **4** — Low / Unknown (n=1), Medium / Unknown (n=1), High / Unknown (n=1), Jam / Unknown (n=1)

## What explains duration (research question)
Permutation importance on the hold-out (increase in MAE, minutes, when a feature is shuffled):

| feature             |   mae_increase |    std |
|:--------------------|---------------:|-------:|
| weather_band        |         1.9426 | 0.0127 |
| traffic_band        |         1.8504 | 0.0209 |
| distance_km         |         1.5609 | 0.0207 |
| courier_age         |         1.3557 | 0.0172 |
| courier_rating      |         1.0981 | 0.0098 |
| vehicle_condition   |         1.0133 | 0.0208 |
| multiple_deliveries |         0.4858 | 0.0165 |
| festival            |         0.1235 | 0.0082 |

Importance is associative within this dataset, not causal.

## Sensitivity check — ambiguous field
Refit `m3_lightgbm` with the same hyper-parameters **without `courier_rating`**
(train only), scored on the same hold-out: MAE **3.347**, RMSE **4.165**
(vs 3.104 / 3.913 with it; naive 7.552 / 9.349).

## Limitations
- **Offline ops-proxy demo, not a production ETA.** Not validated for customer-facing promises, courier dispatch, pay, or performance management.
- **Single public dataset, short window.** ~45k orders spanning roughly six weeks in Feb–Apr 2022 across Indian cities. No evidence it transfers to other cities, seasons, platforms or years.
- **Provenance not verifiable.** The source does not document how records were collected; they may be partly simulated. Treat findings as properties of this dataset, not of real-world delivery operations.
- **Random (not temporal) hold-out.** Hold-out rows share dates with training rows, so the score is optimistic relative to forecasting future weeks.
- **No live GPS, routing, restaurant load or real-time traffic feeds.** Distance is straight-line (haversine), not road distance; placeholder (≈0) coordinates exist.
- **Prediction point is at pickup.** Uses pickup time and preparation time; an estimate at order-placement time would need those removed and would likely be less accurate.
- **Courier rating is ambiguous** (may include post-delivery feedback). See sensitivity check.
- **Subgroup cells with n < 50 are flagged** and should not be used for conclusions.
- Errors are in whole minutes on a 10–54 minute target; individual predictions can be off by much more than the average error.

## Ethical / privacy considerations (C5)
- Public data only; no new data collected. Courier and order IDs and raw coordinates are
  excluded from the feature table and from the dashboard. No re-identification attempted.
- Dashboard is aggregate-first; the case table shows a small sample without IDs or locations.
- The model must not be used to rate or rank individual couriers.

## Reproducibility (C6)
`scripts/run_all.sh` regenerates everything from the raw file. Input/output SHA-256 hashes are in
`artifacts/metrics.json`; every run also writes a dated copy to `artifacts/history/`.

Data: Malik, G. (2022). Food delivery dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

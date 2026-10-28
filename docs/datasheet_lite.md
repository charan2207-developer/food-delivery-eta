# Datasheet-lite — Food delivery dataset (Malik, 2022)

_Generated 2026-10-26T08:23:34+00:00. Pipeline facts come from `data/raw/acquisition.json` and
`data/interim/clean_stats.json`._

## Source & licence (C4)
- **Citation:** Malik, G. (2022). Food delivery dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset
- **File used:** `train.csv` only (45,593 rows; Kaggle `test.csv`
  has no target and is not used). SHA-256 `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`.
- **Licence:** governed by the terms shown on the [Kaggle dataset page](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset).
  Check them before redistributing. This repo **does not redistribute** the data;
  `data/` is git-ignored and is regenerated locally from Kaggle.

## Motivation
Public benchmark for food-delivery duration regression; used here for an offline ops-proxy demo.

## Composition
- One row per delivered order: courier age/rating (pseudonymous courier ID), restaurant &
  delivery coordinates, order date, order/pickup times, weather, traffic density, vehicle
  type/condition, order type, multiple deliveries, festival flag, city type, duration (min).
- Target range 10–54 minutes.
- Missingness is encoded as literal `NaN` strings; see `docs/transform_log.md` for counts.

## Collection process
Not documented in detail by the source. Whether records are real, anonymised, or simulated
cannot be verified. Results should be read as properties of this dataset.

## Preprocessing & exclusions
- Dropped: `ID`, `Delivery_person_ID` (identifiers / privacy).
- Raw coordinates used only to derive `distance_km`, then excluded.
- Sign-error coordinates corrected; out-of-range ratings nulled (53);
  placeholder coordinates flagged (3640).
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
Snapshot as downloaded on 2026-10-26T08:06:43+00:00. If the upstream file changes,
the SHA-256 changes; re-run `scripts/run_all.sh --refreeze` and republish metrics.

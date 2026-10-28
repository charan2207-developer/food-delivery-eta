# Transform log (S1–S6)

_Generated 2026-10-26T08:17:51+00:00 from `data/interim/clean_stats.json`. Counts are from the actual run._

| Stage | Rule | Rows affected |
|---|---|---|
| S1 | Use Kaggle `train.csv` only (test.csv has no target) | 45,593 in |
| S3 | Strip whitespace from all fields | all |
| S3 | Drop columns: `ID`, `Delivery_person_ID` | all |
| S3 | Literal `NaN` strings → null (per column) | courier_age: 1854, courier_rating: 1908, time_ordered: 1731, weather: 616, traffic: 601, multiple_deliveries: 993, festival: 228, city_type: 1200 |
| S3 | `conditions X` → `X` (weather); `(min) NN` → NN (target) | all |
| S3 | Rows with unparseable target dropped | 0 |
| S3 | Negative coordinates → absolute value (sign errors; all orders in India) | restaurant_lat: 431, restaurant_lon: 162, delivery_lat: 0, delivery_lon: 0 |
| S3 | Flag placeholder (≈0) restaurant coordinates as `coords_placeholder` | 3640 |
| S3 | Courier rating outside 1–5 → null | 53 |
| S3 | Exact duplicate rows dropped | 0 |
| S3 | **No imputation** (deferred to train-only model pipelines) | — |
| S4 | `distance_km` = haversine(restaurant, delivery); raw coords then discarded | all |
| S4 | `prep_time_min` = pickup − order time (+24h if negative) | 1731 null (order time missing) |
| S4 | Missing categoricals → `Unknown` level | see dictionary |
| S4 | Bands: distance (0-5/5-10/10-15/15+ km), traffic, weather | all |
| S6 | Seeded 80/20 train/hold-out split on `row_id` | see `data/splits/` |

Rows out of S3/S4: **45,593**. Target range: 10–54 min (no clipping needed).

## Leakage checklist

| Field / step | Decision |
|---|---|
| Target `Time_taken(min)` | Excluded from predictors (it is the label). |
| Prediction point | Defined as **at pickup**: everything used is known once the courier has collected the order. Nothing observed during or after the trip is used. |
| `Time_Order_picked` → `prep_time_min`, `pickup_hour` | Known at pickup, so allowed. If the estimate were needed at *order* time, `prep_time_min` would have to be dropped. |
| `Delivery_person_Ratings` → `courier_rating` | **Ambiguous.** Treated as a courier profile attribute known before the trip, but the source does not say whether it includes the rating of this delivery. Kept, with a sensitivity check (model retrained without it) reported in the model card. |
| `Delivery_person_ID` | Dropped (pseudonymous identifier; privacy + memorisation risk). |
| `ID` | Dropped (order identifier). |
| Raw lat/lon | Used only to compute `distance_km`; not stored in the feature table. |
| Imputation / scaling / encoding | Fit inside model pipelines on the **training split only** (no global statistics computed over the hold-out). |
| Hyper-parameter search & model selection | K-fold CV on the training split only. The hold-out is scored once per model, after selection, for reporting. |

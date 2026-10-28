# Data dictionary — `data/features/delivery_features.parquet`

_Generated 2026-10-26T08:17:51+00:00 by `src/pipeline/document.py` (S5). Rows: **45,593**._

Roles: `id` = join key only, `target` = label, `numeric` / `categorical` = model predictors. Bands used for subgroup evaluation: `traffic_band`, `weather_band`, `distance_band`.

| column                | role        | dtype    |   missing | range / levels                                         | source                         | description                                                                             |
|:----------------------|:------------|:---------|----------:|:-------------------------------------------------------|:-------------------------------|:----------------------------------------------------------------------------------------|
| `row_id`              | id          | int64    |         0 | 0 – 45592                                              | derived (S3)                   | Positional row index in the raw file; join key for splits/predictions. Not a predictor. |
| `time_taken_min`      | target      | float64  |         0 | 10.00 – 54.00                                          | Time_taken(min)                | Delivery duration in minutes.                                                           |
| `courier_age`         | numeric     | float64  |      1854 | 15.00 – 50.00                                          | Delivery_person_Age            | Courier age in years (NaN imputed in model pipeline).                                   |
| `courier_rating`      | numeric     | float64  |      1961 | 1.00 – 5.00                                            | Delivery_person_Ratings        | Courier profile rating 1–5; >5 set NaN in S3. See leakage note in transform log.        |
| `vehicle_condition`   | numeric     | float64  |         0 | 0.00 – 3.00                                            | Vehicle_condition              | Vehicle condition score 0–3.                                                            |
| `multiple_deliveries` | numeric     | float64  |       993 | 0.00 – 3.00                                            | multiple_deliveries            | Number of orders carried together (0–3).                                                |
| `distance_km`         | numeric     | float64  |         0 | 1.47 – 20.97                                           | lat/lon (S4)                   | Great-circle (haversine) restaurant→delivery distance, km.                              |
| `coords_placeholder`  | numeric     | int64    |         0 | 0.00 – 1.00                                            | derived (S3)                   | 1 if restaurant coordinates were ≈0 placeholders (distance less reliable).              |
| `prep_time_min`       | numeric     | float64  |      1731 | 5.00 – 15.00                                           | Time_Orderd, Time_Order_picked | Minutes between order and pickup (midnight wrap handled); NaN where order time missing. |
| `pickup_hour`         | numeric     | float64  |         0 | 0.00 – 23.00                                           | Time_Order_picked              | Hour of day of pickup (0–23).                                                           |
| `day_of_week`         | numeric     | float64  |         0 | 0.00 – 6.00                                            | Order_Date                     | Day of week of order, Monday=0.                                                         |
| `is_weekend`          | numeric     | int64    |         0 | 0.00 – 1.00                                            | Order_Date                     | 1 if Saturday/Sunday.                                                                   |
| `weather_band`        | categorical | category |         0 | Sunny, Cloudy, Windy, Fog, Stormy, Sandstorms, Unknown | Weatherconditions              | Weather condition band; missing -> Unknown.                                             |
| `traffic_band`        | categorical | category |         0 | Low, Medium, High, Jam, Unknown                        | Road_traffic_density           | Road traffic density band Low/Medium/High/Jam; missing -> Unknown.                      |
| `distance_band`       | categorical | category |         0 | 0-5 km, 5-10 km, 10-15 km, 15+ km                      | distance_km                    | Distance band: 0-5 km, 5-10 km, 10-15 km, 15+ km.                                       |
| `order_type`          | categorical | object   |         0 | Snack, Drinks, Buffet, Meal                            | Type_of_order                  | Snack / Meal / Drinks / Buffet.                                                         |
| `vehicle_type`        | categorical | object   |         0 | motorcycle, scooter, electric_scooter, bicycle         | Type_of_vehicle                | motorcycle / scooter / electric_scooter / bicycle.                                      |
| `festival`            | categorical | object   |         0 | No, Yes, Unknown                                       | Festival                       | Festival day Yes/No/Unknown.                                                            |
| `city_type`           | categorical | object   |         0 | Urban, Metropolitian, Semi-Urban, Unknown              | City                           | Metropolitian / Urban / Semi-Urban / Unknown (source spelling kept).                    |

Not present by design: courier ID, order ID, raw coordinates (see transform log).

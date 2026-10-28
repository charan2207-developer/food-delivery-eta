# S2 Profile notes

_Generated 2026-10-26T08:17:51+00:00 from `data/raw/delivery_raw.csv`._

- Rows: **45593**, columns: **20**
- All columns are read as strings first: missing values are encoded as the literal `NaN` (often with trailing spaces), and the target is encoded as `(min) NN`.

## Column profile

| column                      |   missing_like |   missing_pct |   n_unique |   numeric_parse_pct | example          |
|:----------------------------|---------------:|--------------:|-----------:|--------------------:|:-----------------|
| ID                          |              0 |          0    |      45593 |                 0   | 0x4607           |
| Delivery_person_ID          |              0 |          0    |       1320 |                 0   | INDORES13DEL02   |
| Delivery_person_Age         |           1854 |          4.07 |         23 |                95.9 | 37               |
| Delivery_person_Ratings     |           1908 |          4.18 |         29 |                95.8 | 4.9              |
| Restaurant_latitude         |              0 |          0    |        658 |               100   | 22.745049        |
| Restaurant_longitude        |              0 |          0    |        519 |               100   | 75.892471        |
| Delivery_location_latitude  |              0 |          0    |       4373 |               100   | 22.765049        |
| Delivery_location_longitude |              0 |          0    |       4373 |               100   | 75.912471        |
| Order_Date                  |              0 |          0    |         44 |                 0   | 19-03-2022       |
| Time_Orderd                 |           1731 |          3.8  |        177 |                 0   | 11:30:00         |
| Time_Order_picked           |              0 |          0    |        193 |                 0   | 11:45:00         |
| Weatherconditions           |            616 |          1.35 |          7 |                 0   | conditions Sunny |
| Road_traffic_density        |            601 |          1.32 |          5 |                 0   | High             |
| Vehicle_condition           |              0 |          0    |          4 |               100   | 2                |
| Type_of_order               |              0 |          0    |          4 |                 0   | Snack            |
| Type_of_vehicle             |              0 |          0    |          4 |                 0   | motorcycle       |
| multiple_deliveries         |            993 |          2.18 |          5 |                97.8 | 0                |
| Festival                    |            228 |          0.5  |          3 |                 0   | No               |
| City                        |           1200 |          2.63 |          4 |                 0   | Urban            |
| Time_taken(min)             |              0 |          0    |         45 |                 0   | (min) 24         |

## Target: delivery duration (minutes)

|       |   time_taken_min |
|:------|-----------------:|
| count |         45593    |
| mean  |            26.29 |
| std   |             9.38 |
| min   |            10    |
| 25%   |            19    |
| 50%   |            26    |
| 75%   |            32    |
| max   |            54    |

## Data-quality observations (drive S3 rules)

- Restaurant latitude < 0 in 431 rows, longitude < 0 in 162 rows (sign errors; all orders are in India).
- Restaurant coordinates ≈ 0 in 3640 rows (masked/placeholder locations).
- Courier ratings > 5 in 53 rows (outside the 1–5 scale).
- `Time_Orderd` missing in 1731 rows → preparation time not computable there.

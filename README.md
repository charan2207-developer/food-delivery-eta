# Food Delivery Time Analytics and Prediction
### A Regression-Based Decision-Support Solution

**Aim.** Examine what drives food-delivery duration in public records, train and compare
checked tabular regressors, and provide a Streamlit dashboard for ETA-style estimates and
operational review. **Offline ops-proxy demo only, not a production or customer-facing ETA.**

**Research question.** Which features in public food-delivery records best explain duration,
and how well can supervised regression estimate delivery time under MAE and RMSE?

**Data.** Malik, G. (2022). *Food delivery dataset* [Data set]. Kaggle.
<https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset>. About 45k labelled
orders (`train.csv`). Public only. The repo does not redistribute the data.

**Out of scope:** live platform APIs, courier routing/GPS features, collecting or scraping PII,
and deep spatio-temporal ETA networks (literature context only).

---

## Architecture

```
            ┌──────────────── governance: seed=42 · pinned deps · sha256 hashes ────────────────┐
            │                                                                                    │
  Kaggle    ▼                                                                                    ▼
 train.csv ─► S1 Acquire ─► S2 Profile ─► S3 Clean ─► S4 Engineer ─► S5 Document ─► S6 Freeze
             data/raw/      reports/eda/  data/interim/ data/features/  docs/data_dictionary   data/splits/
             delivery_raw   profile_notes delivery_     delivery_       docs/transform_log     row_ids.json
             .csv           .md           clean.csv     features.parquet                       (80/20)
                                                              │                                  │
                                                              ▼                                  ▼
             M0 naive mean ─► M1 OLS / ElasticNet ─► M2 RandomForest ─► M3 XGBoost / LightGBM
                 (train-only 5-fold CV for tuning AND selection; hold-out scored once, for reporting)
                                                              │
                         M4 artifacts/leaderboard.csv  ·  M5 preferred_model.joblib + y_pred_holdout.csv
                                                              │
                                                              ▼
             C1 MAE/RMSE vs naive · C2 subgroups (low_n<50) · C3 model card · C4 datasheet-lite
             C5 privacy · C6 dated metrics.json + sha256  ──►  artifacts/metrics.json (UI's ONLY KPI source)
                                                              │
                                                              ▼
             D1 Overview · D2 Drivers · D3 Pred vs actual · D4 Subgroups · D5 Case table · D6 Notes
                                       app/streamlit_app.py
```

| Layer | Code | Role |
|---|---|---|
| Data / pipeline (S1–S6) | `src/pipeline/` | acquire, profile, clean, engineer, document, freeze |
| Modelling (M0–M5) | `src/models/` | candidate pipelines, train-only CV, exports |
| Evaluation / governance (C1–C6) | `src/evaluation/` | metrics, subgroups, importance, model card, datasheet |
| Dashboard (D1–D6) | `app/streamlit_app.py` | read-only view over the artefacts |

## Install

Python 3.10+ (tested on 3.10, macOS arm64).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

macOS needs `libomp` for XGBoost/LightGBM (`brew install libomp`). SciPy is pinned to 1.14.1
because 1.15.x wheels fail to load on recent macOS.

## Get the data (S1)

Pick one:

1. **Automatic (default).** `run_all.sh` calls `kagglehub.dataset_download(...)`. Public
   datasets usually download without login. If prompted, run `kagglehub` login or put Kaggle
   credentials in `~/.kaggle/`. Never commit credentials.
2. **Manual.** Download the zip from the Kaggle page (accept its terms) and pass it in:
   `bash scripts/run_all.sh --source ~/Downloads/archive.zip` (a bare `train.csv` also works).

Only `train.csv` is used. Kaggle's `test.csv` has no target column.

## Run

```bash
bash scripts/run_all.sh                 # tests → S1–S6 → M0–M5 → C1–C6  (~3 min on a laptop)
bash scripts/run_all.sh --skip-acquire  # reuse data/raw/delivery_raw.csv
.venv/bin/streamlit run app/streamlit_app.py
```

Individual stages: `python -m src.pipeline.run`, `python -m src.models.train`,
`python -m src.evaluation.evaluate`. Tests: `.venv/bin/pytest` (synthetic fixture only, ~3 s).

Results live in `artifacts/metrics.json` and the generated `docs/model_card.md`. They are not
copied into this README so they can't drift.

## Interface contract

| Artefact | Producer → Consumer |
|---|---|
| `data/raw/delivery_raw.csv` | S1 → S2/S3 |
| `data/features/delivery_features.parquet` | S4 → models, eval, UI |
| `data/splits/row_ids.json` `{"train":[…],"holdout":[…]}` | S6 → models, eval |
| `docs/data_dictionary.md`, `docs/transform_log.md` | S5 → humans |
| `artifacts/baseline_metrics.json` | M0 → eval |
| `artifacts/models/{linear,rf,boost,preferred}_model.joblib` | M1–M5 → eval, archive |
| `artifacts/leaderboard.csv` | M4 → eval, docs |
| `artifacts/y_pred_holdout.csv` (`row_id,y_true,y_pred`) | M5 → eval, UI D3/D5 |
| `artifacts/metrics.json` | C1/C6 → **UI KPIs (only source)** |
| `artifacts/subgroup_metrics.csv` (+ `low_n`) | C2 → UI D4 |
| `artifacts/feature_importance.csv` | eval → UI D2, model card |
| `artifacts/history/metrics_<UTC>.json` | C6 audit trail |
| `docs/model_card.md`, `docs/datasheet_lite.md` | C3/C4 → humans |

## Change control (risk R8 / R9)

- The split is frozen. If the features change, `freeze` **refuses** to overwrite a differing
  `row_ids.json`. Re-run with `--refreeze`. `run_all.sh` then retrains the models and
  republishes the metrics, and the UI picks them up on reload.
- The UI reads KPIs only from `metrics.json` and hardcodes none.
- `metrics.json` records SHA-256 hashes of the raw file, features, split, preferred model and
  predictions. Every run also writes a dated copy to `artifacts/history/`.

## Honesty and leakage safeguards

- The prediction point is **at pickup**. The leakage checklist is in `docs/transform_log.md`.
- Imputation, scaling and encoding are fit inside sklearn pipelines on the training split
  only.
- Hyper-parameter search and preferred-model selection use **train-only CV**. A test shuffles
  the hold-out targets and checks that the selection doesn't change
  (`tests/test_models.py::test_no_holdout_peeping`).
- Success rule: the preferred model must beat the naive mean on both MAE and RMSE.
  The result is computed and reported as-is (`success_criteria_met` in `metrics.json`).
- `courier_rating` might include post-delivery feedback. A sensitivity check retrains the
  model without it and reports the score in the model card.
- The synthetic fixture in `tests/` is for CI only. Its metrics carry `fast_mode: true`, and
  the UI and model card flag them as not valid.

## Privacy and scope (NFR2/NFR3)

- Public data only. No scraping, no new PII and no re-identification attempts.
- Courier and order IDs are dropped in S3. Raw coordinates are used only to compute distance
  and never appear in the feature table or the UI.
- The UI is aggregate-first. The case table (D5) shows at most 25 sampled hold-out rows with
  no IDs or locations.

## Demo checklist (≤ 5 minutes)

1. **Landing (15 s):** read the yellow limitations banner aloud. This is an offline
   ops-proxy demo, not a production ETA.
2. **D1 Overview (60 s):** preferred model, MAE/RMSE against naive, the success badge and the
   model ladder. Point out that selection used train-CV, not the hold-out.
3. **D2 Drivers (60 s):** mean duration by traffic, weather and distance band. The
   traffic × weather heatmap masks low-n cells. Permutation importance answers the research
   question; stress that it is associative, not causal.
4. **D3 Pred vs actual (45 s):** the scatter against y = x, the residual histogram and the
   median and 90th-percentile error caption.
5. **D4 Subgroups (45 s):** switch to Traffic × Weather to show hatched `low_n` cells and the
   warning.
6. **D5 Case table (30 s):** filter to Jam + Stormy. Only a constrained sample is shown, with
   no IDs or coordinates.
7. **D6 Notes (30 s):** limitations, intended use, the courier-rating sensitivity result and
   the data citation.

## Power BI (contingency only)

Not required. If Streamlit can't be used, `artifacts/metrics.json`, `leaderboard.csv`,
`subgroup_metrics.csv` and `y_pred_holdout.csv` can be loaded into Power BI as-is. KPIs must
still come from `metrics.json` only, and the D6 limitations must be reproduced on the first
page.

## Repo layout

```
food-delivery-eta/
  README.md  requirements.txt  pytest.ini  .gitignore
  data/{raw,interim,features,splits}/   (git-ignored; regenerated)
  src/config.py                         paths contract, seed, constants
  src/pipeline/   acquire profile clean engineer document freeze run
  src/models/     pipelines train
  src/evaluation/ metrics evaluate docs
  app/streamlit_app.py
  artifacts/      metrics, leaderboard, predictions, models/, history/
  docs/           data_dictionary transform_log model_card datasheet_lite
  reports/eda/    profile_notes.md
  tests/          synthetic fixture + pipeline/model/eval/app tests
  scripts/run_all.sh
```

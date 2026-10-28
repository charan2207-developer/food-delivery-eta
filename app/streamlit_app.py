"""D1–D6 Streamlit decision-support dashboard (offline ops-proxy demo).

    streamlit run app/streamlit_app.py

Data contract:
  D1 KPIs           <- artifacts/metrics.json ONLY (no hardcoded numbers)
  D2 drivers        <- data/features/delivery_features.parquet (+ feature_importance.csv)
  D3 pred vs actual <- artifacts/y_pred_holdout.csv joined to features
  D4 subgroups      <- artifacts/subgroup_metrics.csv
  D5 case table     <- hold-out join, constrained sample, no IDs / coordinates
  D6 notes          <- static copy (shared with the model card)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CITATION, SEED, Paths  # noqa: E402
from src.evaluation.docs import LIMITATIONS  # noqa: E402
from src.pipeline.engineer import (DISTANCE_LABELS, TRAFFIC_ORDER,  # noqa: E402
                                   WEATHER_ORDER)

MODEL_COLOR = "#2a78d6"   # categorical slot 1 (blue)
NAIVE_COLOR = "#eb6834"   # categorical slot 2 (orange)
CASE_TABLE_MAX_ROWS = 25
SCATTER_MAX_POINTS = 4000
BAND_ORDER = {"traffic_band": TRAFFIC_ORDER, "weather_band": WEATHER_ORDER,
              "distance_band": DISTANCE_LABELS}
BAND_LABEL = {"traffic_band": "Traffic", "weather_band": "Weather", "distance_band": "Distance"}

st.set_page_config(page_title="Food Delivery Time Analytics", page_icon="🛵", layout="wide")
PATHS = Paths.default()


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_json(path: str, _mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data
def load_csv(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_parquet(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def require(*paths: Path) -> None:
    missing = [str(p.relative_to(PATHS.root)) for p in paths if not p.exists()]
    if missing:
        st.error("Missing artefacts: " + ", ".join(f"`{m}`" for m in missing)
                 + ". Run `bash scripts/run_all.sh` first (see README).")
        st.stop()


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8),
                      legend=dict(orientation="h", y=1.08, x=0, title=None),
                      bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def fmt(x: float) -> str:
    return f"{x:.2f}"


# ---------------------------------------------------------------- landing + D6 banner
require(PATHS.metrics, PATHS.features, PATHS.y_pred_holdout, PATHS.subgroup_metrics)
metrics = load_json(str(PATHS.metrics), _mtime(PATHS.metrics))
feats = load_parquet(str(PATHS.features), _mtime(PATHS.features))
preds = load_csv(str(PATHS.y_pred_holdout), _mtime(PATHS.y_pred_holdout))
subgroups = load_csv(str(PATHS.subgroup_metrics), _mtime(PATHS.subgroup_metrics))
holdout = preds.merge(feats.drop(columns="time_taken_min"), on="row_id", how="left")
holdout["residual"] = holdout["y_pred"] - holdout["y_true"]

st.title("Food Delivery Time Analytics and Prediction")
st.caption("A regression-based decision-support demo on public data · " + CITATION)
st.warning(
    "**Offline ops-proxy demo — not a production or customer-facing ETA.** "
    "Public Kaggle data only; no live GPS, routing or real-time traffic. Random hold-out "
    "(not a future period). Subgroups with small n are flagged. See **D6 Notes** before "
    "drawing conclusions.", icon="⚠️")
if metrics.get("fast_mode"):
    st.error("These metrics were produced in **fast/smoke mode** and are NOT valid results.")

tabs = st.tabs(["D1 Overview", "D2 Drivers", "D3 Pred vs actual", "D4 Subgroup errors",
                "D5 Case table", "D6 Notes"])

# ---------------------------------------------------------------- D1 Overview
with tabs[0]:
    n, p = metrics["naive"], metrics["preferred"]
    ok = metrics["beats_naive_mae"] and metrics["beats_naive_rmse"]
    st.subheader(f"Preferred model: `{metrics['preferred_model']}`")
    c = st.columns(4)
    c[0].metric("Hold-out MAE (min)", fmt(p["mae"]), delta=fmt(p["mae"] - n["mae"]) + " vs naive",
                delta_color="inverse")
    c[1].metric("Hold-out RMSE (min)", fmt(p["rmse"]),
                delta=fmt(p["rmse"] - n["rmse"]) + " vs naive", delta_color="inverse")
    c[2].metric("Naive MAE (min)", fmt(n["mae"]), help="Predicts the training-set mean for all")
    c[3].metric("Naive RMSE (min)", fmt(n["rmse"]))
    if ok:
        st.success(f"✅ Success criterion met: beats naive on both MAE "
                   f"({metrics['improvement_vs_naive_pct']['mae']:.1f}% lower) and RMSE "
                   f"({metrics['improvement_vs_naive_pct']['rmse']:.1f}% lower).")
    else:
        st.error("❌ Success criterion NOT met: preferred model does not beat naive on both "
                 f"MAE (beats: {metrics['beats_naive_mae']}) and RMSE "
                 f"(beats: {metrics['beats_naive_rmse']}).")

    board = pd.DataFrame(metrics["leaderboard"]).sort_values("mae", ascending=False)
    fig = go.Figure()
    for name, col, color in [("Hold-out MAE", "mae", MODEL_COLOR),
                             ("Hold-out RMSE", "rmse", NAIVE_COLOR)]:
        fig.add_bar(y=board["model"], x=board[col], name=name, orientation="h",
                    marker_color=color, text=board[col].map(fmt), textposition="outside",
                    hovertemplate="%{y}<br>" + name + ": %{x:.3f} min<extra></extra>")
    fig.update_layout(barmode="group", title="Model ladder (hold-out, minutes; lower is better)")
    fig.update_xaxes(title="minutes")
    st.plotly_chart(style(fig, 380), width="stretch")
    st.caption(f"Selection rule: {metrics.get('selection_rule', 'n/a')}. "
               f"Hold-out n = {metrics.get('n_holdout', len(preds)):,}. "
               f"Seed {metrics['seed']}. Metrics generated {metrics['generated_at']}.")
    with st.expander("Leaderboard table"):
        st.dataframe(pd.DataFrame(metrics["leaderboard"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- D2 Drivers
with tabs[1]:
    st.subheader("What drives delivery duration?")
    st.caption("Aggregates over all rows of the feature table (mean duration ± 95% CI, n on hover).")
    cols = st.columns(3)
    for col, band in zip(cols, ["traffic_band", "weather_band", "distance_band"]):
        g = (feats.groupby(band, observed=True)["time_taken_min"]
             .agg(["mean", "std", "count"]).reindex(BAND_ORDER[band]).dropna().reset_index())
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
        fig = go.Figure(go.Bar(
            x=g[band], y=g["mean"], marker_color=MODEL_COLOR,
            error_y=dict(type="data", array=g["ci"], thickness=1.2, width=4),
            customdata=g[["count"]],
            hovertemplate="%{x}<br>mean %{y:.1f} min<br>n=%{customdata[0]:,}<extra></extra>"))
        fig.update_layout(title=f"{BAND_LABEL[band]} band", showlegend=False)
        fig.update_yaxes(title="mean duration (min)", rangemode="tozero")
        col.plotly_chart(style(fig, 320), width="stretch")

    left, right = st.columns(2)
    min_n = metrics["subgroup_min_n"]
    cells = (feats.groupby(["traffic_band", "weather_band"], observed=True)["time_taken_min"]
             .agg(["mean", "count"]))
    mean = cells["mean"].unstack().reindex(index=TRAFFIC_ORDER, columns=WEATHER_ORDER)
    count = cells["count"].unstack().reindex(index=TRAFFIC_ORDER, columns=WEATHER_ORDER)
    masked = mean.where(count >= min_n)  # small cells are not coloured (unreliable means)
    text = np.where(count >= min_n, mean.round(1).astype(str), f"n<{min_n}")
    fig = go.Figure(go.Heatmap(
        z=masked, x=WEATHER_ORDER, y=TRAFFIC_ORDER, text=text, texttemplate="%{text}",
        customdata=count.fillna(0), colorbar=dict(title="mean min"),
        colorscale=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        hovertemplate="%{y} traffic · %{x}<br>mean %{z:.1f} min<br>n=%{customdata:,}"
                      "<extra></extra>"))
    fig.update_layout(title=f"Mean duration: traffic × weather (cells with n<{min_n} masked)")
    fig.update_yaxes(autorange="reversed")
    left.plotly_chart(style(fig, 360), width="stretch")
    if PATHS.feature_importance.exists():
        fi = load_csv(str(PATHS.feature_importance), _mtime(PATHS.feature_importance)).head(10)
        fig = go.Figure(go.Bar(
            y=fi["feature"][::-1], x=fi["mae_increase"][::-1], orientation="h",
            marker_color=MODEL_COLOR, error_x=dict(type="data", array=fi["std"][::-1]),
            hovertemplate="%{y}: +%{x:.2f} min MAE<extra></extra>"))
        fig.update_layout(title="Permutation importance (hold-out MAE increase, min)")
        right.plotly_chart(style(fig, 360), width="stretch")
        right.caption("Associative within this dataset, not causal.")

# ---------------------------------------------------------------- D3 Pred vs actual
with tabs[2]:
    st.subheader(f"Predicted vs actual — hold-out ({len(holdout):,} orders)")
    left, right = st.columns(2)
    sample = holdout.sample(min(SCATTER_MAX_POINTS, len(holdout)), random_state=SEED)
    jitter = np.random.default_rng(SEED).uniform(-0.3, 0.3, len(sample))
    fig = go.Figure(go.Scattergl(
        x=sample["y_true"] + jitter, y=sample["y_pred"], mode="markers",
        marker=dict(size=5, color=MODEL_COLOR, opacity=0.25), name="order",
        hovertemplate="actual %{x:.0f} min<br>predicted %{y:.1f} min<extra></extra>"))
    lo, hi = float(holdout["y_true"].min()), float(holdout["y_true"].max())
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect (y = x)",
                    line=dict(color="#52514e", width=1.5, dash="dash"), hoverinfo="skip")
    fig.update_layout(title=f"Scatter (random sample of {len(sample):,}; x jittered ±0.3 "
                            "since actuals are whole minutes)")
    fig.update_xaxes(title="actual (min)")
    fig.update_yaxes(title="predicted (min)")
    left.plotly_chart(style(fig, 420), width="stretch")

    fig = go.Figure(go.Histogram(x=holdout["residual"], nbinsx=60, marker_color=MODEL_COLOR,
                                 hovertemplate="residual %{x}<br>%{y} orders<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_dash="dash")
    fig.update_layout(title="Residuals (predicted − actual, min)", showlegend=False)
    fig.update_xaxes(title="minutes")
    fig.update_yaxes(title="orders")
    right.plotly_chart(style(fig, 420), width="stretch")
    q = holdout["residual"].abs().quantile([0.5, 0.9]).round(1)
    right.caption(f"Median |error| {q[0.5]} min; 90% of hold-out orders within ±{q[0.9]} min.")

# ---------------------------------------------------------------- D4 Subgroup errors
with tabs[3]:
    st.subheader("Errors by subgroup (hold-out)")
    min_n = metrics["subgroup_min_n"]
    kind = st.radio("Group by", list(subgroups["subgroup"].unique()), horizontal=True,
                    format_func=lambda s: " × ".join(BAND_LABEL.get(x, x) for x in s.split(" x ")))
    g = subgroups[subgroups["subgroup"] == kind].copy()
    order = BAND_ORDER.get(kind)
    if order:
        g["level"] = pd.Categorical(g["level"], order, ordered=True)
        g = g.sort_values("level")
    g["level"] = g["level"].astype(str)
    g["label"] = np.where(g["low_n"], g["level"] + " ⚠ low n", g["level"])
    pattern = np.where(g["low_n"], "/", "")
    fig = go.Figure()
    for name, col, color in [(f"{metrics['preferred_model']} MAE", "mae", MODEL_COLOR),
                             ("naive MAE", "naive_mae", NAIVE_COLOR)]:
        fig.add_bar(x=g["label"], y=g[col], name=name, marker_color=color,
                    marker_pattern_shape=pattern, customdata=g[["n"]],
                    hovertemplate="%{x}<br>" + name + ": %{y:.2f} min<br>n=%{customdata[0]:,}"
                                  "<extra></extra>")
    fig.update_layout(barmode="group", title="MAE by level (hatched = n < "
                      f"{min_n}, not reliable)")
    fig.update_yaxes(title="MAE (min)", rangemode="tozero")
    st.plotly_chart(style(fig, 420), width="stretch")
    n_low = int(g["low_n"].sum())
    if n_low:
        st.warning(f"{n_low} cell(s) have n < {min_n}; treat their errors as unreliable.")
    else:
        st.info(f"All cells in this grouping have n ≥ {min_n}.")
    st.dataframe(g.drop(columns=["label", "subgroup"]), hide_index=True, width="stretch",
                 column_config={"low_n": st.column_config.CheckboxColumn(f"n < {min_n}")})

# ---------------------------------------------------------------- D5 Case table
with tabs[4]:
    st.subheader("Case table (constrained sample)")
    st.caption(f"Up to {CASE_TABLE_MAX_ROWS} random hold-out orders matching the filters. "
               "No order/courier IDs or coordinates are shown.")
    f = st.columns(3)
    pick = {b: f[i].multiselect(BAND_LABEL[b], [x for x in BAND_ORDER[b]
                                                if x in set(holdout[b].astype(str))])
            for i, b in enumerate(["traffic_band", "weather_band", "distance_band"])}
    view = holdout
    for b, vals in pick.items():
        if vals:
            view = view[view[b].astype(str).isin(vals)]
    st.write(f"{len(view):,} matching hold-out orders · MAE "
             f"{view['residual'].abs().mean():.2f} min" if len(view) else "No matching orders.")
    if len(view):
        cols = ["traffic_band", "weather_band", "distance_band", "order_type", "vehicle_type",
                "multiple_deliveries", "festival", "city_type", "y_true", "y_pred", "residual"]
        show = view.sample(min(CASE_TABLE_MAX_ROWS, len(view)), random_state=SEED)[cols]
        show = show.rename(columns={"y_true": "actual (min)", "y_pred": "predicted (min)",
                                    "residual": "error (min)"}).round(1)
        st.dataframe(show, hide_index=True, width="stretch")

# ---------------------------------------------------------------- D6 Notes
with tabs[5]:
    st.subheader("Limitations & intended use")
    st.markdown("**Intended use:** offline analytics / ops-proxy decision support and teaching. "
                "**Not for:** live customer ETAs, courier dispatch, routing, pay or assessing "
                "individual couriers.")
    st.markdown("\n".join(f"- {x.format(min_n=metrics['subgroup_min_n'])}" for x in LIMITATIONS))
    sens = metrics.get("sensitivity")
    if sens:
        st.markdown(f"**Sensitivity:** without `{sens['dropped_feature']}` the preferred model "
                    f"scores MAE {sens['mae']:.2f} / RMSE {sens['rmse']:.2f} min on the same "
                    "hold-out.")
    st.markdown(f"**Data:** {CITATION}. Licence per the Kaggle dataset page; data is not "
                "redistributed by this repo. Details: `docs/model_card.md`, "
                "`docs/datasheet_lite.md`, `docs/data_dictionary.md`, `docs/transform_log.md`.")
    st.caption(f"metrics.json generated {metrics['generated_at']} · notes: {metrics['notes']}")

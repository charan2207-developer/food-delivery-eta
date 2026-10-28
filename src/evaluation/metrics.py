"""Metric helpers shared by models (M4) and evaluation (C1/C2)."""
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def score(y_true, y_pred) -> dict:
    return {"mae": round(mae(y_true, y_pred), 4), "rmse": round(rmse(y_true, y_pred), 4)}

"""Baseline helpers shared by the notebooks.

Gathering of raw glucose lags from the contiguous 1-minute grid, the GlucoBench linear regression (fit and predict)
and the Sparacino AR(1)-WLS forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "gather_lags",
    "lag_columns",
    "fit_glucobench",
    "predict_glucobench",
    "predict_estimator",
    "sparacino_coefficient",
    "sparacino_forecast",
    "predict_sparacino",
]


def gather_lags(grid: pd.DataFrame, windows: pd.DataFrame, offsets: list, target_field: str) -> np.ndarray:
    """Glucose values at fixed minute offsets from each window origin.

    `grid` has columns subject, timestamp and `target_field` on the contiguous one-minute grid;
    `windows` has subject and origin_time. Offsets must be a nonempty sequence of
    nonpositive integer minutes, with every requested position inside the subject's grid.
    """
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all() or (offsets > 0).any() or (offsets != np.floor(offsets)).any():
        raise ValueError("offsets must be a nonempty sequence of nonpositive integer minutes")
    offsets = offsets.astype(int)
    grid = grid.sort_values(["subject", "timestamp"])
    if not grid.groupby("subject")["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("glucose history requires a contiguous one-minute grid per subject")
    subject_start = grid.groupby("subject")["timestamp"].first()
    subject_values = {s: part[target_field].to_numpy() for s, part in grid.groupby("subject")}
    origin_index = ((windows["origin_time"] - windows["subject"].map(subject_start)) // pd.Timedelta(minutes=1)).to_numpy()
    lag_matrix = np.empty((len(windows), len(offsets)))
    for s, idx in windows.groupby("subject").indices.items():
        positions = origin_index[idx][:, None] + offsets[None, :]
        if (positions < 0).any() or (positions >= len(subject_values[s])).any():
            raise ValueError(f"requested glucose history is outside subject {s}'s grid")
        lag_matrix[idx] = subject_values[s][positions]
    if not np.isfinite(lag_matrix).all():
        raise ValueError("requested glucose history contains nonfinite values")
    return lag_matrix


def lag_columns(step: int, points: int, history_min: int = 120) -> list:
    """Most recent `points` observations at `step`-minute spacing, oldest first, ending at the origin."""
    offsets = list(range(-(points - 1) * step, 1, step))
    assert min(offsets) >= -history_min
    return [f"g{o}" for o in offsets]


def fit_glucobench(train: pd.DataFrame, columns: list, target: str = "target_value"):
    """One OLS model on min-max scaled lags; the scaler is fitted on the training lags and target only."""
    scaler = MinMaxScaler().fit(np.r_[train[columns].to_numpy().ravel(), train[target].to_numpy()].reshape(-1, 1))
    scaled = scaler.transform(train[columns].to_numpy().reshape(-1, 1)).reshape(-1, len(columns))
    model = LinearRegression(fit_intercept=True).fit(scaled, scaler.transform(train[[target]].to_numpy()).ravel())
    return model, scaler


def predict_glucobench(model, scaler, frame: pd.DataFrame, columns: list) -> np.ndarray:
    scaled = scaler.transform(frame[columns].to_numpy().reshape(-1, 1)).reshape(-1, len(columns))
    return scaler.inverse_transform(model.predict(scaled).reshape(-1, 1)).ravel()


def predict_estimator(model, scaler, frame: pd.DataFrame, columns: list) -> np.ndarray:
    """Ridge / shallow GBT on registered features: optional training-fitted scaler, then `model.predict`."""
    features = frame[columns]
    return model.predict(scaler.transform(features) if scaler is not None else features)


def sparacino_coefficient(u: np.ndarray, mu: float) -> np.ndarray:
    """Closed-form exponentially weighted LS for u_i = a u_(i-1); u has shape (n_windows, n_obs), oldest first."""
    x, y = u[:, :-1], u[:, 1:]
    weights = mu ** np.arange(x.shape[1] - 1, -1, -1)
    return (weights * x * y).sum(axis=1) / (weights * x * x).sum(axis=1)


def sparacino_forecast(u: np.ndarray, mu: float, steps: int) -> np.ndarray:
    return sparacino_coefficient(u, mu) ** steps * u[:, -1]


def predict_sparacino(frame: pd.DataFrame, columns: list, mu: float, steps: int) -> np.ndarray:
    return sparacino_forecast(frame[columns].to_numpy(), mu, steps)

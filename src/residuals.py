"""Out-of-fold residuals of the fixed GlucoBench base and the ridge factor block fitted on them.

The base keeps the coefficients of the stored baseline fit. Training residuals are out-of-fold: Protocol A cross-fits over time blocks inside each
subject's training period, Protocol B over subject groups; validation residuals always use the base fitted on the full training split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.baselines import fit_glucobench, predict_glucobench

__all__ = ["RIDGE_ALPHAS", "time_block_folds", "grouped_folds", "oof_predictions", "ridge_path_predictions", "subject_error_sums"]

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def time_block_folds(windows: pd.DataFrame, bounds: pd.DataFrame, n_blocks: int = 5) -> np.ndarray:
    """Block index of every training window; -1 when the support [history_start_time, target_time] crosses a block edge.

    `bounds` is indexed by subject with the start and end of the subject's training period (`lo`, `hi`); the period is cut into
    `n_blocks` blocks of equal duration.
    """
    block = np.full(len(windows), -1)
    history_start = windows["history_start_time"].to_numpy()
    target_time = windows["target_time"].to_numpy()
    for subject, idx in windows.groupby("subject").indices.items():
        lo, hi = bounds.loc[subject, "lo"], bounds.loc[subject, "hi"]
        edges = [np.datetime64(lo + (hi - lo) * k / n_blocks) for k in range(n_blocks + 1)]
        for b in range(n_blocks):
            last = b == n_blocks - 1
            inside = (history_start[idx] >= edges[b]) & ((target_time[idx] <= edges[b + 1]) if last else (target_time[idx] < edges[b + 1]))
            block[idx[inside]] = b
    return block


def grouped_folds(subjects: pd.Series, n_splits: int = 5) -> np.ndarray:
    """Deterministic GroupKFold (no shuffle) over subjects."""
    fold = np.empty(len(subjects), int)
    for k, (_, held_out) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(subjects)), groups=subjects)):
        fold[held_out] = k
    return fold


def oof_predictions(frame: pd.DataFrame, fold: np.ndarray, columns: list, n_folds: int = 5) -> np.ndarray:
    """GlucoBench LR predictions of each fold from a base fitted on the other folds (rows with fold < 0 enter no fit)."""
    prediction = np.full(len(frame), np.nan)
    for f in range(n_folds):
        model, scaler = fit_glucobench(frame.loc[(fold != f) & (fold >= 0)], columns)
        held_out = fold == f
        prediction[held_out] = predict_glucobench(model, scaler, frame.loc[held_out], columns)
    return prediction


def ridge_path_predictions(train_x: pd.DataFrame, train_residual: np.ndarray, eval_x: pd.DataFrame, alphas=RIDGE_ALPHAS) -> np.ndarray:
    """Residual predictions of ridge fits for every alpha; shape (rows of eval_x, len(alphas)).

    Median imputation and standardization are fitted on the training rows only (as in manual-ablation notebook); the penalized normal equations
    are solved for all alphas from one eigendecomposition.
    """
    alphas = np.asarray(alphas, float)
    if not len(train_x) or len(train_residual) != len(train_x) or not np.isfinite(train_residual).all():
        raise ValueError("ridge requires nonempty training rows and matching finite residuals")
    mean = float(np.mean(train_residual))
    if train_x.shape[1] == 0:
        return np.full((len(eval_x), len(alphas)), mean)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_x)
    imputed_train = imputer.transform(train_x)
    scaler = StandardScaler().fit(imputed_train)
    xt, xe = scaler.transform(imputed_train), scaler.transform(imputer.transform(eval_x))
    values, vectors = np.linalg.eigh(xt.T @ xt)
    projected = vectors.T @ (xt.T @ (np.asarray(train_residual, float) - mean))
    coefficients = vectors @ (projected[:, None] / (values[:, None] + alphas[None, :]))
    return xe @ coefficients + mean


def subject_error_sums(subjects, errors: np.ndarray) -> tuple:
    """Per-subject window count and sums of |error| and error^2 for one or several error columns (rows: windows).

    Returns (subject ids ascending, windows, sum |error|, sum error^2) with shapes (S,), (S,), (S, columns), (S, columns).
    """
    errors = np.asarray(errors, float)
    if len(errors) != len(subjects) or pd.isna(subjects).any() or not np.isfinite(errors).all():
        raise ValueError("subjects and finite errors must have matching rows and no missing subject IDs")
    errors = errors.reshape(len(subjects), -1) if len(subjects) else errors.reshape(0, errors.shape[1] if errors.ndim == 2 else 1)
    codes, ids = pd.factorize(np.asarray(subjects), sort=True)
    windows = np.bincount(codes)
    absolute = np.stack([np.bincount(codes, weights=np.abs(errors[:, j])) for j in range(errors.shape[1])], axis=1)
    squared = np.stack([np.bincount(codes, weights=errors[:, j] ** 2) for j in range(errors.shape[1])], axis=1)
    return ids, windows, absolute, squared


def subject_table(subjects, errors: np.ndarray) -> pd.DataFrame:
    """Per-subject MAE / RMSE tables (one error vector) in the layout `src.evaluation.paired_inputs` expects."""
    ids, windows, absolute, squared = subject_error_sums(subjects, errors)
    table = pd.DataFrame({"n_windows": windows, "sum_abs_error": absolute[:, 0], "sum_squared_error": squared[:, 0]}, index=pd.Index(ids, name="subject"))
    table["mae"], table["rmse"] = table["sum_abs_error"] / windows, np.sqrt(table["sum_squared_error"] / windows)
    return table


def best_alpha(residual_prediction: np.ndarray, base_prediction: np.ndarray, target: np.ndarray, subjects, alphas=RIDGE_ALPHAS) -> tuple:
    """Alpha with the lowest validation subject-equal RMSE of base + ridge block, and the per-subject table at that alpha."""
    errors = base_prediction[:, None] + residual_prediction - target[:, None]
    ids, windows, absolute, squared = subject_error_sums(subjects, errors)
    best = int(np.argmin(np.sqrt(squared / windows[:, None]).mean(axis=0)))
    table = pd.DataFrame({"n_windows": windows, "sum_abs_error": absolute[:, best], "sum_squared_error": squared[:, best]}, index=pd.Index(ids, name="subject"))
    table["mae"], table["rmse"] = table["sum_abs_error"] / windows, np.sqrt(table["sum_squared_error"] / windows)
    return best, float(np.asarray(alphas)[best]), table


def contrast_row(table_from: pd.DataFrame, table_to: pd.DataFrame, min_windows: int | None = None, n_resamples: int = 2000, seed: int = 42,
                 negligible_mg_dl: float = 0.1, *, include_p: bool = False) -> dict:
    """Paired subject-equal contrast (to minus from; negative = lower error) with the subject-level cluster bootstrap."""
    from src.evaluation import classify_effect, paired_cluster_bootstrap, paired_inputs

    if min_windows is not None:
        keep = table_from.index[(table_from["n_windows"] >= min_windows) & (table_to["n_windows"] >= min_windows)]
        table_from, table_to = table_from.loc[keep], table_to.loc[keep]
    stats = paired_cluster_bootstrap(paired_inputs(table_from, table_to), n_resamples, seed, include_p=include_p)
    d_rmse, low, high = stats["d_rmse"]
    return {"subjects": len(table_from), "validation_windows": int(table_to["n_windows"].sum()),
            "rmse_from": float(table_from["rmse"].mean()), "rmse_to": float(table_to["rmse"].mean()),
            "d_rmse": d_rmse, "d_rmse_low": low, "d_rmse_high": high, "d_mae": stats["d_mae"][0],
            "pooled_d_rmse": stats["pooled_window_d_rmse"][0], "improved_share": stats["improved_share"][0],
            "outcome": classify_effect(d_rmse, stats["improved_share"][0], negligible_mg_dl),
            **({"p_bootstrap": stats["p_bootstrap"]} if include_p else {})}

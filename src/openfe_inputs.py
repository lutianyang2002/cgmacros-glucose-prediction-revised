"""Inputs of the OpenFE comparator: internal train / validation split, `init_scores` and the audited input columns.

Everything here uses the training windows of one protocol only (the configured `separate_openfe_init_scores` rule): no window of the protocol's validation or test split, and no
internal-validation label, reaches an internal-train `init_score`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import fit_glucobench, predict_glucobench
from src.evaluation import every_fifteen_minutes
from src.residuals import grouped_folds, oof_predictions, time_block_folds

__all__ = ["internal_split_time", "internal_split_subjects", "internal_init_scores", "thin_to_fifteen_minutes", "block_folds_inside", "group_folds_inside", "INTERNAL_TRAIN", "INTERNAL_VALIDATION"]

INTERNAL_TRAIN, INTERNAL_VALIDATION, DROPPED = "internal_train", "internal_validation", "dropped"


def internal_split_time(windows: pd.DataFrame, start: pd.Series, train_end: pd.Series, share: float = 0.75) -> tuple:
    """Protocol A: the first `share` of each subject's training period is internal-train, the rest internal-validation.

    A window is internal-train when its whole support [history_start_time, target_time] ends at or before the cut, internal-validation when it starts at or after the cut; the windows that
    cross the cut belong to neither. Returns (role per window, cut per subject).
    """
    cut = start + share * (train_end - start)
    cut_of_window = windows["subject"].map(cut)
    role = np.where(windows["target_time"] <= cut_of_window, INTERNAL_TRAIN, np.where(windows["history_start_time"] >= cut_of_window, INTERNAL_VALIDATION, DROPPED))
    return role, cut


def internal_split_subjects(windows: pd.DataFrame, internal_train_subjects: list, internal_validation_subjects: list) -> np.ndarray:
    """Protocol B: internal roles from the saved subject membership (the other subjects of `windows` are rejected)."""
    train, validation = set(internal_train_subjects), set(internal_validation_subjects)
    if train & validation or set(windows["subject"]) != train | validation:
        raise ValueError("internal training and validation subjects must be disjoint and cover the windows")
    return np.where(windows["subject"].isin(train), INTERNAL_TRAIN, INTERNAL_VALIDATION)


def internal_init_scores(windows: pd.DataFrame, role: np.ndarray, lag_columns: list, internal_train_folds: np.ndarray) -> np.ndarray:
    """`init_scores` of the OpenFE search: GlucoBench linear-regression predictions on the original scale.

    Internal-train rows: out-of-fold predictions inside the internal-train rows (`internal_train_folds`, -1 = no init score). Internal-validation rows: the base fitted on all internal-train
    rows. Rows of neither role get NaN. No row of the internal-validation role takes part in any fit that produces an internal-train score.
    """
    init = np.full(len(windows), np.nan)
    is_train = role == INTERNAL_TRAIN
    train = windows[is_train]
    init[is_train] = oof_predictions(train, internal_train_folds, lag_columns)
    model, scaler = fit_glucobench(train, lag_columns)
    is_validation = role == INTERNAL_VALIDATION
    init[is_validation] = predict_glucobench(model, scaler, windows[is_validation], lag_columns)
    return init


def thin_to_fifteen_minutes(windows: pd.DataFrame, first_origin: pd.Series) -> np.ndarray:
    """Rows on the 15-minute step of the screening (origin minus the subject's first training origin is a multiple of 15 minutes)."""
    return every_fifteen_minutes(windows, first_origin)


def block_folds_inside(windows: pd.DataFrame, start: pd.Series, cut: pd.Series, n_blocks: int = 5) -> np.ndarray:
    """Protocol A: the 5 non-forward time blocks of the internal-train period (-1 for windows whose support crosses a block edge)."""
    return time_block_folds(windows, pd.DataFrame({"lo": start, "hi": cut}), n_blocks)


def group_folds_inside(windows: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Protocol B: deterministic GroupKFold over the internal-train subjects."""
    return grouped_folds(windows["subject"], n_splits)


# ------------------------------------------------------------------ audited input columns
_MEAL_CLASSES = ("breakfast", "lunch", "dinner", "snack")


def input_audit(manual: list, primitive_columns: list) -> pd.DataFrame:
    """One row per column that could reach the search, with its block, source and the reason for inclusion or exclusion.

    Included: the existing 50 features M without the MET features (the filter direction of the MET signal is not stated, so M's MET columns may hold up to 10 minutes of later activity; the
    lagged MET primitives take their place), the logged-meal primitives, heart-rate and activity-calorie statistics over 15 and 60 minutes, the lagged MET primitives and the inferred
    native-reading state with its missing indicator. Excluded: keys, interpolated glucose lags and anything built from them, the target, residuals and base predictions, laboratory values,
    gut and microbiome variables, availability variants of the meal columns, and calibration details of the native-reading anchors.
    """
    rows = []

    def add(column, block, source, included, reason):
        rows.append({"column": column, "block": block, "source": source, "included": included, "reason": reason})

    for column in manual:
        if column.startswith("act_met_"):
            add(column, "M activity", "manual feature", False, "MET filter direction unknown: may hold activity up to 10 minutes after the origin; the lag-10 MET primitives replace it")
        elif column.startswith("act_"):
            add(column, "M activity", "manual feature", True, "trailing window of minute-level activity at or before the origin")
        elif column.startswith("diet_"):
            add(column, "M diet", "manual feature", True, "logged meal rows at or before the origin; logged row time is the event time")
        else:
            add(column, "M baseline", "manual feature", True, "subject-constant baseline characteristic of M (an identity proxy within Protocol A, as in M)")
    present = set(primitive_columns)
    for column in sorted(present):
        if column in manual or column in ("subject", "origin_time", "task_id", "split"):
            continue
        if column.startswith("meal_") and ("_logged_" in column) and not column.endswith("_missing"):
            add(column, "meal primitive", "primitives", True, "latest logged meal of one class at or before the origin")
        elif column.startswith("meal_") and ("delay_" in column):
            add(column, "meal primitive", "primitives", False, "availability variant of the E9 sensitivity")
        elif column.startswith("act_") and column.endswith(("_15m", "_60m")) and ("_mean_" in column or "_max_" in column):
            add(column, "activity primitive", "primitives", True, "trailing window of minute-level activity at or before the origin")
        elif column.startswith("act_") and "_lag10" in column and not ("coverage" in column or "missing" in column):
            add(column, "activity primitive", "primitives", True, "MET window ending 10 minutes before the origin")
        elif column.startswith("act_"):
            add(column, "activity primitive", "primitives", False, "availability mask, or a window already in M")
        elif column in ("native_last", "native_diff", "native_slope_1h", "native_curv_1h", "native_missing"):
            add(column, "native-reading state", "primitives", True, "inferred native readings at or before the origin; NaN where the anchors are uncertain, with the missing indicator")
        elif column.startswith("native_"):
            add(column, "native-reading state", "primitives", False, "calibration detail of the anchor inference, not a state of the glucose signal")
        else:
            add(column, "unclassified", "primitives", False, "not part of the audited input list")
    for column, reason in [("subject", "subject identifier"), ("origin_time", "window key"), ("task_id", "window key"), ("split", "window key"), ("target_value", "target"),
                           ("base_prediction", "base output"), ("residual", "training target of the factor block"), ("oof_prediction", "out-of-fold base output"),
                           ("glucose_lag_*, glucose_mean_*, glucose_std_*, glucose_slope_*, g-120 ... g0", "interpolated glucose history (the base already holds the lags)"),
                           ("bmi_kg_m2", "same values as bio_bmi in M"), ("hba1c_percent, fasting_glucose_mg_dl, fasting_insulin_uU_ml, homa_ir", "laboratory values: offline sensitivity analysis only"),
                           ("gut_*, microbiome_*", "excluded from M and from the factor search")]:
        add(column, "excluded", "tables", False, reason)
    return pd.DataFrame(rows)

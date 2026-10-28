"""Training-split fitness of a DSL expression for the LLM search and its Random-DSL control.

The fitness of an expression is the gain, on held-out data inside the protocol's training split, of a one-factor ridge block fitted on the residual of the fixed GlucoBench
baseline. Every inner split fits the baseline on its own training part only and then forms both residuals from that baseline:

* the training residual is out-of-fold inside the training part (the configured `separate_openfe_init_scores` rule),
* the held-out residual comes from the baseline fitted on the whole training part,

so no held-out label, lag or residual reaches the training residual or the factor fit. The residuals of the main pipeline (computed with the baseline fitted on the other folds of
the whole training split) are not used here for that reason.

Sign: the fitness is a gain, `RMSE(baseline) - RMSE(baseline + factor)` on the held-out windows, averaged over held-out subjects with at least `min_held_windows` windows.
Higher is better; a factor that makes the held-out error larger has a negative fitness.

Protocol A uses one split: the first 75 % of each subject's training period against the rest (whole supports on one side). Protocol B uses a deterministic 3-fold GroupKFold over
the training subjects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.factor_construction import CandidateContext
from src.llm_factor_search import canonical, evaluate_expression, parse_expression, uses_native
from src.openfe_inputs import INTERNAL_TRAIN, INTERNAL_VALIDATION, block_folds_inside, group_folds_inside, internal_init_scores, internal_split_time
from src.residuals import grouped_folds, ridge_path_predictions

__all__ = ["FitnessSplit", "time_split", "group_splits", "split_cv_gain", "build_contexts", "ExpressionFitness", "boundary_report_time", "boundary_report_groups"]


@dataclass(frozen=True)
class FitnessSplit:
    """Row positions (into the frame) to fit on and to score on, and the baseline residual of every row of the split (NaN elsewhere)."""
    train: np.ndarray
    held: np.ndarray
    residual: np.ndarray


def _split(frame: pd.DataFrame, role: np.ndarray, lag_columns: list, train_folds: np.ndarray) -> FitnessSplit:
    init = internal_init_scores(frame, role, lag_columns, train_folds)
    residual = frame["target_value"].to_numpy(float) - init
    usable = frame["cohort_core"].to_numpy(bool) & np.isfinite(residual)
    return FitnessSplit(np.flatnonzero((role == INTERNAL_TRAIN) & usable), np.flatnonzero((role == INTERNAL_VALIDATION) & usable), residual)


def time_split(frame: pd.DataFrame, start: pd.Series, train_end: pd.Series, lag_columns: list, share: float = 0.75) -> FitnessSplit:
    """Protocol A: first `share` of every subject's training period for fitting, the rest held out; windows that cross the cut are in neither."""
    if not frame["split"].eq("train").all():
        raise ValueError("fitness splits require TRAIN-only windows")
    role, cut = internal_split_time(frame, start, train_end, share)
    return _split(frame, role, lag_columns, block_folds_inside(frame[role == INTERNAL_TRAIN], start, cut))


def group_splits(frame: pd.DataFrame, lag_columns: list, n_splits: int = 3, inner_folds: int = 5) -> list:
    """Protocol B: deterministic subject-grouped folds; the baseline of each fold is fitted on the other folds' subjects (with `inner_folds` subject-grouped folds inside them)."""
    if not frame["split"].eq("train").all():
        raise ValueError("fitness splits require TRAIN-only windows")
    outer = grouped_folds(frame["subject"].reset_index(drop=True), n_splits)
    splits = []
    for k in range(n_splits):
        role = np.where(outer == k, INTERNAL_VALIDATION, INTERNAL_TRAIN)
        splits.append(_split(frame, role, lag_columns, group_folds_inside(frame[role == INTERNAL_TRAIN], inner_folds)))
    return splits


def split_cv_gain(x: np.ndarray, splits: list, subjects: np.ndarray, alpha: float, indicator: np.ndarray | None = None, min_defined: int = 50, min_held_windows: int = 20) -> float | None:
    """Held-out gain of a one-factor ridge block (median imputation and standardization fitted on the training part of each split); None when it cannot be evaluated."""
    x = np.asarray(x, float).reshape(len(x), -1)
    if indicator is not None:
        x = np.c_[x, np.asarray(indicator, float)]
    base, with_factor, who = [], [], []
    for sp in splits:
        if np.isfinite(x[sp.train, 0]).sum() < min_defined or len(sp.held) == 0:
            return None
        prediction = ridge_path_predictions(pd.DataFrame(x[sp.train]), sp.residual[sp.train], pd.DataFrame(x[sp.held]), alphas=[alpha])[:, 0]
        residual = sp.residual[sp.held]
        base.append(residual)
        with_factor.append(residual - prediction)
        who.append(np.asarray(subjects)[sp.held])
    frame = pd.DataFrame({"s": np.concatenate(who), "e0": np.concatenate(base) ** 2, "e1": np.concatenate(with_factor) ** 2})
    grouped = frame.groupby("s")
    keep = grouped.size() >= min_held_windows
    if not keep.any():
        return None
    mean = grouped[["e0", "e1"]].mean()[keep]
    return float((np.sqrt(mean["e0"]) - np.sqrt(mean["e1"])).mean())


def build_contexts(frame: pd.DataFrame, minutes: dict, events: dict, bmi: pd.Series, no_events: pd.DataFrame, cut: pd.Series | None = None) -> dict:
    """Per-subject `CandidateContext` for the rows of `frame`. With `cut` (last admitted time per subject) minutes and events after it are dropped before anything is computed;
    values at origins on or before the cut do not change, because every operator reads minutes at or before the origin."""
    contexts = {}
    for subject, rows in frame.groupby("subject").indices.items():
        frame_minutes, frame_events = minutes[subject], events.get(subject, no_events)
        if cut is not None:
            frame_minutes = frame_minutes[frame_minutes["timestamp"] <= cut[subject]]
            frame_events = frame_events[frame_events["timestamp"] <= cut[subject]]
        contexts[subject] = (rows, CandidateContext(frame_minutes, frame_events, frame["origin_time"].iloc[rows].reset_index(drop=True), float(bmi[subject])))
    return contexts


class ExpressionFitness:
    """Callable fitness for `run_search` and `run_random_control`: canonical expression -> held-out gain (higher is better) or None."""

    def __init__(self, frame: pd.DataFrame, splits: list, contexts: dict, alpha: float, min_defined: int = 50, min_held_windows: int = 20):
        if not frame["split"].eq("train").all():
            raise ValueError("the fitness may only see training-split windows")
        self.frame, self.splits, self.contexts, self.alpha = frame.reset_index(drop=True), splits, contexts, alpha
        self.min_defined, self.min_held_windows = min_defined, min_held_windows
        self.subjects = self.frame["subject"].to_numpy()
        self.cache = {}

    def values(self, expression: str) -> np.ndarray:
        node = parse_expression(expression)
        key = canonical(node)
        if key not in self.cache:
            out = np.full(len(self.frame), np.nan)
            for rows, context in self.contexts.values():
                out[rows] = evaluate_expression(node, context)
            self.cache[key] = out
        return self.cache[key]

    def __call__(self, expression: str) -> float | None:
        node = parse_expression(expression)
        indicator = self.frame["native_missing"].to_numpy(float) if uses_native(node) else None
        return split_cv_gain(self.values(expression), self.splits, self.subjects, self.alpha, indicator, self.min_defined, self.min_held_windows)


def boundary_report_time(frame: pd.DataFrame, split: FitnessSplit, start: pd.Series, train_end: pd.Series, lag_columns: list, share: float = 0.75) -> dict:
    """Protocol A: counts of rows that would violate the time isolation of the inner split (all must be 0). A window's label is the glucose value 30 minutes after the origin and its
    history starts 120 minutes before it; `target_time` and `history_start_time` are those two ends."""
    role, cut = internal_split_time(frame, start, train_end, share)
    cut_of = frame["subject"].map(cut)
    fit, held = frame.iloc[split.train], frame.iloc[split.held]
    blocks = block_folds_inside(frame[role == INTERNAL_TRAIN], start, cut)
    fit_rows = frame[role == INTERNAL_TRAIN].assign(block=blocks)
    fit_rows = fit_rows[fit_rows["block"] >= 0]
    overlaps = 0
    for _, g in fit_rows.groupby("subject"):
        edges = g.groupby("block").agg(lo=("history_start_time", "min"), hi=("target_time", "max")).sort_index()
        overlaps += int((edges["hi"].shift(1) > edges["lo"]).sum())
    gap = (held.groupby("subject")["history_start_time"].min() - fit.groupby("subject")["target_time"].max()).dropna()
    return {"fit_rows": len(fit), "held_rows": len(held), "rows_in_both": int(len(set(split.train) & set(split.held))),
            "fit_rows_with_label_after_cut": int((fit["target_time"] > cut_of.iloc[split.train]).sum()),
            "held_rows_with_history_before_cut": int((held["history_start_time"] < cut_of.iloc[split.held]).sum()),
            "held_rows_with_label_before_cut": int((held["target_time"] <= cut_of.iloc[split.held]).sum()),
            "held_rows_with_label_after_training_end": int((held["target_time"] > frame["subject"].map(train_end).iloc[split.held]).sum()),
            "windows_crossing_the_cut_in_neither_part": int((role == "dropped").sum()),
            "overlapping_supports_between_oof_blocks": overlaps, "all_rows_are_training_split": bool(frame["split"].eq("train").all()),
            "smallest_gap_minutes_between_fit_label_and_held_history": float(gap.min() / pd.Timedelta(minutes=1)) if len(gap) else None}


def boundary_report_groups(frame: pd.DataFrame, splits: list, training_subjects: set) -> dict:
    """Protocol B: the held-out subjects of the folds are disjoint from the fitting subjects of their fold, cover every training subject once, and are all training subjects."""
    held_sets = [set(frame["subject"].iloc[s.held]) for s in splits]
    return {"folds": len(splits), "rows_in_both": [int(len(set(s.train) & set(s.held))) for s in splits],
            "held_subjects_per_fold": [len(h) for h in held_sets], "held_subjects_disjoint_across_folds": sum(len(h) for h in held_sets) == len(set().union(*held_sets)),
            "fit_subjects_overlapping_held_subjects": [int(len(set(frame["subject"].iloc[s.train]) & h)) for s, h in zip(splits, held_sets)],
            "subjects_outside_the_training_subjects": int(len(set(frame["subject"]) - training_subjects)), "all_rows_are_training_split": bool(frame["split"].eq("train").all())}

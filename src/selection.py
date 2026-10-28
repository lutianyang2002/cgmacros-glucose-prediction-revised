"""Redundancy filter and validation forward selection of the residual factor block."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.residuals import RIDGE_ALPHAS, best_alpha, ridge_path_predictions

__all__ = ["fit_block", "redundancy_filter", "forward_selection", "one_standard_error_choice", "selection_result"]


def fit_block(train: pd.DataFrame, valid: pd.DataFrame, columns: list, residual: str = "residual") -> tuple:
    """Ridge block on the out-of-fold residual of `train` evaluated on `valid`: (alpha, per-subject validation table, training windows).

    The block is added to the fixed base prediction `base_prediction`; alpha comes from the configured grid, chosen by the validation subject-equal
    RMSE. With no column the base prediction itself is evaluated.
    """
    if columns:
        prediction = ridge_path_predictions(train[columns], train[residual].to_numpy(), valid[columns])
    else:
        prediction = np.zeros((len(valid), len(RIDGE_ALPHAS)))
    _, alpha, table = best_alpha(prediction, valid["base_prediction"].to_numpy(), valid["target_value"].to_numpy(), valid["subject"].to_numpy())
    return (alpha if columns else np.nan), table, len(train)


def redundancy_filter(order: list, candidates: pd.DataFrame, base: pd.DataFrame, threshold: float = 0.9, method: str = "spearman") -> pd.DataFrame:
    """Greedy filter in the given order (descending |mean IC|): a candidate is dropped when |rho| exceeds `threshold` with a base feature or with
    an already kept candidate. rho is computed pairwise-complete over the training windows. Returns one record per candidate."""
    kept, records = [], []
    for rank, name in enumerate(order, start=1):
        verdict = {"candidate_id": name, "rank_by_abs_mean_ic": rank, "decision": "kept", "partner": "", "partner_type": "", "rho": np.nan, "max_abs_rho_checked": 0.0}
        for partner, frame, kind in [(p, base, "base feature") for p in base.columns] + [(p, candidates, "kept candidate") for p in kept]:
            pair = pd.concat([candidates[name], frame[partner]], axis=1, keys=["a", "b"]).dropna()
            if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
                continue
            rho = float(pair["a"].corr(pair["b"], method=method))
            if abs(rho) > verdict["max_abs_rho_checked"]:
                verdict["max_abs_rho_checked"] = abs(rho)
            if abs(rho) > threshold:
                verdict.update(decision="dropped", partner=partner, partner_type=kind, rho=rho)
                break
        if verdict["decision"] == "kept":
            kept.append(name)
        records.append(verdict)
    return pd.DataFrame(records)


def forward_selection(train: pd.DataFrame, valid: pd.DataFrame, pool: list, columns_of, k_max: int = 10) -> tuple:
    """Each step adds the pool member with the lowest validation subject-equal RMSE of base + ridge block (alpha chosen per model).

    `columns_of(selected)` maps the selected candidate ids to model columns (it may append the native-reading missing indicator).
    Returns (curve with one row per k = 0.., per-subject tables of every k).
    """
    alpha, table, _ = fit_block(train, valid, [])
    tables = {0: table}
    rows = [{"k": 0, "added": "", "alpha": np.nan, "subject_equal_rmse": float(table["rmse"].mean()), "pooled_rmse": float(np.sqrt(table["sum_squared_error"].sum() / table["n_windows"].sum()))}]
    selected, remaining = [], list(pool)
    while remaining and len(selected) < k_max:
        trials = []
        for name in remaining:
            a, t, _ = fit_block(train, valid, columns_of(selected + [name]))
            trials.append((float(t["rmse"].mean()), name, a, t))
        rmse, name, a, t = min(trials, key=lambda x: (x[0], pool.index(x[1])))
        selected.append(name)
        remaining.remove(name)
        tables[len(selected)] = t
        rows.append({"k": len(selected), "added": name, "alpha": a, "subject_equal_rmse": rmse, "pooled_rmse": float(np.sqrt(t["sum_squared_error"].sum() / t["n_windows"].sum()))})
    return pd.DataFrame(rows), tables


def one_standard_error_choice(curve: pd.DataFrame, tables: dict) -> dict:
    """Smallest k within one standard error of the best point of the validation curve, under two readings of the standard error.

    unpaired: SE of the mean subject RMSE at the best k (sd over subjects / sqrt(subjects)); rule: RMSE_k <= best + SE.
    paired:   SE of the subject-paired difference RMSE_k - RMSE_best; rule: mean difference <= SE of that difference.
    """
    subjects = tables[0]["rmse"].index
    if not subjects.is_unique or not all(t["rmse"].index.equals(subjects) and np.isfinite(t["rmse"]).all() for t in tables.values()):
        raise ValueError("every k must have finite RMSE on the same unique subjects in the same order")
    best = int(curve["subject_equal_rmse"].idxmin())              # the first minimum: the smallest k among ties
    best_k = int(curve.loc[best, "k"])
    rmse_best = tables[best_k]["rmse"]
    n = len(rmse_best)
    if n < 2:
        raise ValueError("a standard error over subjects needs at least two subjects")
    se_unpaired = float(rmse_best.std(ddof=1) / np.sqrt(n))
    out = {"best_k": best_k, "best_rmse": float(rmse_best.mean()), "se_unpaired": se_unpaired, "subjects": n}
    out["k_unpaired"] = int(next(k for k in curve["k"] if tables[int(k)]["rmse"].mean() <= rmse_best.mean() + se_unpaired))
    paired = {}
    for k in curve["k"]:
        d = tables[int(k)]["rmse"] - rmse_best
        paired[int(k)] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)))
    out["paired_difference_and_se"] = paired
    out["k_paired"] = int(next(k for k in curve["k"] if paired[int(k)][0] <= paired[int(k)][1] or int(k) == best_k))
    return out


def selection_result(curve, tables, protocol, train, valid, include_unpaired=False, **metadata):
    """Attach the shared paired-1-SE fields without changing selection or tie order."""
    rule = one_standard_error_choice(curve, tables)
    curve = curve.copy()
    curve.insert(0, "evaluation_protocol", protocol)
    curve["delta_rmse_vs_base"] = curve["subject_equal_rmse"] - curve.loc[curve["k"].eq(0), "subject_equal_rmse"].iloc[0]
    paired = rule["paired_difference_and_se"]
    curve["paired_diff_vs_best"] = [paired[int(k)][0] for k in curve["k"]]
    curve["paired_se_vs_best"] = [paired[int(k)][1] for k in curve["k"]]
    curve["best_k"], curve["se_unpaired_at_best"] = rule["best_k"], rule["se_unpaired"]
    if include_unpaired:
        curve["within_one_se_unpaired"] = curve["subject_equal_rmse"] <= rule["best_rmse"] + rule["se_unpaired"]
    curve["within_one_se_paired"] = curve["paired_diff_vs_best"] <= curve["paired_se_vs_best"]
    row = {"evaluation_protocol": protocol, "best_k": rule["best_k"], "k_selected": rule["k_paired"],
           "main_rule": "paired one standard error; Spearman redundancy filter", "k_one_se_unpaired_sensitivity": rule["k_unpaired"],
           "factors_in_order": json.dumps(list(curve["added"].iloc[1:])), "selected_set": json.dumps(list(curve["added"].iloc[1:rule["k_paired"] + 1])),
           "validation_windows": len(valid), "validation_subjects": valid["subject"].nunique(), "training_windows": len(train), **metadata}
    return curve, row

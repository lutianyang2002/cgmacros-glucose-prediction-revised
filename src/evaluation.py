"""Evaluation helpers shared by the notebooks.

Subject-equal and pooled error metrics, the subject-level paired cluster bootstrap and the practical-effect label,
defined once so that every notebook uses the same definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "error_summary",
    "evaluate",
    "subject_metrics",
    "summarize_by_subject",
    "paired_inputs",
    "draw_subjects",
    "replicate_statistics",
    "paired_cluster_bootstrap",
    "classify_effect",
    "every_fifteen_minutes",
    "subject_ic",
    "benjamini_yekutieli",
    "summarize_ic",
]


def error_summary(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("error metrics require a nonempty finite error vector")
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2))), "windows": int(len(errors))}


def evaluate(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    """Subject-equal-weight (primary) and pooled-window MAE/RMSE."""
    errors = prediction - frame["target_value"].to_numpy()
    per_subject = pd.DataFrame([error_summary(errors[idx]) for idx in frame.groupby("subject").indices.values()])
    pooled = error_summary(errors)
    return {"subjects": len(per_subject), "windows": pooled["windows"],
            "subject_mean_mae": per_subject["mae"].mean(), "subject_mean_rmse": per_subject["rmse"].mean(),
            "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"]}


def subject_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Per-subject MAE / RMSE / window count."""
    errors = prediction - frame["target_value"].to_numpy()
    return pd.DataFrame([{"subject": s, **error_summary(errors[idx])}
                         for s, idx in frame.groupby("subject").indices.items()])


def summarize_by_subject(predictions: pd.DataFrame, id_cols: list) -> tuple:
    """Per-subject sums of errors and the two aggregates per experiment.

    `predictions` needs `prediction`, `target_value`, `subject`, `origin_time` and `experiment_id` plus the `id_cols`.
    Returns (metrics_by_subject, metrics_summary):
      subject_equal_* = mean over subjects of the per-subject MAE / RMSE (primary);
      pooled_window_* = MAE / RMSE over all windows together (secondary).
    """
    rows = predictions.assign(error=predictions["prediction"] - predictions["target_value"])
    rows["abs_error"] = rows["error"].abs()
    rows["squared_error"] = rows["error"] ** 2
    by_subject = (rows.groupby(id_cols + ["subject"], sort=False)
                  .agg(n_windows=("error", "size"), first_origin_time=("origin_time", "min"), last_origin_time=("origin_time", "max"),
                       sum_abs_error=("abs_error", "sum"), sum_squared_error=("squared_error", "sum"))
                  .reset_index())
    by_subject["mae"] = by_subject["sum_abs_error"] / by_subject["n_windows"]
    by_subject["rmse"] = np.sqrt(by_subject["sum_squared_error"] / by_subject["n_windows"])
    # Experiments in order of first appearance, subjects ascending within each experiment.
    experiment_order = {e: i for i, e in enumerate(predictions["experiment_id"].unique())}
    by_subject = (by_subject.assign(order=by_subject["experiment_id"].map(experiment_order))
                  .sort_values(["order", "subject"]).drop(columns="order").reset_index(drop=True))
    grouped = by_subject.groupby(id_cols, sort=False)
    summary = grouped.agg(
        n_subjects=("subject", "nunique"), n_windows=("n_windows", "sum"),
        windows_per_subject_min=("n_windows", "min"), windows_per_subject_median=("n_windows", "median"), windows_per_subject_max=("n_windows", "max"),
        subject_equal_mae=("mae", "mean"), subject_equal_rmse=("rmse", "mean"),
        sum_abs_error=("sum_abs_error", "sum"), sum_squared_error=("sum_squared_error", "sum"),
    ).reset_index()
    summary["pooled_window_mae"] = summary["sum_abs_error"] / summary["n_windows"]
    summary["pooled_window_rmse"] = np.sqrt(summary["sum_squared_error"] / summary["n_windows"])
    summary = summary.drop(columns=["sum_abs_error", "sum_squared_error"])
    return by_subject, summary


def paired_inputs(subject_rows_from: pd.DataFrame, subject_rows_to: pd.DataFrame) -> dict:
    """Paired per-subject deltas (to minus from; negative = lower error in 'to') on identical subjects.

    Both inputs are per-subject tables indexed by subject with columns n_windows, mae, rmse,
    sum_abs_error, sum_squared_error (the rows of `summarize_by_subject`'s first output, indexed by subject).
    """
    subjects = subject_rows_from.index.sort_values()
    if subjects.hasnans or not subjects.is_unique or not subjects.equals(subject_rows_to.index.sort_values()):
        raise ValueError("paired tables require identical unique subject IDs")
    return {
        "subjects": subjects.to_numpy(),
        "d_mae": (subject_rows_to.loc[subjects, "mae"] - subject_rows_from.loc[subjects, "mae"]).to_numpy(),
        "d_rmse": (subject_rows_to.loc[subjects, "rmse"] - subject_rows_from.loc[subjects, "rmse"]).to_numpy(),
        "sums": {side: s.loc[subjects, ["n_windows", "sum_abs_error", "sum_squared_error"]].to_numpy()
                 for side, s in (("from", subject_rows_from), ("to", subject_rows_to))},
    }


def draw_subjects(n_subjects: int, n_resamples: int = 2000, seed: int = 42) -> np.ndarray:
    if n_subjects < 1 or n_resamples < 1:
        raise ValueError("bootstrap requires subjects and a positive resample count")
    return np.random.default_rng(seed).integers(0, n_subjects, size=(n_resamples, n_subjects))


def replicate_statistics(p: dict, draws: np.ndarray) -> dict:
    if (not np.isfinite(p["d_mae"]).all() or not np.isfinite(p["d_rmse"]).all()
            or any(not np.isfinite(v).all() or (v[:, 0] <= 0).any() for v in p["sums"].values())):
        raise ValueError("bootstrap requires finite paired statistics and positive window counts")
    d_mae, d_rmse = p["d_mae"][draws], p["d_rmse"][draws]
    out = {"d_mae": d_mae.mean(axis=1), "d_rmse": d_rmse.mean(axis=1), "improved_share": (d_rmse < 0).mean(axis=1)}
    pooled = {side: p["sums"][side][draws].sum(axis=1) for side in ("from", "to")}
    out["pooled_window_d_rmse"] = (np.sqrt(pooled["to"][:, 2] / pooled["to"][:, 0]) - np.sqrt(pooled["from"][:, 2] / pooled["from"][:, 0]))
    return out


def paired_cluster_bootstrap(p: dict, n_resamples: int = 2000, seed: int = 42, ci_level: float = 0.95, *, include_p: bool = False) -> dict:
    """Subject-level paired cluster bootstrap on fixed predictions (subjects drawn with replacement).

    Returns {statistic: (point estimate, interval low, interval high)} for d_mae, d_rmse, improved_share and
    pooled_window_d_rmse, using a percentile interval. ``include_p`` also returns the
    approximate two-sided ``p_bootstrap`` from the same RMSE resamples (+1 correction).
    """
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between zero and one")
    alpha = (1 - ci_level) / 2
    stats = replicate_statistics(p, draw_subjects(len(p["subjects"]), n_resamples, seed))
    sums = {side: p["sums"][side].sum(axis=0) for side in ("from", "to")}
    point = {"d_mae": p["d_mae"].mean(), "d_rmse": p["d_rmse"].mean(), "improved_share": (p["d_rmse"] < 0).mean(),
             "pooled_window_d_rmse": np.sqrt(sums["to"][2] / sums["to"][0]) - np.sqrt(sums["from"][2] / sums["from"][0])}
    out = {}
    for name, value in point.items():
        low, high = np.percentile(stats[name], [100 * alpha, 100 * (1 - alpha)])
        out[name] = (value, low, high)
    if include_p:
        d = stats["d_rmse"]
        out["p_bootstrap"] = float(min(1.0, 2 * (min((d >= 0).sum(), (d <= 0).sum()) + 1) / (n_resamples + 1)))
    return out


def classify_effect(d_rmse: float, improved_share: float, negligible_mg_dl: float) -> str:
    """Descriptive effect label using the supplied negligible-effect threshold."""
    if abs(d_rmse) < negligible_mg_dl:
        return "negligible"
    if d_rmse < 0 and improved_share > 0.5:
        return "improved"
    if d_rmse > 0 and improved_share < 0.5:
        return "worse"
    return "mixed"


# ---- subject-level information coefficient (candidate screening) -----------------------------------------------------------

def every_fifteen_minutes(frame: pd.DataFrame, first_origin: pd.Series) -> np.ndarray:
    """Rows on a 15-minute step: origin minus the subject's first training origin is a multiple of 15 minutes (windows are 5 minutes apart)."""
    elapsed = (frame["origin_time"] - frame["subject"].map(first_origin)) // pd.Timedelta(minutes=1)
    return (elapsed % 15 == 0).to_numpy()


def subject_ic(frame: pd.DataFrame, columns: list, target: str = "residual") -> tuple:
    """Spearman correlation of every column with `target`, computed separately for each subject on the windows where the column is defined.

    A subject contributes when at least 3 paired windows exist and neither side is constant (no minimum number of windows beyond that).
    Returns (IC table subjects x columns, window-count table subjects x columns); undefined entries are NaN / 0.
    """
    from scipy.stats import rankdata

    subjects = frame["subject"].to_numpy()
    r = frame[target].to_numpy(float)
    values = frame[columns].to_numpy(float)
    ids = np.unique(subjects)
    ic = np.full((len(ids), len(columns)), np.nan)
    counts = np.zeros((len(ids), len(columns)), int)
    for i, s in enumerate(ids):
        rows = np.flatnonzero(subjects == s)
        sub_r, sub_x = r[rows], values[rows]
        for j in range(len(columns)):
            ok = ~np.isnan(sub_x[:, j]) & ~np.isnan(sub_r)
            counts[i, j] = ok.sum()
            if ok.sum() < 3:
                continue
            a, b = sub_x[ok, j], sub_r[ok]
            if a.min() == a.max() or b.min() == b.max():
                continue
            ra, rb = rankdata(a), rankdata(b)
            ic[i, j] = np.corrcoef(ra, rb)[0, 1]
    index = pd.Index(ids, name="subject")
    return pd.DataFrame(ic, index=index, columns=columns), pd.DataFrame(counts, index=index, columns=columns)


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values (descriptive only); NaN stays NaN."""
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    scale = np.sum(1.0 / np.arange(1, m + 1))
    adjusted = p[order] * m * scale / np.arange(1, m + 1)
    out[order] = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    return out


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    """Across-subject summary per column: mean IC (signed), |mean IC|, same-sign share, subjects, and descriptive t / Wilcoxon p-values with BY adjustment."""
    from scipy.stats import ttest_1samp, wilcoxon

    rows = []
    for column in ic.columns:
        v = ic[column].dropna().to_numpy()
        mean = float(v.mean()) if len(v) else np.nan
        same = float((np.sign(v) == np.sign(mean)).mean()) if len(v) and mean != 0 else np.nan
        p_t = float(ttest_1samp(v, 0.0).pvalue) if len(v) > 2 and v.std() > 0 else np.nan
        try:
            p_w = float(wilcoxon(v).pvalue) if len(v) > 2 else np.nan
        except ValueError:
            p_w = np.nan
        rows.append({"candidate_id": column, "mean_ic": mean, "abs_mean_ic": abs(mean) if len(v) else np.nan, "same_sign_share": same, "subjects_with_ic": len(v),
                     "ic_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan, "p_t_descriptive": p_t, "p_wilcoxon_descriptive": p_w})
    out = pd.DataFrame(rows)
    out["p_t_by_descriptive"] = benjamini_yekutieli(out["p_t_descriptive"].to_numpy())
    out["p_wilcoxon_by_descriptive"] = benjamini_yekutieli(out["p_wilcoxon_descriptive"].to_numpy())
    return out

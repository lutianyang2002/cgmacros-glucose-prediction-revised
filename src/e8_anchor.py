"""Native-anchor sensitivity E8: the GlucoBench base rebuilt from the inferred native readings only (anchor-causal), and the residual blocks on top of it.

Anchor inference follows `anchor_rule`, nine-anchor lag reconstruction and `anchor_phase_mode`:
  * the anchors are inferred with `factor_construction.native_libre_anchors` in `prefix_causal` mode (phase from kinks at or before t - 15 minutes, nothing after t is read);
  * the 24 base lags (offsets -115 .. 0 minutes, step 5) are rebuilt by holding the last native reading: the lag at t + o is the newest anchor at or before t + o. The oldest lag needs an
    anchor at or before t - 115, i.e. 9 anchors for an anchor age of 0-9 minutes and 8 for 10-14 minutes, so the function is called with `n_anchors = 9`;
  * windows whose level is `uncertain` or whose anchors do not reach back far enough are not reliable: they get no rebuilt lags and are excluded (their share is reported);
  * the base is refitted on the (reliable) training windows only and compared with the original base on the same reliable validation windows, subject-equal RMSE, 20-window rule;
  * E8-b refits the residual ridge blocks of the already selected M, T_k and O_k on anchor-causal residuals (OOF residuals: A five time blocks, B five subject folds), penalty
    re-selected on validation; no factor is reselected.
Nothing here reads the test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import fit_glucobench, predict_glucobench
from src.factor_construction import AnchorRule, native_libre_anchors, split_libre_runs
from src.residuals import contrast_row, grouped_folds, oof_predictions, subject_table, time_block_folds
from src.selection import fit_block

__all__ = ["OFFSETS", "RULE9", "rebuild_lags", "identity_check", "causality_check", "window_report", "e8a_compare", "anchor_residuals", "e8b_blocks"]

OFFSETS = list(range(-115, 1, 5))
RULE9 = AnchorRule(n_anchors=9)
PERIOD = RULE9.period
MIN_WINDOWS = 20


def _subject_anchor_arrays(minutes: pd.DataFrame, rule: AnchorRule = RULE9) -> dict:
    """Level, anchor age and anchor values at every minute of one subject's contiguous frame; minutes outside a continuous Libre run are `uncertain`."""
    n = len(minutes)
    level = np.full(n, "uncertain", dtype=object)
    age = np.full(n, np.nan)
    values = np.full((n, rule.n_anchors), np.nan)
    series = minutes["libre_gl"].to_numpy(float)
    for run in split_libre_runs(minutes):
        rows = native_libre_anchors(series[run], phase_mode="prefix_causal", rule=rule)
        level[run] = rows["level"].to_numpy()
        age[run] = rows["anchor_age"].to_numpy(float)
        values[run] = rows[[f"anchor_value_{j}" for j in range(rule.n_anchors)]].to_numpy(float)
    return {"level": level, "age": age, "values": values}


def rebuild_lags(minutes: dict, windows: pd.DataFrame, offsets: list = OFFSETS, rule: AnchorRule = RULE9) -> tuple:
    """Anchor-causal lags of the windows (`subject`, `origin_time`): (lags DataFrame with the original column names, info DataFrame with level, anchor age and the reliable flag).

    The lag at origin + o (o <= 0 minutes) is the newest anchor at or before it: with anchor age a (the newest anchor lies a minutes before the origin) the anchor index is
    j = max(0, ceil((-o - a) / 15)); j must be below `rule.n_anchors` and its value finite, otherwise the window is not reliable.
    """
    lags = np.full((len(windows), len(offsets)), np.nan)
    level = np.full(len(windows), "uncertain", dtype=object)
    age = np.full(len(windows), np.nan)
    needed = np.zeros(len(windows), int)
    for subject, idx in windows.groupby("subject").indices.items():
        frame = minutes[subject].reset_index(drop=True)
        assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()
        arrays = _subject_anchor_arrays(frame, rule)
        position = ((windows["origin_time"].iloc[idx] - frame["timestamp"].iloc[0]) // pd.Timedelta(minutes=1)).to_numpy()
        a = arrays["age"][position]
        lv = arrays["level"][position]
        vals = arrays["values"][position]
        level[idx], age[idx] = lv, a
        ok = lv != "uncertain"
        for c, o in enumerate(offsets):
            j = np.maximum(0, np.ceil((-o - a) / PERIOD)).astype(float)
            valid = ok & np.isfinite(j) & (j < rule.n_anchors)
            col = np.full(len(idx), np.nan)
            pick = np.flatnonzero(valid)
            col[pick] = vals[pick, j[pick].astype(int)]
            lags[idx, c] = col
            needed[idx] = np.maximum(needed[idx], np.where(np.isfinite(j), j + 1, 0).astype(int))
    columns = [f"g{o}" for o in offsets]
    out = pd.DataFrame(lags, columns=columns, index=windows.index)
    info = pd.DataFrame({"level": level, "anchor_age": age, "anchors_needed": needed, "reliable": np.isfinite(lags).all(axis=1) & (level != "uncertain")}, index=windows.index)
    return out, info


def identity_check(rebuilt: pd.DataFrame, original: pd.DataFrame, info: pd.DataFrame, offsets: list = OFFSETS) -> dict:
    """Where the lag time lies exactly on an anchor the rebuilt lag equals the original lag (the interpolated series passes through the native readings); elsewhere it equals the value of
    an earlier anchor, never a later one. Returns counts and the largest difference."""
    reliable = info["reliable"].to_numpy()
    age = info["anchor_age"].to_numpy(float)
    worst, compared, on_anchor = 0.0, 0, 0
    for c, o in enumerate(offsets):
        aligned = reliable & ((-o - age) % PERIOD == 0) & (-o >= age)
        on_anchor += int(aligned.sum())
        if aligned.any():
            d = np.abs(rebuilt.iloc[:, c].to_numpy()[aligned] - original.iloc[:, c].to_numpy()[aligned])
            worst, compared = max(worst, float(d.max())), compared + int(aligned.sum())
    return {"anchor_aligned_lag_values_compared": compared, "max_abs_difference_on_anchor_aligned_lags": worst}


def causality_check(minutes: dict, windows: pd.DataFrame, sample: int = 120, seed: int = 42, rule: AnchorRule = RULE9) -> dict:
    """For a random sample of windows: rebuild the lags from the Libre series cut at the origin and compare with the rebuild from the full series. Equality in every sampled window shows that no
    minute after the origin enters a rebuilt lag, the level or the reliability flag."""
    rng = np.random.default_rng(seed)
    rows = windows.iloc[np.sort(rng.choice(len(windows), min(sample, len(windows)), replace=False))]
    full, info_full = rebuild_lags(minutes, rows, rule=rule)
    differences, level_equal = [], []
    for k, (idx, row) in enumerate(rows.iterrows()):
        frame = minutes[row["subject"]]
        cut = {row["subject"]: frame[frame["timestamp"] <= row["origin_time"]]}
        part, info_part = rebuild_lags(cut, rows.loc[[idx]], rule=rule)
        both = np.isfinite(part.to_numpy()) & np.isfinite(full.loc[[idx]].to_numpy())
        same_missing = np.array_equal(np.isfinite(part.to_numpy()), np.isfinite(full.loc[[idx]].to_numpy()))
        differences.append(float(np.abs(part.to_numpy()[both] - full.loc[[idx]].to_numpy()[both]).max()) if both.any() else 0.0)
        level_equal.append(bool(same_missing and info_part["level"].iloc[0] == info_full.loc[idx, "level"]))
    return {"windows": len(rows), "max_abs_difference": max(differences), "levels_and_missing_pattern_equal_in_all": all(level_equal)}


def window_report(frame: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Reliable and excluded windows and subjects by split (core cohort and all admitted windows)."""
    rows = []
    for split, part in frame.groupby("split"):
        for label, mask in (("all admitted windows", np.ones(len(part), bool)), ("core cohort", part["cohort_core"].to_numpy(bool))):
            sel = part[mask]
            ok = info.loc[sel.index, "reliable"]
            per_subject = ok.groupby(sel["subject"]).sum()
            rows.append({"split": split, "windows": label, "n": len(sel), "reliable": int(ok.sum()), "excluded": int((~ok).sum()), "excluded_share": float((~ok).mean()), "subjects": sel["subject"].nunique(),
                         "subjects_with_at_least_20_reliable": int((per_subject >= MIN_WINDOWS).sum()), "levels": info.loc[sel.index, "level"].value_counts().to_dict()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E8-a
def e8a_compare(table: pd.DataFrame, rebuilt: pd.DataFrame, info: pd.DataFrame, lag_columns: list, n_resamples: int = 2000) -> dict:
    """Original base (the stored `base_prediction`) against the anchor-causal base on the same reliable core validation windows; the anchor-causal base is fitted on the reliable training windows
    only. A control refits the original lags on those same training windows (separates the effect of the lags from the effect of the smaller fitting set)."""
    train = table["split"].eq("train")
    valid = table["split"].eq("validation") & table["cohort_core"]
    ok = info["reliable"]
    fit_rows = table[train & ok].copy()
    fit_rows[lag_columns] = rebuilt.loc[fit_rows.index, lag_columns]
    model, scaler = fit_glucobench(fit_rows, lag_columns)
    eval_rows = table[valid & ok].copy()
    anchor_pred = predict_glucobench(model, scaler, eval_rows.assign(**{c: rebuilt.loc[eval_rows.index, c] for c in lag_columns}), lag_columns)
    control_model, control_scaler = fit_glucobench(table[train & ok], lag_columns)
    control_pred = predict_glucobench(control_model, control_scaler, eval_rows, lag_columns)
    y = eval_rows["target_value"].to_numpy(float)
    tables = {"original base": subject_table(eval_rows["subject"].to_numpy(), eval_rows["base_prediction"].to_numpy(float) - y),
              "original lags refitted on the reliable training windows (control)": subject_table(eval_rows["subject"].to_numpy(), control_pred - y),
              "anchor-causal base": subject_table(eval_rows["subject"].to_numpy(), anchor_pred - y)}
    rows = []
    for source, target in (("original base", "anchor-causal base"), ("original base", "original lags refitted on the reliable training windows (control)"),
                           ("original lags refitted on the reliable training windows (control)", "anchor-causal base")):
        for rule, minimum in (("applied (>= 20 windows)", MIN_WINDOWS), ("not applied", None)):
            rows.append({"from": source, "to": target, "min_window_rule": rule, **contrast_row(tables[source], tables[target], minimum, n_resamples=n_resamples)})
    all_core_valid = table[valid]
    return {"contrasts": pd.DataFrame(rows), "subject_tables": tables, "fit_windows": len(fit_rows), "fit_subjects": fit_rows["subject"].nunique(), "evaluation_windows": len(eval_rows),
            "evaluation_subjects": eval_rows["subject"].nunique(), "validation_core_windows_total": len(all_core_valid), "excluded_share_of_validation_core": 1 - len(eval_rows) / len(all_core_valid)}


# ------------------------------------------------------------------ E8-b
def anchor_residuals(table: pd.DataFrame, rebuilt: pd.DataFrame, info: pd.DataFrame, lag_columns: list, protocol_is_a: bool, bounds: pd.DataFrame | None) -> pd.DataFrame:
    """Anchor-causal base predictions and residuals (training rows are out-of-fold; validation predictions use the base fitted on reliable training windows)."""
    frame = table[info["reliable"]].copy()
    frame[lag_columns] = rebuilt.loc[frame.index, lag_columns]
    train = frame[frame["split"].eq("train")].copy()
    fold = time_block_folds(train, bounds) if protocol_is_a else grouped_folds(train["subject"])
    train["oof"] = oof_predictions(train, fold, lag_columns)
    model, scaler = fit_glucobench(train, lag_columns)
    valid = frame[frame["split"].eq("validation")].copy()
    out = pd.concat([train, valid])
    out["base_prediction"] = np.nan
    out.loc[train.index, "base_prediction"] = train["oof"]
    out.loc[valid.index, "base_prediction"] = predict_glucobench(model, scaler, valid, lag_columns)
    out["residual"] = out["target_value"] - out["base_prediction"]
    return out


def e8b_blocks(original: pd.DataFrame, anchored: pd.DataFrame, blocks: dict, n_resamples: int = 2000) -> pd.DataFrame:
    """Ridge blocks (columns of the selected M, T_k, O_k) on the original and on the anchor-causal residuals, evaluated on the same reliable core validation windows; penalty re-selected on
    validation for every block and base. `original` holds the main pipeline's `residual` / `base_prediction`, `anchored` the anchor-causal ones."""
    ok_valid = anchored[anchored["split"].eq("validation") & anchored["cohort_core"]]
    keys = ["subject", "origin_time", "task_id"]
    orig_valid = original[original["split"].eq("validation") & original["cohort_core"]].merge(ok_valid[keys], on=keys)
    anch_valid = ok_valid.merge(orig_valid[keys], on=keys)
    orig_train = original[original["split"].eq("train") & original["cohort_core"] & original["residual"].notna()]
    anch_train = anchored[anchored["split"].eq("train") & anchored["cohort_core"] & anchored["residual"].notna()]
    rows, fits = [], {}
    for label, train, valid in (("original lags", orig_train, orig_valid), ("anchor-causal lags", anch_train, anch_valid)):
        for name, cols in {"base": [], **blocks}.items():
            fits[(label, name)] = fit_block(train, valid, cols)
    for name in blocks:
        for rule, minimum in (("applied (>= 20 windows)", MIN_WINDOWS), ("not applied", None)):
            for label in ("original lags", "anchor-causal lags"):
                rows.append({"contrast_type": "marginal effect of the block on the base of the same lags", "lags": label, "block": name, "min_window_rule": rule, "alpha": fits[(label, name)][0],
                             **contrast_row(fits[(label, "base")][1], fits[(label, name)][1], minimum, n_resamples=n_resamples)})
            rows.append({"contrast_type": "same block, lags rebuilt from native readings (original base + block -> anchor-causal base + block)", "lags": "anchor-causal vs original", "block": name,
                         "min_window_rule": rule, "alpha": fits[("anchor-causal lags", name)][0], **contrast_row(fits[("original lags", name)][1], fits[("anchor-causal lags", name)][1], minimum, n_resamples=n_resamples)})
            rows.append({"contrast_type": "total change (original base -> anchor-causal base + block)", "lags": "anchor-causal vs original", "block": name, "min_window_rule": rule,
                         "alpha": fits[("anchor-causal lags", name)][0], **contrast_row(fits[("original lags", "base")][1], fits[("anchor-causal lags", name)][1], minimum, n_resamples=n_resamples)})
    return pd.DataFrame(rows)

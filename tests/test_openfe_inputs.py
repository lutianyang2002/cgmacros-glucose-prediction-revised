"""Checks of the OpenFE inputs on synthetic windows: internal split, `init_scores` isolation and the input audit (no project data).

Run with `python tests/test_openfe_inputs.py` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import lag_columns  # noqa: E402
from src.openfe_inputs import (INTERNAL_TRAIN, INTERNAL_VALIDATION, block_folds_inside, group_folds_inside, input_audit, internal_init_scores, internal_split_subjects,  # noqa: E402
                               internal_split_time)

LAGS = lag_columns(5, 24, 120)


def _windows(n_subjects=6, n_per_subject=500, seed=0):
    """Windows every 5 minutes with a 120-minute history and a 30-minute horizon; the lags drive the target."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2023-01-01")
    parts = []
    for s in range(1, n_subjects + 1):
        origin = t0 + pd.to_timedelta(np.arange(n_per_subject) * 5, unit="min")
        lags = rng.normal(110, 20, (n_per_subject, len(LAGS)))
        frame = pd.DataFrame(lags, columns=LAGS)
        frame["target_value"] = 0.8 * frame[LAGS[-1]] + 0.1 * frame[LAGS[0]] + 12 + rng.normal(0, 5, n_per_subject)
        frame.insert(0, "subject", s)
        frame.insert(1, "origin_time", origin)
        frame["history_start_time"] = origin - pd.Timedelta(minutes=120)
        frame["target_time"] = origin + pd.Timedelta(minutes=30)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _bounds(windows):
    start = windows.groupby("subject")["history_start_time"].min()
    end = windows.groupby("subject")["target_time"].max()
    return start, end


def test_time_split_keeps_whole_supports_on_one_side():
    windows = _windows()
    start, end = _bounds(windows)
    role, cut = internal_split_time(windows, start, end)
    role = pd.Series(role)
    assert set(role) == {INTERNAL_TRAIN, INTERNAL_VALIDATION, "dropped"}
    for s, g in windows.groupby("subject"):
        r = role[g.index]
        assert g.loc[r == INTERNAL_TRAIN, "target_time"].max() <= cut[s] <= g.loc[r == INTERNAL_VALIDATION, "history_start_time"].min()
        # a window that crosses the cut is dropped: its support starts before and ends after the cut
        crossing = g[(g["history_start_time"] < cut[s]) & (g["target_time"] > cut[s])]
        assert (r[crossing.index] == "dropped").all() and len(crossing) > 0


def test_subject_split_rejects_overlap_and_unknown_subjects():
    windows = _windows()
    role = internal_split_subjects(windows, [1, 2, 3, 4], [5, 6])
    assert (role[windows["subject"].isin([5, 6])] == INTERNAL_VALIDATION).all()
    for train, valid in (([1, 2, 3, 4], [4, 5, 6]), ([1, 2, 3], [5, 6])):
        try:
            internal_split_subjects(windows, train, valid)
        except ValueError:
            continue
        raise AssertionError("overlapping or incomplete membership accepted")


def _scores_time(windows):
    start, end = _bounds(windows)
    role, cut = internal_split_time(windows, start, end)
    folds = block_folds_inside(windows[role == INTERNAL_TRAIN], start, cut)
    return role, folds, internal_init_scores(windows, role, LAGS, folds)


def test_time_split_init_scores_do_not_see_internal_validation_rows():
    windows = _windows()
    role, folds, init = _scores_time(windows)
    is_validation = role == INTERNAL_VALIDATION
    assert np.isnan(init[role == "dropped"]).all() and np.isfinite(init[is_validation]).all()
    assert np.isnan(init[(role == INTERNAL_TRAIN)][folds < 0]).all() and (folds < 0).sum() > 0        # windows on a block edge have no score
    noisy = windows.copy()
    rng = np.random.default_rng(5)
    noisy.loc[is_validation, "target_value"] += rng.normal(0, 400, is_validation.sum())
    noisy.loc[is_validation, LAGS] += rng.normal(0, 400, (is_validation.sum(), len(LAGS)))
    _, _, init_noisy = _scores_time(noisy)
    assert np.array_equal(init[role == INTERNAL_TRAIN], init_noisy[role == INTERNAL_TRAIN], equal_nan=True)      # internal-train scores: unchanged
    only_labels = windows.copy()
    only_labels.loc[is_validation, "target_value"] += rng.normal(0, 400, is_validation.sum())
    _, _, init_labels = _scores_time(only_labels)
    assert np.array_equal(init[is_validation], init_labels[is_validation])                                        # validation scores never depend on validation labels


def test_group_split_init_scores_do_not_see_internal_validation_subjects():
    windows = _windows(n_subjects=9, n_per_subject=120)
    role = internal_split_subjects(windows, [1, 2, 3, 4, 5, 6, 7], [8, 9])
    folds = group_folds_inside(windows[role == INTERNAL_TRAIN])
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]
    members = [set(windows[role == INTERNAL_TRAIN][folds == k]["subject"]) for k in range(5)]
    assert sum(len(m) for m in members) == len(set().union(*members)) == 7                                      # folds hold disjoint subjects
    init = internal_init_scores(windows, role, LAGS, folds)
    noisy = windows.copy()
    mask = role == INTERNAL_VALIDATION
    noisy.loc[mask, "target_value"] += 500
    noisy.loc[mask, LAGS] += 500
    assert np.array_equal(init[role == INTERNAL_TRAIN], internal_init_scores(noisy, role, LAGS, folds)[role == INTERNAL_TRAIN])


def test_isolation_check_fails_for_a_leaky_implementation():
    """Negative control: scores from a base fitted on all rows (internal-validation rows included) change when the internal-validation labels change."""
    from src.baselines import fit_glucobench, predict_glucobench

    windows = _windows()
    start, end = _bounds(windows)
    role, _ = internal_split_time(windows, start, end)

    def leaky(frame):
        model, scaler = fit_glucobench(frame, LAGS)
        return predict_glucobench(model, scaler, frame, LAGS)

    noisy = windows.copy()
    mask = role == INTERNAL_VALIDATION
    noisy.loc[mask, "target_value"] += np.random.default_rng(1).normal(0, 400, mask.sum())
    assert not np.array_equal(leaky(windows)[role == INTERNAL_TRAIN], leaky(noisy)[role == INTERNAL_TRAIN])


def test_input_audit_excludes_leaky_or_unstated_columns():
    manual = ["act_hr_mean_30m", "act_met_mean_30m", "diet_last_meal_carbs", "bio_bmi"]
    primitives = ["subject", "origin_time", "task_id", "split", "act_hr_mean_30m", "act_hr_mean_60m", "act_hr_coverage_60m", "act_met_mean_30m_lag10", "act_met_missing_30m_lag10",
                  "meal_lunch_logged_carbs", "meal_lunch_delay_all_carbs", "native_last", "native_missing", "native_anchor_3", "native_level"]
    audit = input_audit(manual, primitives).set_index("column")
    included = set(audit.index[audit["included"]])
    assert included == {"act_hr_mean_30m", "diet_last_meal_carbs", "bio_bmi", "act_hr_mean_60m", "act_met_mean_30m_lag10", "meal_lunch_logged_carbs", "native_last", "native_missing"}
    for column in ("subject", "origin_time", "target_value", "residual", "base_prediction", "bmi_kg_m2"):
        assert column not in included


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception as error:      # noqa: BLE001
            failures += 1
            print("FAIL", name, type(error).__name__, error)
    sys.exit(1 if failures else 0)

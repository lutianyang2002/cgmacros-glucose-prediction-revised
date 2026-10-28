"""Checks of the training-split fitness of the LLM search on synthetic windows: isolation of the held-out part, sign and scale of the gain, rule for thin subjects.

Run with `python tests/test_llm_fitness.py` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import llm_factor_search as L  # noqa: E402
from src.baselines import lag_columns  # noqa: E402
from src.llm_fitness import ExpressionFitness, build_contexts, group_splits, split_cv_gain, time_split  # noqa: E402
from test_llm_factor_search import _subject  # noqa: E402

LAGS = lag_columns(5, 24, 120)


def _frame(n_subjects=9, n_per_subject=500, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2023-01-01")
    parts = []
    for s in range(1, n_subjects + 1):
        origin = t0 + pd.to_timedelta(np.arange(n_per_subject) * 5, unit="min")
        frame = pd.DataFrame(rng.normal(110, 20, (n_per_subject, len(LAGS))), columns=LAGS)
        frame["target_value"] = 0.8 * frame[LAGS[-1]] + 0.1 * frame[LAGS[0]] + 12 + rng.normal(0, 5, n_per_subject)
        frame.insert(0, "subject", s)
        frame.insert(1, "origin_time", origin)
        frame["history_start_time"], frame["target_time"] = origin - pd.Timedelta(minutes=120), origin + pd.Timedelta(minutes=30)
        frame["cohort_core"], frame["split"], frame["native_missing"] = True, "train", 0
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _bounds(frame):
    return frame.groupby("subject")["history_start_time"].min(), frame.groupby("subject")["target_time"].max()


def test_time_split_residuals_of_the_fitting_part_do_not_see_the_held_out_part():
    frame = _frame()
    start, end = _bounds(frame)
    split = time_split(frame, start, end, LAGS)
    assert len(split.train) > 0 and len(split.held) > 0 and not set(split.train) & set(split.held)
    noisy = frame.copy()
    rows = np.zeros(len(frame), bool)
    rows[split.held] = True
    rng = np.random.default_rng(2)
    noisy.loc[rows, "target_value"] += rng.normal(0, 300, rows.sum())
    noisy.loc[rows, LAGS] += rng.normal(0, 300, (rows.sum(), len(LAGS)))
    other = time_split(noisy, start, end, LAGS)
    assert np.array_equal(split.residual[split.train], other.residual[other.train])          # fitting residuals: bit-identical
    assert not np.array_equal(split.residual[split.held], other.residual[other.held])        # the held-out residual does follow its own label
    assert frame["history_start_time"].iloc[split.held].min() >= frame["target_time"].iloc[split.train].max() - pd.Timedelta(days=3)


def test_group_splits_hold_out_whole_subjects_and_fit_the_baseline_without_them():
    frame = _frame()
    splits = group_splits(frame, LAGS)
    held_subjects = [set(frame["subject"].iloc[s.held]) for s in splits]
    assert len(splits) == 3 and sum(len(h) for h in held_subjects) == frame["subject"].nunique() == len(set().union(*held_subjects))
    for index, (sp, held) in enumerate(zip(splits, held_subjects)):
        assert not held & set(frame["subject"].iloc[sp.train])
        noisy = frame.copy()
        mask = frame["subject"].isin(held).to_numpy()
        noisy.loc[mask, "target_value"] += 500
        noisy.loc[mask, LAGS] += 500
        again = group_splits(noisy, LAGS)[index]
        assert np.array_equal(sp.residual[sp.train], again.residual[again.train])


def test_gain_is_positive_for_a_real_signal_negative_for_a_harmful_one_and_zero_for_noise():
    frame = _frame()
    start, end = _bounds(frame)
    split = time_split(frame, start, end, LAGS)
    subjects = frame["subject"].to_numpy()
    residual = split.residual
    x_good = np.full(len(frame), np.nan)
    x_good[split.train], x_good[split.held] = residual[split.train], residual[split.held]
    gain = split_cv_gain(x_good + np.random.default_rng(0).normal(0, 2, len(frame)), [split], subjects, 1.0)
    assert gain is not None and gain > 0.2                                                    # higher fitness = lower held-out error
    x_bad = x_good.copy()
    x_bad[split.held] = -residual[split.held]                                                 # the relation reverses on the held-out part
    assert split_cv_gain(x_bad, [split], subjects, 1.0) < -0.2                                # a harmful factor has a negative fitness
    noise = np.random.default_rng(1).normal(size=len(frame))
    assert abs(split_cv_gain(noise, [split], subjects, 1.0)) < 0.1


def test_subjects_with_few_held_out_windows_do_not_count_and_missing_factors_cannot_be_scored():
    frame = _frame()
    splits = group_splits(frame, LAGS)
    subjects = frame["subject"].to_numpy()
    x = np.random.default_rng(3).normal(size=len(frame))
    target = subjects[splits[0].held][0]
    sp = splits[0]
    keep_without = sp.held[subjects[sp.held] != target]
    three = sp.held[subjects[sp.held] == target][:3]
    wild = np.where(np.isin(np.arange(len(sp.residual)), three), 1e6, sp.residual)                 # three windows of one subject with an absurd residual
    thin = type(sp)(sp.train, np.r_[keep_without, three], wild)
    without = type(sp)(sp.train, keep_without, sp.residual)
    assert split_cv_gain(x, [thin], subjects, 1.0) == split_cv_gain(x, [without], subjects, 1.0)    # the subject below 20 windows is ignored
    assert split_cv_gain(x, [thin], subjects, 1.0, min_held_windows=1) != split_cv_gain(x, [without], subjects, 1.0)
    assert split_cv_gain(np.where(np.arange(len(x)) < 10, x, np.nan), splits, subjects, 1.0) is None  # fewer than 50 defined training values
    assert split_cv_gain(x, splits, subjects, 1.0, min_held_windows=10 ** 9) is None


def test_fitness_refuses_validation_rows():
    frame = _frame()
    splits = group_splits(frame, LAGS)
    bad = frame.copy()
    bad.loc[0, "split"] = "validation"
    try:
        ExpressionFitness(bad, splits, {}, 1.0)
    except ValueError:
        return
    raise AssertionError("a frame with a validation row was accepted")


def test_cutting_minutes_at_the_training_end_does_not_change_values_before_it():
    minutes, events, origins = _subject()
    frame = pd.DataFrame({"subject": 1, "origin_time": origins.to_numpy()})
    cut = pd.Series({1: origins.iloc[40]})
    bmi = pd.Series({1: 27.5})
    empty = events.iloc[:0]
    full = build_contexts(frame, {1: minutes}, {1: events}, bmi, empty)
    cutted = build_contexts(frame.iloc[:41], {1: minutes}, {1: events}, bmi, empty, cut)
    for text in ('ts_mean("hr", 60)', 'conv_gamma("any", "carbs", 30, 2)', 'native("slope_1h")', 'minutes_since_meal("lunch")'):
        node = L.parse_expression(text)
        a = L.evaluate_expression(node, full[1][1])[:41]
        b = L.evaluate_expression(node, cutted[1][1])
        assert np.array_equal(a, b, equal_nan=True), text


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception as error:      # noqa: BLE001
            import traceback
            failures += 1
            print("FAIL", name, type(error).__name__, error)
            traceback.print_exc()
    sys.exit(1 if failures else 0)

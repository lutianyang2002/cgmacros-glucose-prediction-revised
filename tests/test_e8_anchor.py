"""Checks of the E8 anchor-causal lag rebuild on a synthetic Libre series (no data files, no model).

Run with `python tests/test_e8_anchor.py` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import e8_anchor as E  # noqa: E402

PHASE, N = 7, 3000


def _subject(seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(PHASE, N, 15)
    readings = 120 + np.cumsum(rng.normal(scale=6, size=len(times)))
    x = np.interp(np.arange(N), times, readings)                                  # the released series: linear interpolation between native readings
    frame = pd.DataFrame({"timestamp": pd.Timestamp("2021-03-01") + pd.to_timedelta(np.arange(N), unit="m"), "libre_gl": x, "source_segment_id": 1, "is_inserted_grid_row": False})
    return frame, x


def _windows(frame, positions):
    return pd.DataFrame({"subject": 1, "origin_time": frame["timestamp"].iloc[positions].to_numpy()}, index=range(len(positions)))


def test_the_rebuilt_lags_hold_the_newest_native_reading_at_or_before_each_lag_time():
    frame, x = _subject()
    positions = np.arange(200, 2900, 5)
    windows = _windows(frame, positions)
    lags, info = E.rebuild_lags({1: frame}, windows)
    assert info["reliable"].all() and set(info["level"]) <= {"confirmed", "grid"}
    for c, o in enumerate(E.OFFSETS):
        t = positions + o
        expected = x[np.floor((t - PHASE) / 15).astype(int) * 15 + PHASE]
        assert np.allclose(lags.iloc[:, c].to_numpy(), expected), o
    assert (info["anchors_needed"] <= 9).all() and info["anchors_needed"].max() == 9


def test_early_windows_without_enough_anchors_are_not_reliable():
    frame, _ = _subject()
    windows = _windows(frame, np.arange(0, 60, 5))
    lags, info = E.rebuild_lags({1: frame}, windows)
    assert not info["reliable"].any() and lags.isna().any(axis=1).all()
    report = E.window_report(pd.DataFrame({"subject": 1, "split": "validation", "cohort_core": True}, index=windows.index), info)
    assert report["excluded_share"].eq(1.0).all()


def test_identity_on_anchor_times_and_no_dependence_on_later_minutes():
    frame, x = _subject()
    positions = np.arange(202, 2900, 5)                                          # origins on the 5-minute grid whose phase matches the anchors (anchor ages 0, 5, 10)
    windows = _windows(frame, positions)
    lags, info = E.rebuild_lags({1: frame}, windows)
    original = pd.DataFrame({f"g{o}": x[positions + o] for o in E.OFFSETS}, index=windows.index)
    check = E.identity_check(lags, original, info)
    assert check["anchor_aligned_lag_values_compared"] > 1000 and check["max_abs_difference_on_anchor_aligned_lags"] < 1e-9
    causal = E.causality_check({1: frame}, windows, sample=40)
    assert causal["max_abs_difference"] < 1e-9 and causal["levels_and_missing_pattern_equal_in_all"]
    spoiled = frame.copy()
    spoiled.loc[1500:, "libre_gl"] = 999.0                                         # nothing after minute 1500 may reach a window at or before it
    early = windows[windows["origin_time"] <= frame["timestamp"].iloc[1500]]
    again, _ = E.rebuild_lags({1: spoiled}, early)
    base, _ = E.rebuild_lags({1: frame}, early)
    assert np.allclose(again.to_numpy(), base.to_numpy(), equal_nan=True)


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

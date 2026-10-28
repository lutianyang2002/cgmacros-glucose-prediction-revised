"""Synthetic checks of the bootstrap p-value, the percentile interval and the Holm step-down of `src/factor_evaluation.py`, and the train-only fit of the evaluator (fully synthetic data; no test data, no model files).

Run with `python tests/test_bootstrap_holm.py` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import factor_evaluation as FE  # noqa: E402
from src.evaluation import draw_subjects, paired_cluster_bootstrap, paired_inputs  # noqa: E402
from src.residuals import contrast_row  # noqa: E402

SAVED = ROOT / "results" / "factor_evaluation" / "comparison_results.csv"
N, B, SEED = 30, 2000, 42


def bootstrap_p(first, second, minimum, n_resamples=2000, seed=42):
    return contrast_row(first, second, minimum, n_resamples, seed, include_p=True)["p_bootstrap"]


def tables(d, base=5.0):
    """Per-subject tables whose paired RMSE differences (to minus from) are exactly `d`; every subject has N windows."""
    d = np.asarray(d, float)
    idx = pd.Index(np.arange(1, len(d) + 1), name="subject")

    def one(rmse):
        return pd.DataFrame({"n_windows": N, "rmse": rmse, "mae": 0.8 * rmse, "sum_squared_error": N * rmse ** 2, "sum_abs_error": N * 0.8 * rmse}, index=idx)
    return one(np.full(len(d), base)), one(base + d)


def reference_p(d, n_resamples=B, seed=SEED):
    """Independent re-implementation of the documented formula."""
    d = np.asarray(d, float)
    draws = np.random.default_rng(seed).integers(0, len(d), size=(n_resamples, len(d)))
    means = d[draws].mean(axis=1)
    k = min(int((means >= 0).sum()), int((means <= 0).sum()))
    return min(1.0, 2 * (k + 1) / (n_resamples + 1)), means


def test_the_p_value_follows_the_documented_formula():
    rng = np.random.default_rng(1)
    for n in (9, 20, 44):
        for shift in (-1.0, -0.3, 0.0, 0.4):
            d = rng.normal(shift, 1.0, n)
            t_from, t_to = tables(d)
            assert abs(bootstrap_p(t_from, t_to, None) - reference_p(d)[0]) < 1e-12
    # the smallest attainable value and the cap
    assert abs(bootstrap_p(*tables([-1.0] * 9), None) - 2 / 2001) < 1e-12 and abs(bootstrap_p(*tables([1.0] * 9), None) - 2 / 2001) < 1e-12
    assert bootstrap_p(*tables([0.0] * 9), None) == 1.0
    # a different number of resamples changes the floor: 2 / (n_resamples + 1)
    assert abs(bootstrap_p(*tables([-1.0] * 9), None, n_resamples=200) - 2 / 201) < 1e-12


def test_zero_effect_all_zero_differences_and_ties():
    t_from, t_to = tables([0.0] * 9)
    assert bootstrap_p(t_from, t_to, None) == 1.0
    row = contrast_row(t_from, t_to, None, n_resamples=B, seed=SEED)
    assert row["d_rmse"] == 0 and row["d_rmse_low"] == 0 and row["d_rmse_high"] == 0 and row["outcome"] == "negligible"
    d = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], float)               # symmetric around exactly 0: the observed mean is 0
    assert bootstrap_p(*tables(d), None) > 0.9
    # ties: half of the subjects have a difference of exactly 0 (resamples with mean 0 are counted on both sides)
    d = np.array([0, 0, 0, 0, -1, -1, -1, -1, -1], float)
    p, means = reference_p(d)
    assert (means == 0).sum() >= 1 and abs(bootstrap_p(*tables(d), None) - p) < 1e-12
    assert p > 2 / 2001                                                  # the tied resamples raise the p-value above the floor
    # identical differences of any sign: a degenerate bootstrap distribution gives the floor even for two subjects (documented limit)
    assert abs(bootstrap_p(*tables([-1.0, -1.0]), None) - 2 / 2001) < 1e-12


def test_clear_effects_and_symmetry():
    rng = np.random.default_rng(2)
    d = rng.normal(-1.0, 0.5, 20)
    p_neg = bootstrap_p(*tables(d), None)
    p_pos = bootstrap_p(*tables(-d), None)
    assert p_neg < 0.01 and p_neg == p_pos
    # swapping the two configurations (to <-> from) flips the sign and leaves the p-value unchanged
    t_from, t_to = tables(d)
    assert bootstrap_p(t_to, t_from, None) == p_neg
    low, high = paired_cluster_bootstrap(paired_inputs(t_from, t_to), B, SEED)["d_rmse"][1:]
    assert high < 0
    low2, high2 = paired_cluster_bootstrap(paired_inputs(t_to, t_from), B, SEED)["d_rmse"][1:]
    assert abs(low2 + high) < 1e-12 and abs(high2 + low) < 1e-12        # the interval mirrors, no direction bias


def test_the_interval_is_the_percentile_interval_of_the_same_resamples_as_the_p_value():
    rng = np.random.default_rng(3)
    d = rng.normal(-0.2, 1.0, 9)
    p, means = reference_p(d)
    t_from, t_to = tables(d)
    est, low, high = paired_cluster_bootstrap(paired_inputs(t_from, t_to), B, SEED)["d_rmse"]
    assert abs(est - d.mean()) < 1e-12
    assert abs(low - np.percentile(means, 2.5)) < 1e-12 and abs(high - np.percentile(means, 97.5)) < 1e-12     # the same index draws (seed 42) feed the interval and the p-value
    assert (draw_subjects(9, B, SEED) == draw_subjects(9, B, SEED)).all()
    row = contrast_row(t_from, t_to, None, n_resamples=B, seed=SEED)
    assert row["d_rmse_low"] == low and row["d_rmse_high"] == high


def test_the_twenty_window_rule_is_applied_before_the_resampling():
    d = np.array([-1.0] * 6 + [1.0] * 3)
    t_from, t_to = tables(d)
    t_from.loc[7:9, "n_windows"] = 5
    t_to.loc[7:9, "n_windows"] = 5                                       # the three positive subjects fall below 20 windows
    assert abs(bootstrap_p(t_from, t_to, 20) - 2 / 2001) < 1e-12
    assert bootstrap_p(t_from, t_to, None) > 0.05


def test_holm_matches_the_textbook_algorithm_and_statsmodels():
    for p in ([0.01, 0.04, 0.03, 0.005], [0.5, 0.01], [0.2, 0.2, 0.2, 0.2], [0.001] * 4, [0.9, 0.95, 0.99, 1.0], [0.0125, 0.0125, 0.03, 0.2]):
        adjusted = FE.holm(p)
        order = np.argsort(p)
        expected = np.empty(len(p))
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(p) - rank) * p[i]))
            expected[i] = running
        assert np.allclose(adjusted, expected)
        assert all(a >= x - 1e-15 for a, x in zip(adjusted, p)) and max(adjusted) <= 1.0
        # a permutation of the input permutes the output (ties included)
        perm = np.random.default_rng(0).permutation(len(p))
        assert np.allclose(np.array(FE.holm([p[i] for i in perm])), np.array(adjusted)[perm])
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:                                             # the check against the library is optional
            continue
        assert np.allclose(adjusted, multipletests(p, method="holm")[1])
    assert FE.holm([0.2, 0.2, 0.2, 0.2]) == [0.8] * 4 and FE.holm([0.6, 0.6, 0.6, 0.6]) == [1.0] * 4
    # the rank-one p-value is multiplied by the family size, the smallest bootstrap p-value of four gives 4 * 2 / 2001
    assert abs(FE.holm([2 / 2001, 0.5, 0.5, 0.5])[0] - 8 / 2001) < 1e-15


def test_the_saved_primary_holm_family_is_four_per_protocol():
    frame = pd.read_csv(SAVED)
    for tag, group in frame.groupby("protocol_tag"):
        family = group[group["p_holm_primary_family"].notna()]
        assert len(family) == 4 and set(family["id"].str.split(":").str[1]) == {"P1", "P2", "P3", "P4"}
        assert (family["cohort"] == "core").all() and (family["stratum"] == "all").all() and (family["group"] == "all").all() and family["min_window_rule"].str.startswith("applied").all()
        assert np.allclose(family["p_holm_primary_family"].to_numpy(), FE.holm(family["p_bootstrap"].tolist()))
        # nothing else carries a Holm value, and the p-value exists only on the all-window/all-subject rows
        assert group["p_holm_primary_family"].notna().sum() == 4
    assert frame[frame["p_bootstrap"].notna()][["stratum", "group"]].drop_duplicates().values.tolist() == [["all", "all"]]


def _split_table(rows=60, subjects=6, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(1, subjects + 1):
        x = rng.normal(size=(rows, 2))
        base = 120 + rng.normal(size=rows) * 10
        target = base + 2 * x[:, 0] + rng.normal(size=rows)
        split = np.repeat(["train", "validation", "test"], rows // 3)
        frames.append(pd.DataFrame({"subject": s, "split": split, "origin_time": pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(rows) * 5, unit="m"), "task_id": 1, "x0": x[:, 0], "x1": x[:, 1],
                                    "base_prediction": base, "target_value": target, "residual": target - base, "cohort_core": True}))
    return pd.concat(frames, ignore_index=True)


def test_fits_use_the_training_split_only_whatever_the_evaluation_split():
    configs = {"base": {"columns": [], "alpha": None}, "base + X": {"columns": ["x0", "x1"], "alpha": 1.0}}
    table = _split_table()
    for eval_split, other in (("validation", "test"), ("test", "validation")):
        reference = FE.Evaluator(table, configs, [], {}, eval_split).errors_of("base + X")
        assert set(reference["subject"]) == set(table["subject"]) and len(reference) == (table["split"] == eval_split).sum()
        spoiled = table.copy()
        mask = spoiled["split"] == other                                   # the split that is NOT evaluated: labels, residuals and features are destroyed
        spoiled.loc[mask, ["target_value", "residual", "x0", "x1", "base_prediction"]] = 1e6
        again = FE.Evaluator(spoiled, configs, [], {}, eval_split).errors_of("base + X")
        assert np.array_equal(reference["error"].to_numpy(), again["error"].to_numpy()), eval_split      # not merged into any fit, not evaluated
        # the evaluation split's own labels do not enter the fit either (only the error's target term moves)
        spoiled2 = table.copy()
        spoiled2.loc[spoiled2["split"] == eval_split, ["target_value", "residual"]] = 999.0
        third = FE.Evaluator(spoiled2, configs, [], {}, eval_split).errors_of("base + X")
        moved = third["error"].to_numpy() - reference["error"].to_numpy()
        ev = table[table["split"] == eval_split]
        assert np.allclose(moved, ev["target_value"].to_numpy() - 999.0)
    # the full cohort refits on the training windows of that cohort only
    table["cohort_core"] = np.tile([True, False], len(table) // 2)
    full = FE.Evaluator(table, configs, [], {}, "validation").errors_of("base + X", "full")
    assert len(full) == (table["split"] == "validation").sum()



def test_empty_and_nonfinite_statistics_have_explicit_contracts():
    from src.evaluation import evaluate
    from src.residuals import subject_table, ridge_path_predictions
    from unittest import TestCase
    case = TestCase()
    empty = subject_table(np.array([], dtype=int), np.array([]))
    assert empty.empty and list(empty) == ["n_windows", "sum_abs_error", "sum_squared_error", "mae", "rmse"]
    with case.assertRaises(ValueError):
        evaluate(pd.DataFrame({"subject": [], "target_value": []}), np.array([]))
    for invalid in (np.nan, np.inf):
        with case.assertRaises(ValueError):
            evaluate(pd.DataFrame({"subject": [1], "target_value": [0.]}), np.array([invalid]))
        with case.assertRaises(ValueError):
            subject_table([1], np.array([invalid]))
        with case.assertRaises(ValueError):
            ridge_path_predictions(pd.DataFrame({"x": [1.]}), np.array([invalid]), pd.DataFrame({"x": [1.]}))
        with case.assertRaises(ValueError):
            FE.holm([0.01, 0.02, 0.03, invalid])
    with case.assertRaises(ValueError):
        paired_cluster_bootstrap(paired_inputs(empty, empty))
    one, _ = tables([0.])
    result = paired_cluster_bootstrap(paired_inputs(one, one), 20)
    assert result["d_rmse"] == (0., 0., 0.)
    for p in (-0.01, 1.01):
        with case.assertRaises(ValueError): FE.holm([p])
    assert FE.holm([]) == []
    one.loc[1, "n_windows"] = 0
    with case.assertRaises(ValueError): paired_cluster_bootstrap(paired_inputs(one, one), 20)


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

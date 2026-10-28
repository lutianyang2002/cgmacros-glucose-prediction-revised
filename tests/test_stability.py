"""Checks of the development-data analysis stability code on synthetic data (no data files, no model).

Run with `python tests/test_stability.py` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import stability as S  # noqa: E402
from src.evaluation import subject_ic  # noqa: E402


def _data(n_subjects=24, rows=120, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
    frames = []
    for s in range(1, n_subjects + 1):
        x = rng.normal(size=(rows, len(names)))
        r = 1.0 * x[:, 0] - 0.8 * x[:, 5] + 0.3 * rng.normal(size=rows)          # a0 and b0 carry the signal
        frames.append(pd.DataFrame(x, columns=names).assign(subject=s, residual=r))
    frame = pd.concat(frames, ignore_index=True)
    ic, _ = subject_ic(frame, names)
    ic.index = sorted(frame["subject"].unique())
    family = pd.Series({c: ("A" if c.startswith("a") else "B") for c in names})
    return frame, ic, family, names


def test_the_family_pool_keeps_the_largest_absolute_mean_ic_per_family():
    frame, ic, family, names = _data()
    pool = S.family_pool(ic, family, k_per_family=2)
    assert len(pool) == 4 and {"a0", "b0"} <= set(pool) and sum(c.startswith("a") for c in pool) == 2


def test_the_lasso_path_returns_the_true_signals_first():
    frame, ic, family, names = _data()
    picked = S.lasso_first_q(frame[names].to_numpy(), frame["residual"].to_numpy(), q=2)
    assert {names[i] for i in picked} == {"a0", "b0"}


def test_the_selection_reads_only_the_rows_of_its_subjects():
    frame, ic, family, names = _data()
    subjects = list(range(1, 13))
    first = S.select_ic_lasso(frame, ic, subjects, family, k_per_family=3, q=3)
    other = ~frame["subject"].isin(subjects)
    spoiled = frame.copy()
    spoiled.loc[other, names + ["residual"]] = np.random.default_rng(1).normal(size=(int(other.sum()), len(names) + 1)) * 1000
    spoiled_ic = ic.copy()
    spoiled_ic.loc[~spoiled_ic.index.isin(subjects)] = 5.0
    assert S.select_ic_lasso(spoiled, spoiled_ic, subjects, family, k_per_family=3, q=3) == first
    assert "a0" in first and "b0" in first


def test_complementary_pairs_are_disjoint_complete_and_reproducible():
    subjects = list(range(1, 28))
    pairs = S.complementary_pairs(subjects, 50, 42)
    assert len(pairs) == 50 and pairs == S.complementary_pairs(subjects, 50, 42) and pairs != S.complementary_pairs(subjects, 50, 7)
    for left, right in pairs:
        assert not set(left) & set(right) and set(left) | set(right) == set(subjects) and len(left) == 13 and len(right) == 14
    assert len({tuple(p[0]) for p in pairs}) == 50


def test_frequency_jaccard_and_loso():
    freq, sets = S.selection_frequency([["a", "b"], ["a", "c"], ["a", "b"], ["b", "c"]], ["a", "b", "c", "d"])
    assert freq["a"] == 0.75 and freq["d"] == 0.0 and sets[0.7] == ["a", "b"] and sets[0.5] == ["a", "b", "c"]
    assert S.jaccard(["a", "b"], ["b", "c"]) == 1 / 3 and S.jaccard([], []) == 1.0
    out = S.loso_jaccard(lambda rest: ["a", "b"] if len(rest) > 2 else ["a"], [1, 2, 3, 4], ["a", "b"])
    assert len(out) == 4 and (out["jaccard_with_full"] == 1.0).all()


def test_group_stability_flags_a_sign_flip_between_groups():
    rng = np.random.default_rng(3)
    rows = []
    for s in range(1, 10):
        x = rng.normal(size=(200, 2))
        sign = 1 if s <= 6 else -1                                               # factor f1 flips its sign in the third group (subjects 7-9)
        r = x[:, 0] + (sign * x[:, 1] if True else 0) + 0.2 * rng.normal(size=200)
        rows.append(pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "residual": r, "subject": s}))
    frame = pd.concat(rows, ignore_index=True)
    ic, _ = subject_ic(frame, ["f0", "f1"])
    ic.index = sorted(frame["subject"].unique())
    group_of = pd.Series(["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3, index=range(1, 10))
    table = S.group_stability(frame, ic, ["f0", "f1"], group_of, alpha=1.0)
    stable = table.drop_duplicates("factor").set_index("factor")["stable"]
    assert bool(stable["f0"]) and not bool(stable["f1"])
    assert S.metabolic_group(pd.Series([5.0, 5.7, 6.4, 6.5])).tolist() == ["normal (<5.7)", "prediabetes (5.7-6.4)", "prediabetes (5.7-6.4)", "diabetes (>6.4)"]


def test_empty_group_is_not_reported_as_a_sign_disagreement():
    from unittest import TestCase
    frame, ic, _, _ = _data(n_subjects=3, rows=20)
    groups = pd.Series(pd.Categorical(["present"] * 3, categories=["present", "absent"]), index=[1, 2, 3])
    table = S.group_stability(frame, ic, ["a0"], groups, alpha=1.)
    case = TestCase()
    case.assertTrue(table.loc[table.group.eq("absent"), "ridge_coefficient"].isna().all())
    case.assertTrue(table["stable"].isna().all())
    case.assertTrue(S.group_stability(frame, ic, [], groups, alpha=1.).empty)


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

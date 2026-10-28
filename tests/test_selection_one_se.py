"""Deterministic checks of the paired one-standard-error rule in `src.selection.one_standard_error_choice`.

Every expected value below was computed by hand (see the comment of each case) before the function was run.
Run with `python tests/test_selection_one_se.py` (or with pytest, if installed).

Rule under test (post-hoc clarification C1 of theory-factor discovery): for each k, d_k = RMSE_k - RMSE_best per subject; the standard error is
sd(d_k, ddof=1) / sqrt(n); the selected k is the smallest k with mean(d_k) <= SE(d_k). The best point is the first minimum of the mean curve.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selection import one_standard_error_choice  # noqa: E402

SUBJECTS = [1, 2, 3, 4]


def build(matrix, subjects=None):
    """matrix: {k: per-subject RMSE list}; returns (curve, tables) in the shape that `forward_selection` produces."""
    subjects = subjects or SUBJECTS
    tables = {k: pd.DataFrame({"rmse": pd.Series(v, index=pd.Index(subjects, name="subject"), dtype=float)}) for k, v in matrix.items()}
    curve = pd.DataFrame({"k": list(matrix), "subject_equal_rmse": [float(np.mean(v)) for v in matrix.values()]})
    return curve, tables


def shifted(base, shift):
    return [b + s for b, s in zip(base, shift)]


def test_best_k_positive_with_exact_and_just_failing_boundaries():
    # best vector B (k=3) = [9, 11, 10, 10]: mean 10; deviations [-1, 1, 0, 0]; sum of squares 2; variance 2/3; unpaired SE = sqrt(2/3)/2 = 0.408248
    B = [9, 11, 10, 10]
    # k=0: d=[6,2,2,6]  mean 4;     deviations [2,-2,-2,2], sum of squares 16, variance 16/3, sd 2.309401, SE 1.154701  -> 4 > 1.154701: fails clearly
    # k=1: d=[4,0,0,.3] mean 1.075; deviations [2.925,-1.075,-1.075,-.775], sum of squares 11.4675, variance 3.8225, sd 1.955121, SE 0.977561 -> fails by 0.097
    # k=2: d=[4,0,0,0]  mean 1;     deviations [3,-1,-1,-1], sum of squares 12, variance 4, sd 2, SE 1 -> 1 <= 1: satisfies exactly
    # k=4: d=[2,2,2,2]  mean 2, SE 0; k=5: d=[3,3,3,3] mean 3, SE 0
    matrix = {0: shifted(B, [6, 2, 2, 6]), 1: shifted(B, [4, 0, 0, .3]), 2: shifted(B, [4, 0, 0, 0]), 3: B, 4: shifted(B, [2, 2, 2, 2]), 5: shifted(B, [3, 3, 3, 3])}
    curve, tables = build(matrix)
    expected_means = {0: 14.0, 1: 11.075, 2: 11.0, 3: 10.0, 4: 12.0, 5: 13.0}
    for k, m in expected_means.items():
        assert math.isclose(curve.loc[curve["k"].eq(k), "subject_equal_rmse"].iloc[0], m, abs_tol=1e-12)
    out = one_standard_error_choice(curve, tables)
    assert out["best_k"] == 3 and out["subjects"] == 4
    assert math.isclose(out["best_rmse"], 10.0, abs_tol=1e-12)
    assert math.isclose(out["se_unpaired"], math.sqrt(2 / 3) / 2, abs_tol=1e-12)
    expected = {0: (4.0, 1.154701), 1: (1.075, 0.977561), 2: (1.0, 1.0), 3: (0.0, 0.0), 4: (2.0, 0.0), 5: (3.0, 0.0)}
    for k, (diff, se) in expected.items():
        got = out["paired_difference_and_se"][k]
        assert math.isclose(got[0], diff, abs_tol=1e-9), (k, got, diff)
        assert math.isclose(got[1], se, abs_tol=1e-6), (k, got, se)
    assert out["k_paired"] == 2          # k=0 and k=1 fail, k=2 satisfies with equality, k=3 is the best point
    assert out["k_unpaired"] == 3        # threshold 10 + 0.408248: k=2 (11.0) is outside


def test_best_k_zero():
    # RMSE_0 = [10]*4 is the best point (means: k=0 10, k=1 11.5, k=2 10.5)
    # k=1: d=[1,2,1,2]  mean 1.5, deviations +-0.5, sum of squares 1, variance 1/3, SE 0.288675; k=2: d=[.5]*4 mean 0.5, SE 0
    matrix = {0: [10, 10, 10, 10], 1: [11, 12, 11, 12], 2: [10.5, 10.5, 10.5, 10.5]}
    out = one_standard_error_choice(*build(matrix))
    assert out["best_k"] == 0 and out["k_paired"] == 0 and out["k_unpaired"] == 0
    assert out["se_unpaired"] == 0.0
    assert out["paired_difference_and_se"][0] == (0.0, 0.0)
    assert math.isclose(out["paired_difference_and_se"][1][0], 1.5, abs_tol=1e-12) and math.isclose(out["paired_difference_and_se"][1][1], 0.288675, abs_tol=1e-6)
    assert math.isclose(out["paired_difference_and_se"][2][0], 0.5, abs_tol=1e-12) and out["paired_difference_and_se"][2][1] == 0.0


def test_identical_rmse_of_two_k_goes_to_the_smaller_k():
    T = [9, 11, 10, 10]
    # k=2 and k=3 have identical per-subject RMSE (mean 10): the first minimum is k=2; k=3 has d=0 and SE=0
    # k=0: d=[3]*4 mean 3 (fails); k=1: d=[1]*4 mean 1, SE 0 (fails)
    matrix = {0: shifted(T, [3] * 4), 1: shifted(T, [1] * 4), 2: T, 3: list(T)}
    out = one_standard_error_choice(*build(matrix))
    assert out["best_k"] == 2 and out["k_paired"] == 2 and out["k_unpaired"] == 2
    assert out["paired_difference_and_se"][3] == (0.0, 0.0)
    # tie between k=0 and k=1 at the best point: the baseline wins
    out = one_standard_error_choice(*build({0: T, 1: list(T), 2: shifted(T, [1] * 4)}))
    assert out["best_k"] == 0 and out["k_paired"] == 0


def test_different_subject_sets_are_rejected():
    matrix = {0: [10, 10, 10, 10], 1: [9, 9, 9, 9]}
    curve, tables = build(matrix)
    tables[1] = pd.DataFrame({"rmse": pd.Series([9.0, 9.0, 9.0, 9.0], index=pd.Index([1, 2, 3, 5], name="subject"))})   # subject 4 replaced by 5
    _expect_rejection(curve, tables)
    tables[1] = pd.DataFrame({"rmse": pd.Series([9.0, 9.0, 9.0], index=pd.Index([1, 2, 3], name="subject"))})            # subject 4 absent
    _expect_rejection(curve, tables)


def test_missing_subject_rmse_is_rejected():
    curve, tables = build({0: [10, 10, 10, 10], 1: [9, 9, np.nan, 9]})
    _expect_rejection(curve, tables)


def test_too_few_subjects_for_a_standard_error():
    # one subject: the sample standard deviation of the paired differences is undefined; the function must refuse instead of selecting silently
    curve, tables = build({0: [10.0], 1: [9.0], 2: [8.5]}, subjects=[1])
    try:
        one_standard_error_choice(curve, tables)
    except ValueError:
        return
    raise AssertionError("one subject must raise ValueError")


def test_formal_selection_follows_from_the_stored_paired_differences():
    # results/factor_discovery/validation_curves_T.csv stores, per k, the paired mean difference to the best point and its SE
    stored = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "factor_discovery" / "validation_curves_T.csv")
    chosen = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "factor_discovery" / "selection_T.csv").set_index("evaluation_protocol")["k_selected"]
    for name, g in stored.groupby("evaluation_protocol"):
        g = g.sort_values("k")
        meets = g[g["paired_diff_vs_best"] <= g["paired_se_vs_best"] + 1e-15]
        assert int(meets["k"].min()) == int(chosen[name]), (name, int(meets["k"].min()), int(chosen[name]))
        below = g[g["k"] < chosen[name]]
        assert (below["paired_diff_vs_best"] > below["paired_se_vs_best"]).all()      # every smaller k is outside one SE
        at = g[g["k"].eq(chosen[name])].iloc[0]
        assert at["paired_diff_vs_best"] <= at["paired_se_vs_best"]


def _expect_rejection(curve, tables):
    try:
        one_standard_error_choice(curve, tables)
    except ValueError:
        return
    raise AssertionError("the function accepted inconsistent subject sets or a missing RMSE")


def test_shared_selection_result_preserves_prefix_and_paired_rule():
    import json
    from src.selection import selection_result, one_standard_error_choice

    index = pd.Index([1, 2, 3], name="subject")
    tables = {0: pd.DataFrame({"rmse": [3., 4., 5.]}, index=index),
              1: pd.DataFrame({"rmse": [2., 3., 4.]}, index=index),
              2: pd.DataFrame({"rmse": [2., 3., 4.]}, index=index)}
    curve = pd.DataFrame({"k": [0, 1, 2], "added": ["", "first", "second"], "subject_equal_rmse": [4., 3., 3.]})
    expected = one_standard_error_choice(curve, tables)
    rows = pd.DataFrame({"subject": [1, 2, 3]})
    out, chosen = selection_result(curve, tables, "A", rows, rows, include_unpaired=True, pool_size=2)
    assert chosen["k_selected"] == expected["k_paired"] == 1
    assert json.loads(chosen["selected_set"]) == ["first"]
    assert json.loads(chosen["factors_in_order"]) == ["first", "second"]
    assert out["within_one_se_paired"].tolist() == [False, True, True]
    assert "evaluation_protocol" not in curve  # caller's curve is not mutated


def test_notebook_selection_scope_keeps_original_window_rule():
    import ast
    import json
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "08_factor_discovery.ipynb").read_text(encoding="utf-8"))
    functions = [node for cell in nb["cells"] if cell["cell_type"] == "code" for node in ast.walk(ast.parse("".join(cell["source"])))
                 if isinstance(node, ast.FunctionDef) and node.name in {"core_rows", "selection_rows"}]
    frame = pd.DataFrame({"subject": [1] * 20 + [2] * 19 + [1, 2], "split": ["validation"] * 39 + ["train"] * 2,
                          "residual": 1., "cohort_core": True})
    context = {"MODEL_TABLES": {"A": frame}, "MIN_WINDOWS": 20}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "scope-test", "exec"), context)
    assert len(context["core_rows"]("A")[1]) == 39
    train, selected = context["selection_rows"]("A")
    assert len(train) == 2 and len(selected) == 20 and set(selected["subject"]) == {1}


def test_batch_ic_reuse_matches_combined_columns_including_missing_values():
    from src.evaluation import subject_ic

    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"subject": np.repeat([1, 2, 3], 30), "residual": rng.normal(size=90),
                          "first": rng.normal(size=90), "second": rng.normal(size=90), "added": rng.normal(size=90)})
    frame.loc[::4, "first"] = np.nan
    expected, expected_counts = subject_ic(frame, ["first", "second", "added"])
    old, old_counts = subject_ic(frame, ["first", "second"])
    added, added_counts = subject_ic(frame, ["added"])
    pd.testing.assert_frame_equal(pd.concat([old, added], axis=1), expected)
    pd.testing.assert_frame_equal(pd.concat([old_counts, added_counts], axis=1), expected_counts)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception as error:      # noqa: BLE001 - report every failure
            failures += 1
            print("FAIL", name, type(error).__name__, error)
    sys.exit(1 if failures else 0)

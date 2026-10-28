"""Checks of the OpenFE comparator on synthetic data (no project data, nothing from any test split).

Run from the repository root with the OpenFE environment: `.venv-openfe\\Scripts\\python.exe tests/test_openfe_runner.py`.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openfe import get_candidate_features  # noqa: E402
from openfe.utils import formula_to_tree, tree_to_formula  # noqa: E402

from src.openfe_runner import GLOBAL_OPERATORS, ROW_WISE_OPERATORS, candidate_formulas, evaluate_formula, formula_columns, parse_formula, run_search  # noqa: E402

COLUMNS = ["act_hr_mean_30m", "diet_last_meal_carbs", "bio_bmi", "native_slope_1h"]


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(0, 3, n) for c in COLUMNS})
    frame.loc[::7, "bio_bmi"] = 0.0                       # zeros: division and logarithm
    frame.loc[::11, "native_slope_1h"] = np.nan           # missing values
    frame["diet_last_meal_carbs"] = frame["diet_last_meal_carbs"].round(1)
    frame.loc[::13, "act_hr_mean_30m"] = 800.0            # overflow in square and sigmoid
    frame.index = pd.RangeIndex(1000, 1000 + n)
    return frame


def _reference(node, frame):
    values = node.calculate(frame, is_root=True)
    return values.replace([np.inf, -np.inf], np.nan)


def test_candidates_use_row_wise_operators_only():
    formulas = candidate_formulas(COLUMNS)
    n = len(COLUMNS)
    assert len(formulas) == 7 * n + 6 * n * (n - 1) // 2
    operators = {op for text in formulas for op in _ops(parse_formula(text))}
    assert operators == set(ROW_WISE_OPERATORS) and not operators & set(GLOBAL_OPERATORS)


def _ops(node):
    return [] if isinstance(node, str) else [node[0]] + [o for child in node[1] for o in _ops(child)]


def test_parser_and_evaluator_equal_the_library_for_order_two_formulas():
    frame = _frame()
    nodes = [n for n in get_candidate_features(numerical_features=COLUMNS[:3], order=2) if n.name != "freq" and "freq" not in tree_to_formula(n)]
    checked = 0
    for node in nodes:
        text = tree_to_formula(node)
        if set(_ops(parse_formula(text))) - set(ROW_WISE_OPERATORS):
            continue
        assert tree_to_formula(formula_to_tree(text)) == text
        ours, theirs = evaluate_formula(text, frame), _reference(formula_to_tree(text), frame)
        assert formula_columns(text) == [c for c in COLUMNS[:3] if c in formula_columns(text)] or set(formula_columns(text)) <= set(COLUMNS[:3])
        both_nan = ours.isna() & theirs.isna()
        assert (both_nan | np.isclose(ours, theirs, rtol=1e-12, atol=0, equal_nan=False)).all(), text
        checked += 1
    assert checked > 1000


def test_row_wise_values_do_not_depend_on_other_rows():
    frame = _frame()
    for text in candidate_formulas(COLUMNS)[:60]:
        full = evaluate_formula(text, frame)
        part = evaluate_formula(text, frame.iloc[100:160])
        single = pd.concat([evaluate_formula(text, frame.iloc[[i]]) for i in range(100, 110)])
        assert full.iloc[100:160].equals(part) and full.iloc[100:110].equals(single)


def test_global_operators_of_the_library_use_other_rows():
    """Documents why they are excluded: a feature of one row changes when other rows are added or changed."""
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=60), "g": rng.integers(0, 4, 60).astype(float)})
    test_a = pd.DataFrame({"x": rng.normal(size=20), "g": rng.integers(0, 4, 20).astype(float)})
    test_b = test_a.assign(x=test_a["x"] + 50.0)
    for text in ("GroupByThenMean(x,g)", "GroupByThenStd(x,g)", "freq(g)", "GroupByThenRank(x,g)"):
        node = formula_to_tree(text)
        both_a, both_b = pd.concat([train, test_a]), pd.concat([train, test_b])
        if text.startswith("freq"):
            both_b = pd.concat([train, test_a.assign(g=test_a["g"] + 9)])
        first_a = node.calculate(both_a, is_root=True).iloc[:60].to_numpy()
        first_b = node.calculate(both_b, is_root=True).iloc[:60].to_numpy()
        assert not np.allclose(first_a, first_b, equal_nan=True), text                 # the training rows' features changed with the test rows
        alone = node.calculate(test_a.iloc[[0]], is_root=True).iloc[0]
        batch = node.calculate(test_a, is_root=True).iloc[0]
        assert not (np.isclose(alone, batch) or (np.isnan(alone) and np.isnan(batch))), text    # one row alone differs from the same row inside a batch


def _synthetic_search(directory, init_with_base, leak, seed=3, misalign=False, label_derived=False):
    rng = np.random.default_rng(seed)
    n = 3200
    table = pd.DataFrame({c: rng.normal(size=n) for c in ["a", "b", "c", "d", "e"]})
    base = 2.0 * table["c"] - 1.0 * table["d"] + 100.0
    noise = rng.normal(0, 0.5, n)
    y = base + 1.5 * table["a"] * table["b"] + noise
    role = np.where(np.arange(n) < 2400, "internal_train", "internal_validation")
    if leak:                                                                      # a column that equals the residual on validation rows only
        table["leak"] = np.where(role == "internal_validation", (y - base).to_numpy(), rng.normal(size=n))
    if label_derived:                                                             # negative control: a column built from the label on every row
        table["leak"] = (y - base).to_numpy() + rng.normal(0, 0.1, n)
    columns = list(table.columns)
    table.insert(0, "openfe_index", np.arange(n) * 3 + 5)
    init = base.to_numpy() if init_with_base else np.zeros(n)
    if misalign:                                                                  # negative control: scores shifted against the rows they belong to
        init = np.roll(init, 11)
    table["role"], table["label"], table["init_score"] = role, y.to_numpy(), init
    path = Path(directory) / f"synthetic_{init_with_base}_{leak}.parquet"
    table.to_parquet(path)
    output = Path(tempfile.mkdtemp(dir=directory)) / "search"
    config = {"input": str(path), "columns": columns, "output_dir": str(output), "tag": f"s{int(init_with_base)}{int(leak)}{int(misalign)}{int(label_derived)}", "seed": 42, "n_jobs": 2, "tmp_dir": str(output.parent),
              "n_data_blocks": 2, "min_candidate_features": 30, "verbose": False}
    run_search(config)
    return pd.read_csv(output / f"ranking_{config['tag']}.csv")


def test_init_scores_define_the_increment_and_the_split_is_explicit():
    with tempfile.TemporaryDirectory() as directory:
        boosted = _synthetic_search(directory, init_with_base=True, leak=True)
        top = boosted["formula"].head(3).tolist()
        assert any(set(formula_columns(f)) == {"a", "b"} and ("*" in f) for f in top), top                  # the planted a*b increment over the base is found
        leaked = boosted[boosted["formula"].str.contains("leak")]                                                 # a column that only validation labels explain cannot be learned from the training rows
        assert leaked["stage2_gain_importance"].max() < 0.01 * boosted["stage2_gain_importance"].iloc[0] and leaked["stage1_last_round_score"].max() < 0.05 * boosted["stage1_last_round_score"].iloc[0]
        zeros = _synthetic_search(directory, init_with_base=False, leak=False)
        assert any(set(formula_columns(f)) <= {"c", "d"} for f in zeros["formula"].head(3)), zeros.head(3)  # without the base the main effects dominate: init_scores are used
        again = _synthetic_search(directory, init_with_base=True, leak=True)
        assert again["formula"].tolist() == boosted["formula"].tolist()                                      # deterministic for a fixed seed


def _planted_score(ranking):
    row = ranking[ranking["formula"].isin(["(a*b)", "(b*a)"])]
    return float(row["stage1_last_round_score"].iloc[0]) if len(row) else 0.0


def test_negative_controls_make_the_checks_fail():
    """The invariants above must be able to fail: shifted init_scores lose the planted increment, and a column derived from the label is found at once."""
    with tempfile.TemporaryDirectory() as directory:
        aligned = _synthetic_search(directory, init_with_base=True, leak=False)
        shifted = _synthetic_search(directory, init_with_base=True, leak=False, misalign=True)
        assert _planted_score(shifted) < 0.5 * _planted_score(aligned), (_planted_score(aligned), _planted_score(shifted))
        print("planted a*b stage-1 score: aligned %.3f, shifted %.3f" % (_planted_score(aligned), _planted_score(shifted)))
        derived = _synthetic_search(directory, init_with_base=True, leak=False, label_derived=True)
        assert "leak" in derived["formula"].iloc[0], derived.head(3)



def test_mismatched_input_digest_is_refused_before_loading_or_searching():
    import hashlib
    from unittest.mock import patch
    from unittest import TestCase
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.parquet"
        path.write_bytes(b"synthetic input identity")
        with patch.object(pd, "read_parquet", side_effect=AssertionError("input was loaded")) as read:
            with TestCase().assertRaises(ValueError):
                run_search({"input": str(path), "input_sha256": "0" * 64})
            read.assert_not_called()
            with TestCase().assertRaisesRegex(AssertionError, "input was loaded"):
                run_search({"input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == "__main__":
    failures = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]:
        try:
            fn()
            print("PASS", name)
        except Exception as error:      # noqa: BLE001
            failures += 1
            import traceback
            print("FAIL", name, type(error).__name__, error)
            traceback.print_exc()
    sys.exit(1 if failures else 0)

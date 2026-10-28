"""Checks of the nested subject-grouped CV on synthetic data (no data files, no model).

Run with `python tests/test_grouped_cv.py` from the repository root.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import grouped_cv as G  # noqa: E402

LAGS = [f"g{o}" for o in range(-115, 1, 5)]
CAT1 = pd.DataFrame({
    "candidate_id": ["abs_lunch_carbs_t20_k2", "abs_lunch_carbs_t30_k2", "act_a", "act_b", "act_c", "nat_a"],
    "family": ["meal_absorption", "meal_absorption", "activity_kernels", "activity_kernels", "activity_kernels", "native_glucose_state"],
    "parameters": [json.dumps({"tau": 20, "k": 2}), json.dumps({"tau": 30, "k": 2}), "{}", "{}", "{}", "{}"],
    "generator_params": [json.dumps({"meal_class": "lunch", "nutrient": "carbs"})] * 2 + ["{}"] * 4,
    "uses_native_state": [False, False, False, False, False, True]})


def _table(n_subjects=12, windows=220, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(1, n_subjects + 1):
        walk = np.cumsum(rng.normal(size=windows + 40)) * 3 + 120
        lags = np.stack([walk[20 + i - np.arange(115, -1, -5) // 5 * 1 if False else np.arange(i, i + 24)] for i in range(windows)])   # 24 values per window
        sig = rng.normal(size=(windows, 6))
        target = lags[:, -1] + 4 * sig[:, 0] - 3 * sig[:, 2] + rng.normal(size=windows)
        frame = pd.DataFrame(lags, columns=LAGS)
        for j, c in enumerate(CAT1["candidate_id"]):
            frame[c] = sig[:, j]
        frame["m1"], frame["m2"] = rng.normal(size=windows), rng.normal(size=windows)
        frame["target_value"] = target
        frame["subject"], frame["task_id"] = s, 1
        frame["origin_time"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(windows) * 5, unit="m")
        frame["split"], frame["cohort_core"], frame["native_missing"] = "train", True, 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _setup(table):
    calls = []

    def builder(tau_star):
        calls.append(dict(tau_star))
        rng = np.random.default_rng(len(str(sorted(tau_star.items()))))
        values = pd.DataFrame({"int_x": table["act_a"].to_numpy() * table["m1"].to_numpy(), "int_noise": rng.normal(size=len(table))}, index=table.index)
        cat2 = pd.DataFrame({"candidate_id": ["int_x", "int_noise"], "family": ["meal_x_baseline_state"] * 2, "uses_native_state": [False, False]})
        return cat2, values

    return G.FoldSetup(lag_columns=LAGS, manual_columns=["m1", "m2"], catalog1=CAT1, batch2_builder=builder, k_max=4), calls


def test_groupkfold_membership_is_deterministic_independent_of_row_order_and_has_no_seed():
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(5)
    counts = pd.Series(rng.integers(500, 3000, 36), index=range(1, 37))
    table = pd.DataFrame({"subject": np.repeat(counts.index.to_numpy(), counts.to_numpy()), "split": "train"})
    folds, info, frame = G.membership_from_table(table, 6)
    assert sorted(s for v in folds.values() for s in v) == list(range(1, 37)) and info["rule"].startswith("sklearn.model_selection.GroupKFold(n_splits=6, shuffle=False)")
    shuffled = table.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert G.membership_from_table(shuffled, 6)[0] == folds                                                     # the row order does not matter
    expected = {k: sorted(int(s) for s in np.unique(table["subject"].to_numpy()[held])) for k, (_, held) in enumerate(GroupKFold(n_splits=6, shuffle=False).split(np.zeros(len(table)), groups=table["subject"]))}
    assert folds == expected
    assert frame["n_windows"].sum() == len(table) and set(frame["main_pipeline_role"]) == {"train"} and info["sklearn_version"]


def test_a_registered_membership_is_read_and_a_changed_file_is_refused():
    import hashlib
    import tempfile

    table = pd.DataFrame({"subject": np.repeat(np.arange(1, 13), 50), "split": "train"})
    folds, info, frame = G.membership_from_table(table, 3)
    text = frame.to_csv(index=False, lineterminator="\n").encode()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "membership.csv"
        path.write_bytes(text)
        digest = hashlib.sha256(text).hexdigest()
        assert G.load_membership(path, digest) == folds
        path.write_bytes(text + b"1,0,1,train")                                             # any change of the bytes
        try:
            G.load_membership(path, digest)
        except RuntimeError:
            return
        raise AssertionError("a changed membership file was accepted")


def test_the_fold_selects_inside_the_training_subjects_and_the_outer_subjects_do_not_matter():
    table = _table()
    setup, calls = _setup(table)
    outer = [1, 2, 3]
    result = G.run_fold(table, outer, setup, fold_id=0)
    audit = result["audit"]
    assert audit["subjects_disjoint"] and set(audit["training_subjects"]).isdisjoint(outer) and audit["ic_rows_subjects_subset_of_training"]
    assert audit["inner_split_subjects_subset_of_training"] and audit["inner_split_held_subjects_disjoint_from_their_training"] and audit["outer_evaluations"] == 1
    assert set(audit["base_fit_subjects"]) == set(audit["training_subjects"])
    selected = result["selection"]["selected"]
    assert {"abs_lunch_carbs_t20_k2", "act_a"} <= set(selected)                                          # the planted signals are found
    assert result["selection"]["tau_star"] == {"carbs|lunch": [20, 2]}                                   # tau* is re-selected from the training subjects (the stronger kernel wins)
    # nothing of the outer subjects reaches a selection step: every selection output is identical when their labels, lags and features are replaced by noise
    spoiled = G.run_fold(table, outer, setup, fold_id=0, perturb_outer=True, evaluate=False)
    for key in ("tau_star", "pool_after_ic", "pool_after_filter", "factors_in_order", "k_selected", "selected", "alpha_T", "alpha_M", "alpha_MT", "inner_curve"):
        assert json.dumps(spoiled["selection"][key]) == json.dumps(result["selection"][key]), key                  # NaN-safe comparison
    # evaluation: four configurations on the same outer subjects, with the three contrasts of the plan
    assert set(result["outer"]) == {"base", "base + M", "base + T", "base + M + T"}
    assert all(set(t["subject"]) == set(outer) for t in result["outer"].values())
    assert set(result["contrasts"]) == {"base -> base + M", "base -> base + T", "base + M -> base + M + T"}
    assert result["contrasts"]["base -> base + T"]["d_rmse"] < 0                                     # the planted factors help outside the training subjects
    assert result["timing"]["total_s"] > 0 and result["timing"]["peak_rss_mb"] > 0


def test_aggregation_stacks_every_subject_once():
    table = _table(n_subjects=12)
    setup, _ = _setup(table)
    folds = G.membership_from_table(table, 4)[0]
    results = [G.run_fold(table, folds[k], setup, fold_id=k) for k in range(4)]
    out = G.aggregate_folds(results, n_resamples=200)
    assert out["subjects"].iloc[0] == 12 and set(out["to_config"]) == {"base + M", "base + T", "base + M + T"} and len(out) == 6
    assert len(set(s for r in results for s in r["outer"]["base"]["subject"])) == 12


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

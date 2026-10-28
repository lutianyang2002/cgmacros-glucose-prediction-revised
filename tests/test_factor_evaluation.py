"""Checks of the factor-evaluation notebook engine, the lock and the test-split guard (synthetic data and the lock file; no test data, no model).

Run with `python tests/test_factor_evaluation.py` from the repository root.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import factor_evaluation as FE  # noqa: E402
from src import factor_validation as FG  # noqa: E402
from src import comparison_registry as FL  # noqa: E402
from src import window_data as W  # noqa: E402
from src.residuals import ridge_path_predictions  # noqa: E402

from test_bootstrap_holm import bootstrap_p

LOCK = ROOT / "results" / "factor_discovery" / "evaluation_plan.json"


def _table(n_subjects=10, rows=80, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(1, n_subjects + 1):
        x = rng.normal(size=(rows, 2))
        base = 120 + rng.normal(size=rows) * 10
        target = base + 2 * x[:, 0] + rng.normal(size=rows)
        frames.append(pd.DataFrame({"subject": s, "split": np.where(np.arange(rows) < 50, "train", "validation"), "origin_time": pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(rows) * 5, unit="m"),
                                    "task_id": 1, "x0": x[:, 0], "x1": x[:, 1], "base_prediction": base, "target_value": target, "residual": target - base, "cohort_core": rng.random(rows) > 0.1,
                                    "diet_no_logged_meal_30m": (rng.random(rows) > 0.5).astype(int)}))
    table = pd.concat(frames, ignore_index=True)
    table["diet_no_logged_meal_120m"] = np.where(table["diet_no_logged_meal_30m"] == 0, 0, (np.random.default_rng(1).random(len(table)) > 0.5).astype(int))
    return table


CONFIGS = {"base": {"columns": [], "alpha": None}, "base + X": {"columns": ["x0", "x1"], "alpha": 1.0}, "base + X (heavy penalty)": {"columns": ["x0", "x1"], "alpha": 1000.0}}


def test_the_engine_uses_the_fixed_alpha_and_nothing_from_the_evaluation_split():
    table = _table()
    ev = FE.Evaluator(table, CONFIGS, [], {}, "validation")
    errors = ev.errors_of("base + X")
    train = table[(table["split"] == "train") & table["cohort_core"] & table["residual"].notna()]
    valid = table[(table["split"] == "validation") & table["cohort_core"]]
    prediction = ridge_path_predictions(train[["x0", "x1"]], train["residual"].to_numpy(), valid[["x0", "x1"]], alphas=[1.0])[:, 0]
    assert np.allclose(errors["error"].to_numpy(), valid["base_prediction"].to_numpy() + prediction - valid["target_value"].to_numpy())
    spoiled = table.copy()
    mask = spoiled["split"] == "validation"
    spoiled.loc[mask, ["target_value", "residual"]] = 999.0                                   # evaluation labels do not reach the fit
    again = FE.Evaluator(spoiled, CONFIGS, [], {}, "validation").errors_of("base + X")
    assert np.allclose(again["error"] - errors["error"], valid["target_value"].to_numpy() - 999.0)
    train_spoiled = spoiled[(spoiled["split"] == "train") & spoiled["cohort_core"]]
    assert train_spoiled["residual"].equals(train["residual"])


def test_contrasts_strata_holm_and_p_values():
    table = _table()
    ev = FE.Evaluator(table, CONFIGS, [], {}, "validation")
    comparison = {"id": "A:P1", "evaluation_protocol": "p", "family": "primary", "role": "confirmatory (Holm family of 4)", "from_config": "base", "to_config": "base + X"}
    rows = pd.DataFrame(FE.contrast_rows(ev, comparison, "core", "all", None, 200))
    assert set(rows["min_window_rule"]) == {"applied (>= 20 windows)", "not applied"} and (rows["d_rmse"] < 0).all()            # the planted signal helps
    masks = FE.strata_masks(table)
    assert (sum(m.astype(int) for m in masks.values()) == 1).all()
    heavy = {"id": "A:P2", "evaluation_protocol": "p", "family": "primary", "role": "confirmatory (Holm family of 4)", "from_config": "base", "to_config": "base + X (heavy penalty)"}
    out = FE.run_comparisons(ev, [comparison, heavy], None, 200, strata_ids=("P1",))
    assert set(out[out["stratum"] != "all"]["id"]) == {"A:P1"} and out["p_holm_primary_family"].notna().sum() == 2
    assert FE.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06] or np.allclose(FE.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(FE.holm([0.5, 0.01]), [0.5, 0.02])
    t_from, t_to = ev.subject_tables("base"), ev.subject_tables("base + X")
    assert bootstrap_p(t_from, t_to, 20, 200) <= 0.05 and bootstrap_p(t_from, t_from, 20, 200) == 1.0


def test_pooled_levels_use_the_same_filtered_subjects_as_the_pooled_delta():
    class SyntheticEvaluator:
        def __init__(self):
            self.tables = {
                "from": pd.DataFrame(
                    {
                        "n_windows": [20, 19, 22],
                        "mae": [8.0, 9.0, 16.0],
                        "rmse": [10.0, 12.0, 20.0],
                        "sum_abs_error": [160.0, 171.0, 352.0],
                        "sum_squared_error": [2000.0, 2736.0, 8800.0],
                    },
                    index=pd.Index([101, 102, 103], name="subject"),
                ),
                "to": pd.DataFrame(
                    {
                        "n_windows": [20, 21, 22],
                        "mae": [7.0, 8.0, 15.0],
                        "rmse": [8.0, 10.0, 18.0],
                        "sum_abs_error": [140.0, 168.0, 330.0],
                        "sum_squared_error": [1280.0, 2100.0, 7128.0],
                    },
                    index=pd.Index([101, 102, 103], name="subject"),
                ),
            }
            self.keys = pd.Index([0, 1, 2], name="prediction_key")

        def errors_of(self, config, cohort="core"):
            return pd.DataFrame({"error": [0.0, 0.0, 0.0]}, index=self.keys)

        def subject_tables(self, config, cohort="core", mask=None):
            return self.tables[config].copy()

    evaluator = SyntheticEvaluator()
    comparison = {
        "id": "A:P1",
        "evaluation_protocol": "synthetic",
        "family": "primary",
        "role": "synthetic",
        "from_config": "from",
        "to_config": "to",
    }
    rows = FE.contrast_rows(evaluator, comparison, "core", "all", None, 100)
    applied = next(row for row in rows if row["min_window_rule"] == "applied (>= 20 windows)")

    expected_from = np.sqrt((20 * 10.0**2 + 22 * 20.0**2) / 42)
    expected_to = np.sqrt((20 * 8.0**2 + 22 * 18.0**2) / 42)
    assert applied["subjects"] == 2
    assert applied["validation_windows"] == 42
    assert np.isclose(applied["pooled_rmse_from"], expected_from)
    assert np.isclose(applied["pooled_rmse_to"], expected_to)
    assert np.isclose(applied["pooled_rmse_to"] - applied["pooled_rmse_from"], applied["pooled_d_rmse"])


def test_contrasts_reject_different_prediction_keys():
    class SyntheticEvaluator:
        def errors_of(self, config, cohort="core"):
            keys = [0, 1] if config == "from" else [0, 2]
            return pd.DataFrame({"error": [0.0, 0.0]}, index=keys)

        def subject_tables(self, config, cohort="core", mask=None):
            return pd.DataFrame(
                {
                    "n_windows": [20, 20],
                    "mae": [1.0, 1.0],
                    "rmse": [1.0, 1.0],
                    "sum_abs_error": [20.0, 20.0],
                    "sum_squared_error": [20.0, 20.0],
                },
                index=pd.Index([101, 102], name="subject"),
            )

    comparison = {
        "id": "A:P1",
        "evaluation_protocol": "synthetic",
        "family": "primary",
        "role": "synthetic",
        "from_config": "from",
        "to_config": "to",
    }
    try:
        FE.contrast_rows(SyntheticEvaluator(), comparison, "core", "all", None, 100)
    except ValueError as exc:
        assert "identical prediction keys" in str(exc)
    else:
        raise AssertionError("different from/to prediction keys were accepted")


def test_the_guard_refuses_the_test_split_before_any_file_is_read():
    for access in (W.Access(), W.Access(execute_test=True), W.Access(lock_approved=True)):
        with FG.ReadRecorder() as recorder:
            try:
                W.load_keys(W.PROTOCOL_A, ("train", "test"), access)
            except W.TestAccessDenied:
                pass
            else:
                raise AssertionError("the test split was not refused")
        assert recorder.log == []
    assert W.check_access(("train", "validation"), W.Access()) == ("train", "validation") and W.check_access(("train", "test"), W.Access(True, True)) == ("train", "test")
    try:
        W.check_access(("train", "dev"), W.Access())
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown split was accepted")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    try:
        FE.run_fixed_evaluation(lock, "test", W.Access())
    except W.TestAccessDenied:
        pass
    else:
        raise AssertionError("the locked evaluation ran on the test split")
    try:
        FE.assemble_protocol("A", lock, ("train", "test"), W.Access(execute_test=True))
    except W.TestAccessDenied:
        pass
    else:
        raise AssertionError("assemble_protocol read the test split without an approved lock")


def test_scientific_plan_has_fifty_comparisons_and_fixed_statistics():
    plan = json.loads(LOCK.read_text(encoding="utf-8"))
    assert plan["execute_test_default"] is False
    assert plan["comparison_counts"] == {"A": 24, "B": 26, "total": 50}
    ids = [c["id"] for c in plan["comparisons"]]
    assert len(set(ids)) == len(ids) == 50
    for comparison in plan["comparisons"]:
        configurations = plan["configs"][comparison["protocol_tag"]]
        assert comparison["from_config"] in configurations and comparison["to_config"] in configurations
    assert plan["statistics"]["bootstrap"] == {"resamples": 2000, "seed": 42, "unit": "subjects, paired", "ci": 0.95}
    assert plan["metrics"]["min_windows_per_subject"] == 20


def test_explicit_training_context_does_not_execute_notebook_or_load_factor_pools():
    from unittest.mock import patch
    from src import factor_data as data

    keys = pd.DataFrame({"subject": [1, 1], "origin_time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                         "task_id": [1, 1], "split": ["train", "validation"]})
    reg = {"G_FEATURES": ["g"], "FEATURE_GROUPS": {"A": ["m"], "D": [], "B": []}, "MANUAL": ["m"]}
    def build(protocol, splits, registry=None, factor_dir=None, load_residuals=True):
        assert splits == ("train", "validation") and registry is reg
        return keys.assign(base_prediction=1., target_value=2., cohort_core=True, **({"residual": 1.} if load_residuals else {})), ["g-0"]
    def csv(path, *args, **kwargs):
        assert Path(path).name == "metabolic_covariates.csv"
        return pd.DataFrame({"subject": [1], "bmi_kg_m2": [20.]})
    with patch.object(W, "registries", return_value=reg), patch.object(W, "build_table", side_effect=build),\
         patch.object(pd, "read_csv", side_effect=csv), patch.object(pd, "read_parquet", side_effect=AssertionError("unrequested factor pool")):
        context = data.load_context(include_factors=False)
        upstream = data.load_context(include_factors=False, load_residuals=False)
        assert all("residual" not in table for table in upstream["tables"].values())
    assert set(context["tables"]) == set(data.PROTOCOLS.values())
    assert {"get_inputs", "generate_matrix", "BMI", "WINDOW_KEY", "EVENT_COLUMNS", "DATA_QUALITY_DIR"} <= set(context)


def test_explicit_factor_context_keeps_downstream_columns_and_lazy_inputs():
    from types import SimpleNamespace
    from unittest.mock import patch
    from src import factor_data as data

    keys = pd.DataFrame({"subject": pd.Series([1, 1], dtype="int32"), "origin_time": pd.to_datetime(["2020-01-02", "2020-01-01"]), "task_id": [1, 1], "split": ["validation", "train"]})
    names = list(data.PROTOCOLS.values())
    reg = {"G_FEATURES": ["g"], "FEATURE_GROUPS": {"A": ["m"], "D": [], "B": []}, "MANUAL": ["m"]}
    def csv(path, *args, **kwargs):
        name = Path(path).name
        if name == "metabolic_covariates.csv": return pd.DataFrame({"subject": [1], "bmi_kg_m2": [20.]})
        if name in {"candidate_catalog.csv", "candidate_catalog.csv"}: return pd.DataFrame({"candidate_id": ["t", "n"], "uses_native_state": [False, True]})
        if name == "selection_T.csv": return pd.DataFrame({"evaluation_protocol": names, "selected_set": [json.dumps(["t", "n"])] * 2})
        if name == "selection_O.csv": return pd.DataFrame({"evaluation_protocol": names, "selected_set": [json.dumps(["O"])] * 2})
        if name == "openfe_features.csv": return pd.DataFrame({"candidate_id": ["O"], "formula": ["(native_last+diet_no_logged_meal_30m)"]})
        return pd.DataFrame()
    def parquet(path, **kwargs):
        name = Path(path).name
        if name.startswith("primitives_"): return keys.assign(native_missing=np.array([1, 0], dtype="int8"))
        column = "O" if name == "openfe_values.parquet" else "t" if "batch1" in name else "n"
        return pd.concat([keys.assign(evaluation_protocol=protocol, **{column: np.array([np.nan, 1.], dtype="float32")}) for protocol in names], ignore_index=True)
    with patch.object(W, "registries", return_value=reg), patch.object(W, "build_table", side_effect=lambda *a, **k: (keys.assign(base_prediction=1., residual=1.), ["g-0"])),\
         patch.object(pd, "read_csv", side_effect=csv), patch.object(pd, "read_parquet", side_effect=parquet),\
         patch.object(W, "load_keys", return_value=keys), patch.object(W, "load_inputs", return_value=({1: pd.DataFrame()}, {})) as load,\
         patch.object(data, "batch1_catalog", return_value=pd.DataFrame({"candidate_id": ["t"]})):
        context = data.load_context()
        assert {"catalog", "catalog_batch1", "batch2_catalog", "generate_matrix", "screening", "curves"} <= set(context)
        for name in names:
            assert {"native_missing", "t", "n", "O"} <= set(context["O_TABLES"][name])
        assert context["model_columns"](["n"]) == ["n", "native_missing"]
        assert context["o_model_columns"](["O"]) == ["O", "native_missing"]
        assert context["get_inputs"](names[0]) is context["get_inputs"](names[0])
        assert load.call_count == 1
        assert context["FORMULA_OF"]["O"] == context["openfe_features"].iloc[0]["formula"]
        notebook = json.loads((ROOT / "notebooks/08_factor_discovery.ipynb").read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "703aaa0e")
        init = ast.parse("".join(next(c["source"] for c in notebook["cells"] if c.get("id") == "1559096e")))
        targets = ast.literal_eval(next(n.value for n in init.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FACTOR_TARGETS" for t in n.targets))) - {"theory"}
        assert {"e9", "t16"} <= targets
        helper_source = "".join(next(c["source"] for c in notebook["cells"] if c.get("id") == "50ae4d45"))
        openfe_targets = {"openfe_selection", "e2_o", "e3", "llm_description"}
        for target in targets:
            def existing_context(**kwargs):
                assert kwargs == dict(include_factors=True, include_openfe=target in openfe_targets, factor_dir=Path("inputs"), openfe_dir=Path("openfe"), openfe_selection_dir=Path("selection"), require_openfe_selection=target != "openfe_selection")
                return {k: v for k, v in context.items() if target in openfe_targets or k not in {"O_TABLES", "O_K", "o_model_columns", "openfe_features", "FORMULA_OF"}}
            namespace = dict(REBUILD_TARGET=target, FACTOR_TARGETS=targets | {"theory"}, discovery=SimpleNamespace(load_context=existing_context), INPUT_DIR=Path("inputs"), OPENFE_FACTOR_DIR=Path("openfe"), OPENFE_SELECTION_DIR=Path("selection"))
            exec(compile(source, "synthetic_context_consumer", "exec"), namespace)
            if target in {"e9", "t16"}:
                exec(compile(helper_source, "synthetic_row_helpers", "exec"), namespace)
                assert callable(namespace["core_rows"]) and callable(namespace["selection_rows"])
            for name in names:
                expected = context["tables"][name]
                for file in ("candidates_batch1.parquet", "primitives_A.parquet", "candidates_batch2.parquet"):
                    part = parquet(Path(file))
                    if "evaluation_protocol" in part:
                        part = part[part["evaluation_protocol"].eq(name)].drop(columns="evaluation_protocol")
                    expected = expected.merge(part, on=W.WINDOW_KEY + ["split"], validate="one_to_one")
                pd.testing.assert_frame_equal(namespace["MODEL_TABLES"][name], expected, check_exact=True)
                if target in openfe_targets:
                    part = parquet(Path("openfe_values.parquet"))
                    part = part[part["evaluation_protocol"].eq(name)].drop(columns="evaluation_protocol")
                    pd.testing.assert_frame_equal(namespace["O_TABLES"][name], expected.merge(part, on=W.WINDOW_KEY + ["split"], validate="one_to_one"), check_exact=True)
        assert "O_TABLES" not in data.load_context(include_openfe=False)
        csv_reads, parquet_reads = [], []
        def explicit_csv(path, **kwargs):
            csv_reads.append(Path(path))
            return csv(path, **kwargs)
        def explicit_parquet(path, **kwargs):
            parquet_reads.append(Path(path))
            return parquet(path, **kwargs)
        with patch.object(pd, "read_csv", side_effect=explicit_csv), patch.object(pd, "read_parquet", side_effect=explicit_parquet):
            rebuilt = data.load_context(factor_dir=Path("new_inputs"), openfe_dir=Path("new_pool"), openfe_selection_dir=Path("new_selection"))
            assert next(p for p in csv_reads if p.name == "selection_O.csv").parent == Path("new_selection")
            assert next(p for p in parquet_reads if p.name == "openfe_values.parquet").parent == Path("new_pool")
            assert next(p for p in csv_reads if p.name == "selection_T.csv").parent == Path("new_inputs")
            for name in names:
                pd.testing.assert_frame_equal(context["O_TABLES"][name], rebuilt["O_TABLES"][name], check_exact=True)
            csv_reads.clear()
            assert data.load_context(openfe_dir=Path("pool_without_selection"), require_openfe_selection=False)["O_K"] == {}
            assert not any(p.name == "selection_O.csv" for p in csv_reads)
            csv_reads.clear()
            with patch.object(Path, "exists", return_value=True):
                data.load_context(include_openfe=False, factor_dir=Path("modern_inputs"))
            assert any(p == Path("modern_inputs/candidate_catalog.csv") for p in csv_reads)


def test_e9_delays_match_hand_calculated_events_and_unrenormalized_gamma():
    from src.factor_construction import batch1_catalog

    start = pd.Timestamp("2020-01-01")
    minutes = {1: pd.DataFrame({"timestamp": pd.date_range(start, periods=241, freq="min"),
                               "is_published_source_row": True, "heart_rate_bpm": 70., "activity_calories_per_minute": 1.})}
    events = {1: pd.DataFrame({"timestamp": start + pd.to_timedelta([140, 150], unit="m"), "meal_type_normalized": ["Lunch", "Dinner"],
                              "carbs": [10., 20.], "protein": [1., 2.], "fat": [1., 2.], "fiber": [1., 2.], "meal_calories": [50., 100.]})}
    keys = pd.DataFrame({"subject": [1, 1], "origin_time": start + pd.to_timedelta([160, 180], unit="m"),
                         "task_id": [1, 1], "split": ["validation", "validation"]})
    recipe = batch1_catalog().query("candidate_id == 'abs_dinner_carbs_t30_k2'")
    ages = np.arange(121, dtype=float)
    weights = ages * np.exp(-ages / 30.)
    weights /= weights.sum()  # early ages stay in this denominator even when unavailable
    for version, expected_count, expected_kernel in [("logged", [2., 1.], [20 * weights[10], 20 * weights[30]]),
                                                     ("delay_free", [1., 1.], [0., 20 * weights[30]]),
                                                     ("delay_all", [0., 1.], [0., 20 * weights[30]])]:
        diet = W.diet_matrix(keys, minutes, events, ["diet_logged_meal_count_30m"], version)
        factors = W.candidate_matrix(keys, minutes, events, recipe, list(recipe["candidate_id"]), pd.Series({1: 20.}), version)
        assert np.array_equal(diet["diet_logged_meal_count_30m"], expected_count)
        assert np.allclose(factors["abs_dinner_carbs_t30_k2"], expected_kernel)
        pd.testing.assert_frame_equal(factors[W.WINDOW_KEY + ["split"]], keys)


def test_validation_references_are_read_once_across_rules_and_protocols():
    from unittest.mock import patch

    reference = pd.DataFrame({"evaluation_protocol": ["A", "B"], "from_config": ["base", "base"], "to_config": ["x", "x"], "d_rmse": [0.1, 0.2]})
    frames = {}
    with patch.object(pd, "read_csv", return_value=reference) as read:
        for protocol in ("A", "B", "A"):
            picked = FG._reference(Path("unused"), {"artifact": "same.csv", "to_config": "x"}, protocol, frames)
            assert len(picked) == 1 and picked.iloc[0]["evaluation_protocol"] == protocol
        assert read.call_count == 1



def test_empty_keys_do_not_open_files_and_task_filters_match_full_keys():
    from unittest.mock import patch
    import tempfile
    keys = pd.DataFrame({"subject": [1, 1], "origin_time": pd.to_datetime(["2020-01-01"] * 2), "task_id": ["a", "b"]})
    with patch.object(W.ds, "dataset", side_effect=AssertionError("empty keys opened a file")):
        for loader in (W.load_values, W.load_minutes):
            try:
                loader(keys.iloc[:0], "unopened.parquet", ["value"])
            except ValueError:
                pass
            else:
                raise AssertionError("empty keys accepted")
        for loader in (W.load_values, W.load_minutes):
            try:
                loader(keys.assign(subject=np.nan), "unopened.parquet", ["value"])
            except ValueError:
                pass
            else:
                raise AssertionError("missing keys accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.parquet"
        keys.assign(value=[11., 22.]).to_parquet(path, index=False)
        result = W.load_values(keys.iloc[[1]], path, ["value"])
        assert result["task_id"].tolist() == ["b"] and result["value"].tolist() == [22.]
        pd.concat([keys, keys.iloc[[1]]]).assign(value=1.).to_parquet(path, index=False)
        try:
            W.load_values(keys.iloc[[1]], path, ["value"])
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError("duplicate keys accepted")


def test_saved_comparison_registry_matches_scientific_lock():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    configs, comparisons = FL.build_registry(LOCK.parent, FL.selection_from_files(LOCK.parent))
    assert configs == lock["configs"]
    assert [(c["id"], c["from_config"], c["to_config"]) for c in comparisons] == [(c["id"], c["from_config"], c["to_config"]) for c in lock["comparisons"]]


def test_upstream_table_does_not_require_a_previous_residual_file():
    from unittest.mock import patch
    keys = pd.DataFrame({"subject": [1, 1], "origin_time": pd.to_datetime(["2020-01-01", "2020-01-02"]), "task_id": [1, 1], "split": ["train", "validation"]})
    groups = {"A": ["a"], "D": ["d"], "B": ["b"], "B_lab": [], "Gut_ordinal": [], "Gut_categorical": []}
    reg = {"G_FEATURES": ["g"], "FEATURE_GROUPS": groups}
    reads = []
    def values(k, path, columns):
        return k.assign(**{c: np.array([2., 3.]) for c in columns})
    def parquet(path, **kwargs):
        reads.append(Path(path).name)
        if Path(path).name == "comparison_cohorts.parquet":
            return keys[W.WINDOW_KEY].assign(cohort_core=True, cohort_gut=True)
        if Path(path).name == "static_features.parquet":
            return pd.DataFrame({"subject": [1], "b": [20.]})
        if Path(path).name == "base_residuals.parquet":
            return keys.iloc[:1].assign(oof_fold=2, residual=4.)
        raise AssertionError(path)
    fit = {"fits": {f"glucobench_linear_regression|{W.PROTOCOL_A}|{W.BASE_CONFIG}": {"model": None, "scaler": None, "inputs": ["x0"]}}}
    with patch.object(W, "load_keys", return_value=keys), patch.object(W, "load_values", side_effect=values),\
         patch.object(W, "load_minutes", return_value=pd.DataFrame()), patch.object(pd, "read_parquet", side_effect=parquet),\
         patch.object(W.joblib, "load", return_value=fit), patch.object(W, "gather_lags", return_value=np.ones((2, 1))),\
         patch.object(W, "predict_glucobench", return_value=np.ones(2)):
        upstream, lags = W.build_table(W.PROTOCOL_A, ("train", "validation"), registry=reg, load_residuals=False)
        assert "base_residuals.parquet" not in reads and not {"residual", "oof_fold"} & set(upstream)
        standard, default_lags = W.build_table(W.PROTOCOL_A, ("train", "validation"), registry=reg)
    assert reads.count("base_residuals.parquet") == 1 and lags == default_lags == ["x0"]
    pd.testing.assert_frame_equal(standard[upstream.columns], upstream, check_exact=True)
    assert standard["residual"].tolist() == [4., 2.] and standard.loc[0, "oof_fold"] == 2


def test_completed_result_tables_require_full_semantic_coverage():
    from unittest import TestCase
    plan = json.loads(LOCK.read_text(encoding="utf-8"))
    directory = ROOT / "results/factor_evaluation"
    completion = json.loads((directory / "completed_evaluation.json").read_text(encoding="utf-8"))
    gains, metrics, strata = [pd.read_csv(directory / f) for f in ("comparison_results.csv", "configuration_metrics.csv", "meal_strata.csv")]
    assert len(FL.validate_saved_results(plan, completion, gains, metrics, strata)) == 50
    for bad_gains, bad_metrics, bad_strata in [
        (gains.iloc[1:], metrics, strata), (pd.concat([gains, gains.iloc[:1]]), metrics, strata),
        (gains.assign(protocol_tag="B"), metrics, strata), (gains.drop(columns="from_config"), metrics, strata),
        (gains.assign(d_rmse=np.inf), metrics, strata), (gains.assign(p_bootstrap=np.nan), metrics, strata),
        (gains, metrics.iloc[1:], strata), (gains, metrics.assign(rmse_pooled=np.inf), strata),
        (gains, metrics, strata.iloc[1:])]:
        with TestCase().assertRaises(ValueError):
            FL.validate_saved_results(plan, completion, bad_gains, bad_metrics, bad_strata)


def test_contrast_p_value_reuses_interval_resamples_exactly():
    from unittest.mock import patch
    from src import evaluation as evaluation
    from src.residuals import contrast_row, subject_table
    first = subject_table(np.repeat([1, 2, 3], 20), np.tile([1., 2., 3.], 20))
    second = first.copy()
    second["rmse"] -= [0.1, 0.2, -0.1]
    second["sum_squared_error"] = second["rmse"] ** 2 * second["n_windows"]
    plain = contrast_row(first, second, 20, n_resamples=30)
    expected_p = bootstrap_p(first, second, 20, 30)
    with patch.object(evaluation, "draw_subjects", wraps=evaluation.draw_subjects) as draw:
        combined = contrast_row(first, second, 20, n_resamples=30, include_p=True)
        assert draw.call_count == 1
    assert combined.pop("p_bootstrap") == expected_p
    assert combined == plain


def test_runtime_boundaries_survive_optimization():
    from unittest import TestCase
    from src.baselines import gather_lags
    from src.openfe_inputs import internal_split_subjects
    from src.llm_fitness import time_split, group_splits, ExpressionFitness
    from src.grouped_cv import run_fold
    from src.selection import one_standard_error_choice
    from src.windows import assign_splits
    case = TestCase()
    grid = pd.DataFrame({"subject": [1] * 4, "timestamp": pd.date_range("2020", periods=4, freq="min"), "glucose": [11., 22., 33., 44.]})
    window = pd.DataFrame({"subject": [1], "origin_time": [grid.timestamp.iloc[1]]})
    case.assertEqual(gather_lags(grid, window, [-1, 0], "glucose").tolist(), [[11., 22.]])
    for origin, offsets in ((grid.timestamp.iloc[0], [-1, 0]), (grid.timestamp.iloc[-1] + pd.Timedelta(minutes=1), [0]), (grid.timestamp.iloc[1], [])):
        with case.assertRaises(ValueError):
            gather_lags(grid, window.assign(origin_time=origin), offsets, "glucose")
    with case.assertRaises(ValueError):
        gather_lags(grid.drop(index=1), window, [0], "glucose")
    for train, held in (([1], [1]), ([], [])):
        with case.assertRaises(ValueError):
            internal_split_subjects(window, train, held)
    bad = window.assign(split="validation")
    for call in (lambda: time_split(bad, None, None, []), lambda: group_splits(bad, []), lambda: ExpressionFitness(bad, [], {}, 1.)):
        with case.assertRaises(ValueError):
            call()
    for table, outer in ((pd.DataFrame({"subject": [1, 2], "residual": 0}), [1]), (pd.DataFrame({"subject": [1, 2]}), []), (pd.DataFrame({"subject": [1, 2]}), [1, 2])):
        with case.assertRaises(ValueError):
            run_fold(table, outer, None)
    paired = {0: pd.DataFrame({"rmse": [1., 2.]}, index=[1, 2]), 1: pd.DataFrame({"rmse": [1., 2.]}, index=[2, 1])}
    with case.assertRaises(ValueError):
        one_standard_error_choice(None, paired)
    scope = pd.DataFrame({"subject": [1], "start_time": [pd.Timestamp("2020-01-01")], "end_time": [pd.Timestamp("2020-01-04")],
                          "protocol_a_train_end_time": [pd.Timestamp("2020-01-02")], "protocol_a_val_end_time": [pd.Timestamp("2020-01-03")], "protocol_b_group": ["train"]})
    for history, target in (("2019-12-31", "2020-01-02"), ("2020-01-01", "2020-01-05")):
        with case.assertRaises(ValueError):
            assign_splits(pd.DataFrame({"subject": [1], "history_start_time": pd.to_datetime([history]), "target_time": pd.to_datetime([target])}), scope,
                          window_columns=[], labels=[], protocol_a_name="A", protocol_b_name="B", sample_key=[])


def test_validation_comparison_rejects_nonfinite_values_and_label_changes():
    from unittest import TestCase
    case = TestCase()
    row = {"id": "A:P1", "cohort": "core", "stratum": "all", "group": "all", "min_window_rule": "applied (>= 20 windows)",
           "subjects": 2, "validation_windows": 40, "rmse_from": 5., "rmse_to": 4., "d_rmse": -1., "d_rmse_low": -2., "d_rmse_high": 3.,
           "improved_share": 1., "pooled_d_rmse": -1., "d_mae": -1., "outcome": "improved"}
    plan = {"comparisons": [{"id": "A:P1", "evaluation_protocol": "A", "validation_reference": {"artifact": "reference.csv"}}]}
    def compare(changes, reference=None):
        result = FG.compare_validation_results(pd.DataFrame([{**row, **changes}]), plan, Path("unopened"), {"reference.csv": pd.DataFrame([row if reference is None else reference])})
        return result.iloc[0]["status"]
    case.assertEqual(compare({}), "match")
    for changes in ({"d_rmse_high": np.nan}, {"rmse_to": np.inf}, {"outcome": "worse"}, {"d_rmse_high": np.nan, "outcome": "worse"}):
        case.assertEqual(compare(changes), "MISMATCH")
    case.assertEqual(compare({}, {**row, "d_rmse_high": np.nan}), "MISMATCH")


def test_public_calculation_outputs_refuse_existing_directories_before_loading():
    import tempfile
    from unittest import TestCase
    from unittest.mock import patch
    from src import discovery_cli, lab_experiment, openfe_runner
    case = TestCase()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        out = root / "reproductions/existing"
        out.mkdir(parents=True)
        marker = out / "preserve.txt"
        marker.write_text("unchanged")
        with patch.object(openfe_runner.pd, "read_parquet", side_effect=AssertionError("input opened")):
            with case.assertRaises(ValueError):
                openfe_runner.run_search({"output_dir": str(out)})
            temporary = root / "openfe_tmp_synthetic.feather"
            temporary.write_text("preserve")
            with case.assertRaises(ValueError):
                openfe_runner.run_search({"output_dir": str(root / "new"), "tmp_dir": str(root), "tag": "synthetic"})
        with patch.object(discovery_cli.data, "PROJECT_ROOT", root), patch.object(discovery_cli.data, "load_context", side_effect=AssertionError("context loaded")):
            with case.assertRaises(SystemExit):
                discovery_cli.main(["e4", "--dry-run", "--output-dir", str(out)])
            with case.assertRaises(ValueError):
                discovery_cli.cmd_cv_folds(True, output_dir=out)
        with patch.object(lab_experiment, "ROOT", root), patch.object(lab_experiment, "OUT", None), patch.object(lab_experiment, "core", side_effect=AssertionError("fit started")):
            with case.assertRaises(SystemExit):
                lab_experiment.main(["--task", "core", "--output-dir", str(out)])
        case.assertEqual(marker.read_text(), "unchanged")
        case.assertEqual(len(list(out.iterdir())), 1)


def test_isolated_lab_core_then_selection_do_not_republish_results():
    import tempfile
    from unittest import TestCase
    from unittest.mock import patch
    from src import lab_experiment as lab
    case = TestCase()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "config.json"
        config.write_text(json.dumps({"execution_enabled": True, "data_access_policy": {"test_statistical_use": False}, "alpha_grid": lab.RIDGE_ALPHAS.tolist()}))
        out = root / "reproductions/lab"
        def core(_):
            lab.record(status="FITTED")
        def selection(_):
            lab.csv("selection_C.csv", [{"synthetic": 1}])
            lab.record(status="COMPLETED")
        with patch.object(lab, "ROOT", root), patch.object(lab, "CONFIG", config), patch.object(lab, "OUT", None), patch.object(lab, "core", side_effect=core), patch.object(lab, "selection", side_effect=selection):
            lab.main(["--task", "core", "--output-dir", str(out)])
            lab.main(["--task", "selection", "--output-dir", str(out)])
            before = (out / "selection_C.csv").read_bytes()
            with case.assertRaises(SystemExit):
                lab.main(["--task", "selection", "--output-dir", str(out)])
            case.assertEqual((out / "selection_C.csv").read_bytes(), before)


def test_best_alpha_reuses_identical_subject_statistics():
    from src.residuals import RIDGE_ALPHAS, best_alpha, subject_table
    rng = np.random.default_rng(7)
    subjects = np.array([3, 1, 2, 3, 1, 2])
    predictions = rng.normal(size=(6, len(RIDGE_ALPHAS)))
    base, target = rng.normal(size=6), rng.normal(size=6)
    best, alpha, table = best_alpha(predictions, base, target, subjects)
    pd.testing.assert_frame_equal(table, subject_table(subjects, base + predictions[:, best] - target), check_exact=True)


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

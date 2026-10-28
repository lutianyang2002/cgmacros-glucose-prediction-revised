"""Load admitted window keys, values and causal inputs for requested splits.

Default access permits TRAIN and VALIDATION. TEST requires both an approved lock
and explicit execution permission. Mixed-file filtering is statistical isolation."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.baselines import gather_lags, predict_glucobench
from src.factor_construction import CandidateContext, diet_features, generate_candidates, subject_primitives
from src.paths import PROJECT_ROOT

DATA_QUALITY_DIR = PROJECT_ROOT / "results" / "data_quality"
ALIGNED_DIR = PROJECT_ROOT / "results" / "aligned_data"
WINDOWS_DIR = PROJECT_ROOT / "results" / "forecast_windows"
BASELINE_DIR = PROJECT_ROOT / "results" / "baselines"
MANUAL_DIR = PROJECT_ROOT / "results" / "manual_ablation"
FACTOR_DIR = PROJECT_ROOT / "results" / "factor_discovery"
PROTOCOL_A, PROTOCOL_B = "Within-subject Future Prediction", "Unseen-subject Generalization"
WINDOW_KEY = ["subject", "origin_time", "task_id"]
EVENT_COLUMNS = ["meal_type_normalized", "carbs", "protein", "fat", "fiber", "meal_calories"]
BASE_CONFIG = "in_len=24"
PRE_LOCK_SPLITS = ("train", "validation")


class TestAccessDenied(RuntimeError):
    """The test split was requested although the lock is not approved or `execute_test` is False."""


@dataclass(frozen=True)
class Access:
    """What the caller may read. The default permits train and validation only."""
    execute_test: bool = False
    lock_approved: bool = False

    def permits_test(self) -> bool:
        return bool(self.execute_test and self.lock_approved)


def check_access(splits, access: Access) -> tuple:
    splits = tuple(splits)
    unknown = set(splits) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown split names {sorted(unknown)}")
    if "test" in splits and not access.permits_test():
        raise TestAccessDenied("the test split cannot be read: the lock is not approved or EXECUTE_TEST is False")
    return splits


# ------------------------------------------------------------------ registries (no window data)
def registries() -> dict:
    """Feature groups and glucose-history feature names."""
    g_features = pd.read_csv(MANUAL_DIR / "feature_group_registry.csv").query("record_type == 'glucose_history_feature'")["candidate_feature"].tolist()
    ad = pd.read_csv(MANUAL_DIR / "activity_diet_feature_registry.csv")
    coverage = pd.read_csv(MANUAL_DIR / "static_feature_coverage.csv")
    groups = {
        "A": ad.loc[ad["group_code"].eq("A"), "feature_name"].tolist(),
        "D": ad.loc[ad["group_code"].eq("D") & ad["variant"].eq("primary"), "feature_name"].tolist(),
        "B": coverage.loc[coverage["role"].eq("core") & ~coverage["encoding"].eq("availability flag"), "feature_name"].tolist(),
        "B_lab": coverage.loc[coverage["role"].str.startswith("offline laboratory") & coverage["record_type"].eq("feature"), "feature_name"].tolist(),
        "Gut_ordinal": coverage.loc[coverage["feature_name"].str.startswith("gut_ord_"), "feature_name"].tolist(),
        "Gut_categorical": coverage.loc[coverage["feature_name"].str.startswith("gut_cat_"), "feature_name"].tolist(),
    }
    manual = groups["A"] + groups["D"] + groups["B"]
    if len(manual) != 50 or len(groups["B"]) != 3:
        raise ValueError("manual feature registry must contain 50 features, including three baseline covariates")
    return {"G_FEATURES": g_features, "FEATURE_GROUPS": groups, "MANUAL": manual}


# ------------------------------------------------------------------ loaders
def load_keys(protocol: str, splits, access: Access = Access()) -> pd.DataFrame:
    """Keys and split tags of the admitted windows of `splits` (no labels, no features)."""
    splits = check_access(splits, access)
    rows = pd.read_parquet(WINDOWS_DIR / "fold_assignments.parquet", columns=["evaluation_protocol", "split", "subject", "origin_time", "task_id", "history_start_time", "target_time"],
                           filters=[("evaluation_protocol", "==", protocol), ("admitted_to_split", "==", True), ("split", "in", list(splits))])
    if not set(rows["split"]) <= set(splits):
        raise ValueError("loaded keys include an unrequested split")
    if "test" not in splits and protocol == PROTOCOL_B:
        scope = pd.read_csv(DATA_QUALITY_DIR / "split_scope_v1.csv")
        if set(rows["subject"]) & set(scope.loc[scope["protocol_b_group"].eq("test"), "subject_id"]):
            raise ValueError("unseen-subject TRAIN/VALIDATION keys include TEST subjects")
    return rows.reset_index(drop=True)


def load_values(keys: pd.DataFrame, path, columns: list) -> pd.DataFrame:
    """Labels or features of exactly the admitted keys (the read filter lists their origins per subject)."""
    if keys.empty or keys[WINDOW_KEY].isna().any().any():
        raise ValueError("load_values requires nonempty complete window keys; no file was opened")
    dataset = ds.dataset(path)
    origin_type = dataset.schema.field("origin_time").type
    keep = None
    for subject, part in keys.groupby("subject"):
        rows = None
        for task, task_keys in part.groupby("task_id"):
            task_rows = ((ds.field("subject") == int(subject)) & (ds.field("task_id") == task)
                         & ds.field("origin_time").isin(pa.array(task_keys["origin_time"]).cast(origin_type)))
            rows = task_rows if rows is None else rows | task_rows
        keep = rows if keep is None else keep | rows
    values = dataset.to_table(columns=[*WINDOW_KEY, *columns], filter=keep).to_pandas()
    merged = keys.merge(values, on=WINDOW_KEY, how="left", validate="one_to_one", indicator=True)
    if len(values) != len(keys) or not merged["_merge"].eq("both").all():
        raise ValueError("loaded values do not match the requested window keys")
    return merged.drop(columns="_merge")


def load_minutes(keys: pd.DataFrame, path, columns: list, time_field: str = "timestamp") -> pd.DataFrame:
    """Minute-level or event rows up to each subject's last admitted origin of `keys`; nothing later is read."""
    if keys.empty or keys[["subject", "origin_time"]].isna().any().any():
        raise ValueError("load_minutes requires nonempty complete subject/time keys; no file was opened")
    last = keys.groupby("subject")["origin_time"].max()
    keep = None
    for subject, origin in last.items():
        rows = (ds.field("subject") == int(subject)) & (ds.field(time_field) <= origin.to_pydatetime())
        keep = rows if keep is None else keep | rows
    frame = ds.dataset(path).to_table(columns=["subject", time_field, *columns], filter=keep).to_pandas()
    if not frame.merge(last.rename("last"), left_on="subject", right_index=True).eval(f"{time_field} <= last").all():
        raise ValueError("minute inputs extend beyond an admitted prediction origin")
    return frame.sort_values(["subject", time_field]).reset_index(drop=True)


def load_inputs(keys: pd.DataFrame, splits, access: Access = Access()) -> tuple:
    """Per-subject minute frames (Libre, heart rate, activity calories, MET, source flags) and meal events, capped at the last admitted origin of `keys`."""
    check_access(splits, access)
    dynamic = load_minutes(keys, ALIGNED_DIR / "dynamic_aligned_1min.parquet", ["libre_gl", "source_segment_id", "is_inserted_grid_row"])
    activity = load_minutes(keys, ALIGNED_DIR / "activity_canonical.parquet", ["heart_rate_bpm", "activity_calories_per_minute", "activity_met", "is_published_source_row"])
    minutes = dynamic.merge(activity, on=["subject", "timestamp"], validate="one_to_one")
    if not len(minutes) == len(dynamic) == len(activity):
        raise ValueError("dynamic and activity inputs have different minute keys")
    events = load_minutes(keys, ALIGNED_DIR / "meal_events.parquet", EVENT_COLUMNS)
    return {s: g.reset_index(drop=True) for s, g in minutes.groupby("subject")}, {s: g for s, g in events.groupby("subject")}


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime([]), **{c: [] for c in EVENT_COLUMNS}})


def bmi_of(subjects) -> pd.Series:
    """BMI of the subjects of the requested windows only (the covariate file is cut to them immediately)."""
    covariates = pd.read_csv(FACTOR_DIR / "metabolic_covariates.csv")
    return covariates[covariates["subject"].isin(set(subjects))].set_index("subject")["bmi_kg_m2"]


# ------------------------------------------------------------------ the modeling table
def build_table(protocol: str, splits, access: Access = Access(), registry: dict | None = None, factor_dir=None, *, load_residuals: bool = True) -> tuple:
    """Admitted windows with labels, glucose lags, the existing A/D/B features, cohorts, the stored base prediction and the training residuals for the requested splits.

    Training rows carry the out-of-fold residual of the configured OOF design (read from `base_residuals.parquet`, which holds training windows only for this purpose); the other rows get
    the residual of the base fitted on the whole training set. With ``load_residuals=False``,
    return upstream inputs only: no residual or OOF-fold columns and no residual-file read.
    """
    splits = check_access(splits, access)
    reg = registries() if registry is None else registry
    keys = load_keys(protocol, splits, access)
    table = load_values(keys, WINDOWS_DIR / "glucose_history_samples.parquet", ["target_value", *reg["G_FEATURES"]])
    manual = load_values(keys, MANUAL_DIR / "activity_diet_features.parquet", reg["FEATURE_GROUPS"]["A"] + reg["FEATURE_GROUPS"]["D"])
    cohorts = pd.read_parquet(MANUAL_DIR / "comparison_cohorts.parquet", columns=[*WINDOW_KEY, "cohort_core", "cohort_gut"], filters=[("evaluation_protocol", "==", protocol), ("split", "in", list(splits))])
    static = pd.read_parquet(MANUAL_DIR / "static_features.parquet")
    g = reg["FEATURE_GROUPS"]
    table = (table.merge(manual[WINDOW_KEY + g["A"] + g["D"]], on=WINDOW_KEY, validate="one_to_one").merge(cohorts, on=WINDOW_KEY, validate="one_to_one")
                  .merge(static[["subject", *g["B"], *g["B_lab"], *g["Gut_ordinal"], *g["Gut_categorical"]]], on="subject", validate="many_to_one"))
    fit = joblib.load(BASELINE_DIR / "baseline_fits.joblib")["fits"][f"glucobench_linear_regression|{protocol}|{BASE_CONFIG}"]
    grid = load_minutes(keys, ALIGNED_DIR / "dynamic_aligned_1min.parquet", ["libre_gl"])
    offsets = [int(c[1:]) for c in fit["inputs"]]
    table = pd.concat([table, pd.DataFrame(gather_lags(grid, table, offsets, "libre_gl"), columns=fit["inputs"])], axis=1)
    table["base_prediction"] = predict_glucobench(fit["model"], fit["scaler"], table, fit["inputs"])
    table = table.sort_values(["subject", "origin_time"]).reset_index(drop=True)
    if len(table) != len(keys):
        raise ValueError("model table lost admitted windows")
    if not load_residuals:
        return table, list(fit["inputs"])
    residuals = pd.read_parquet((FACTOR_DIR if factor_dir is None else Path(factor_dir)) / "base_residuals.parquet", columns=[*WINDOW_KEY, "split", "oof_fold", "residual"], filters=[("evaluation_protocol", "==", protocol), ("split", "==", "train")])
    table = table.merge(residuals.rename(columns={"residual": "oof_residual"})[[*WINDOW_KEY, "split", "oof_residual", "oof_fold"]], on=[*WINDOW_KEY, "split"], how="left", validate="one_to_one")
    table["residual"] = np.where(table["split"].eq("train"), table["oof_residual"], table["target_value"] - table["base_prediction"])
    return table.drop(columns="oof_residual"), list(fit["inputs"])


# ------------------------------------------------------------------ rebuilt factors
def primitives_table(keys: pd.DataFrame, minutes: dict, events: dict) -> pd.DataFrame:
    """Window-level primitives (meal, activity, inferred native-reading state) of every window of `keys`."""
    empty = empty_events()
    parts = []
    for subject, rows in keys[WINDOW_KEY + ["split"]].groupby("subject"):
        primitives = subject_primitives(minutes[subject], events.get(subject, empty), rows["origin_time"])
        parts.append(pd.concat([rows, primitives], axis=1))
    return pd.concat(parts, ignore_index=True)


def closure(recipes: pd.DataFrame, wanted: list) -> pd.DataFrame:
    """The wanted recipes together with the candidates they are built on."""
    ids, stack = set(), list(wanted)
    index = recipes.set_index("candidate_id")
    while stack:
        cid = stack.pop()
        if cid not in ids:
            ids.add(cid)
            stack.extend(json.loads(index.loc[cid, "inputs"]))
    return recipes[recipes["candidate_id"].isin(ids)]


def candidate_matrix(keys: pd.DataFrame, minutes: dict, events: dict, recipes: pd.DataFrame, wanted: list, bmi: pd.Series, version: str = "logged") -> pd.DataFrame:
    """Window values of the selected candidate recipes (and the candidates they are built on) with the meal timing `version`."""
    empty = empty_events()
    needed = closure(recipes, wanted)
    parts = []
    for subject, rows in keys.groupby("subject"):
        context = CandidateContext(minutes[subject], events.get(subject, empty), rows["origin_time"], bmi[subject], version)
        values = generate_candidates(needed, context)
        values = values[[c for c in values.columns if c in wanted]]
        values.index = rows.index
        parts.append(pd.concat([rows, values], axis=1))
    return pd.concat(parts).sort_index()


def diet_matrix(keys: pd.DataFrame, minutes: dict, events: dict, d_columns: list, version: str = "logged") -> pd.DataFrame:
    """The manual diet features under a meal-class availability delay."""
    empty = empty_events()
    parts = []
    for subject, rows in keys.groupby("subject"):
        frame = minutes[subject]
        position = ((rows["origin_time"] - frame["timestamp"].iloc[0]) // pd.Timedelta(minutes=1)).to_numpy()
        values = diet_features(events.get(subject, empty), frame["timestamp"].iloc[0], len(frame), position, version)
        parts.append(pd.concat([rows, pd.DataFrame({c: values[c] for c in d_columns}, index=rows.index)], axis=1))
    return pd.concat(parts).sort_index()

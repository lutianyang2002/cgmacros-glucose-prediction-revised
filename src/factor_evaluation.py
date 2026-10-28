"""Evaluate factor comparisons using fixed penalties and paired subject-level statistics.

Validation reconstruction uses saved references. TEST requires explicit approved access.
Core and full cohorts, the 20-window rule, 2,000 paired subject bootstrap draws,
and protocol-specific four-comparison Holm families retain their scientific definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import e8_anchor as E8
from src.evaluation import classify_effect
from src.residuals import contrast_row, ridge_path_predictions, subject_table
from src.window_data import WINDOW_KEY

MIN_WINDOWS = 20
N_RESAMPLES = 2000
SEED = 42
THRESHOLD = 0.1
STRATA = ("P30", "P31_120", "NoMeal120")
RULES = {"applied (>= 20 windows)": MIN_WINDOWS, "not applied": None}


# ------------------------------------------------------------------ interpretation labels (columns added to the result tables; no value is computed or changed)
P_VALUE_METHOD = "exploratory approximate subject-bootstrap"
INFERENCE_LABEL_COLUMNS = ["p_value_method", "nominal_fwer_control_verified", "test_split_independent", "inference_role"]


def inference_role(role: str) -> str:
    """Interpret comparison roles descriptively, including legacy confirmatory labels."""
    role = str(role)
    if role.startswith(("confirmatory", "primary")):
        return "primary descriptive"
    return "exploratory sensitivity" if "exploratory" in role else "secondary descriptive"


def add_inference_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds the interpretation columns. `role` is kept as it is for the program; the evaluation split is never an independent hold-out (validation in validation reconstruction, a previously used test split in authorized test evaluation)."""
    if not len(frame):
        return frame
    has_p = frame.get("p_bootstrap", pd.Series(np.nan, index=frame.index)).notna() | frame.get("p_holm_primary_family", pd.Series(np.nan, index=frame.index)).notna()
    out = frame.copy()
    out["p_value_method"] = np.where(has_p, P_VALUE_METHOD, "")
    out["nominal_fwer_control_verified"] = False
    out["test_split_independent"] = False
    out["inference_role"] = out["role"].map(inference_role)
    return out


# ------------------------------------------------------------------ configuration -> columns
def resolve_columns(columns: list, manual: list, groups: dict) -> list:
    """Expand the placeholders of the lock: `<M>` is the 50 existing features, `<groups G+A+D>` the columns of the named groups (G is the base itself)."""
    out = []
    for c in columns:
        if c == "<M>":
            out += manual
        elif c.startswith("<groups "):
            names = c[len("<groups "):-1].split("+")
            out += [col for g in names if g != "G" for col in groups[g]]
        else:
            out.append(c)
    if len(out) != len(set(out)):
        raise ValueError("a configuration lists a column twice")
    return out


def strata_masks(table: pd.DataFrame) -> dict:
    """Post-meal strata from the logged-time diet features (information before the origin only): P30 = a meal logged in the last 30 minutes, P31-120 = none in 30 but one in 120,
    NoMeal120 = none in 120."""
    no30, no120 = table["diet_no_logged_meal_30m"].to_numpy() == 1, table["diet_no_logged_meal_120m"].to_numpy() == 1
    return {"P30": ~no30, "P31_120": no30 & ~no120, "NoMeal120": no120}


# ------------------------------------------------------------------ predictions
class Evaluator:
    """Fits the configured models of one protocol and keeps their window-level errors on the evaluation split.

    `table` holds the train windows and the evaluation windows (`eval_split`) with the existing features, the factor columns, the stored base prediction and the training residual;
    `variants` the meal-timing variants of E9 (`{version: frame aligned to table with the diet and T columns}`); `anchored` the E8 columns `anchor_base`, `anchor_residual`,
    `anchor_reliable` (NaN / False where the lags cannot be rebuilt).
    """

    def __init__(self, table: pd.DataFrame, configs: dict, manual: list, groups: dict, eval_split: str, variants: dict | None = None, anchored: pd.DataFrame | None = None):
        self.table, self.configs, self.manual, self.groups, self.eval_split = table, configs, manual, groups, eval_split
        self.variants, self.anchored = variants or {}, anchored
        self.cache = {}

    def _frame(self, spec: dict) -> pd.DataFrame:
        """The table a configuration works on (its meal-timing variant, its base columns)."""
        version = spec.get("meal_version", "logged")
        frame = self.table
        if version != "logged":
            frame = frame.copy()
            for c in self.variants[version].columns:
                frame[c] = self.variants[version][c]
        if spec.get("lags") == "anchor_causal":
            frame = frame.copy()
            frame["base_prediction"], frame["residual"] = self.anchored["anchor_base"], self.anchored["anchor_residual"]
        return frame

    def _rows(self, spec: dict, cohort: str) -> tuple:
        frame = self.table
        in_cohort = frame["cohort_core"].to_numpy(bool) if cohort == "core" else np.ones(len(frame), bool)
        reliable = self.anchored["anchor_reliable"].to_numpy(bool) if spec.get("windows") == "core_and_anchor_reliable" or spec.get("lags") == "anchor_causal" else np.ones(len(frame), bool)
        train = frame["split"].eq("train").to_numpy() & in_cohort
        evaluation = frame["split"].eq(self.eval_split).to_numpy() & in_cohort
        if spec.get("windows") == "core_and_anchor_reliable" or spec.get("lags") == "anchor_causal":
            train, evaluation = train & reliable, evaluation & reliable
        return train, evaluation

    def errors_of(self, config: str, cohort: str = "core") -> pd.DataFrame:
        """Window-level error (prediction minus label) of a configuration on the evaluation windows of a cohort: columns subject, error."""
        key = (config, cohort)
        if key in self.cache:
            return self.cache[key]
        spec = self.configs[config]
        frame = self._frame(spec)
        train, evaluation = self._rows(spec, cohort)
        columns = resolve_columns(spec["columns"], self.manual, self.groups)
        train = train & frame["residual"].notna().to_numpy()
        ev = frame[evaluation]
        if columns:
            tr = frame[train]
            prediction = ridge_path_predictions(tr[columns], tr["residual"].to_numpy(float), ev[columns], alphas=[spec["alpha"]])[:, 0]
        else:
            prediction = np.zeros(len(ev))
        error = ev["base_prediction"].to_numpy(float) + prediction - ev["target_value"].to_numpy(float)
        out = pd.DataFrame({"subject": ev["subject"].to_numpy(), "error": error}, index=ev.index)
        self.cache[key] = out
        return out

    def subject_tables(self, config: str, cohort: str = "core", mask: np.ndarray | None = None) -> pd.DataFrame:
        e = self.errors_of(config, cohort)
        if mask is not None:
            e = e[mask[self.table.index.get_indexer(e.index)]]
        return subject_table(e["subject"].to_numpy(), e["error"].to_numpy())


# ------------------------------------------------------------------ statistics


def holm(p_values: list) -> list:
    """Holm step-down adjusted p-values."""
    if not np.isfinite(p_values).all() or any(p < 0 or p > 1 for p in p_values):
        raise ValueError("Holm requires finite p-values in [0, 1]; the specified family cannot be reduced")
    order = np.argsort(p_values)
    m, adjusted, running = len(p_values), [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def contrast_rows(evaluator: Evaluator, comparison: dict, cohort: str, stratum: str, group_subjects: set | None, n_resamples: int, mask: np.ndarray | None = None) -> list:
    """The paired contrast of one comparison, with and without the 20-window rule."""
    errors_from = evaluator.errors_of(comparison["from_config"], cohort)
    errors_to = evaluator.errors_of(comparison["to_config"], cohort)
    if not errors_from.index.equals(errors_to.index):
        raise ValueError("from/to errors must use identical prediction keys")
    t_from = evaluator.subject_tables(comparison["from_config"], cohort, mask)
    t_to = evaluator.subject_tables(comparison["to_config"], cohort, mask)
    if group_subjects is not None:
        t_from, t_to = t_from[t_from.index.isin(group_subjects)], t_to[t_to.index.isin(group_subjects)]
    rows = []
    for rule, minimum in RULES.items():
        n_ok = int(((t_from["n_windows"] >= (minimum or 0)) & (t_to["n_windows"] >= (minimum or 0))).sum())
        base = {"id": comparison["id"], "evaluation_protocol": comparison["evaluation_protocol"], "family": comparison["family"], "role": comparison["role"], "from_config": comparison["from_config"],
                "to_config": comparison["to_config"], "cohort": cohort, "stratum": stratum, "group": "all" if group_subjects is None else "subgroup", "min_window_rule": rule}
        if n_ok < 2:
            rows.append({**base, "subjects": n_ok, "note": "fewer than two subjects: no interval"})
            continue
        row = contrast_row(t_from, t_to, minimum, n_resamples=n_resamples, seed=SEED,
                           include_p=stratum == "all" and group_subjects is None)
        p_bootstrap = row.pop("p_bootstrap", np.nan)
        pooled_from, pooled_to = t_from, t_to
        if minimum is not None:
            keep = t_from.index[(t_from["n_windows"] >= minimum) & (t_to["n_windows"] >= minimum)]
            pooled_from, pooled_to = t_from.loc[keep], t_to.loc[keep]
        pooled = {"pooled_rmse_from": float(np.sqrt(pooled_from["sum_squared_error"].sum() / pooled_from["n_windows"].sum())),
                  "pooled_rmse_to": float(np.sqrt(pooled_to["sum_squared_error"].sum() / pooled_to["n_windows"].sum()))}
        rows.append({**base, **row, **pooled, "p_bootstrap": p_bootstrap, "note": ""})
    return rows


def run_comparisons(evaluator: Evaluator, comparisons: list, hba1c_groups: pd.Series | None = None, n_resamples: int = N_RESAMPLES, strata_ids: tuple = ("P1", "P2", "P3", "P4", "S1", "S2", "S3"),
                    group_ids: tuple = ("P1", "P2", "P3", "P4")) -> pd.DataFrame:
    """Every comparison of one protocol on the core cohort (all windows, strata for the primary and the secondary E2 contrasts, HbA1c groups for the primary ones) and on the full cohort (all windows)."""
    rows = []
    masks = strata_masks(evaluator.table)
    for c in comparisons:
        short = c["id"].split(":")[1]
        rows += contrast_rows(evaluator, c, "core", "all", None, n_resamples)
        rows += contrast_rows(evaluator, c, "full", "all", None, n_resamples)
        if short in strata_ids:
            for name, mask in masks.items():
                rows += contrast_rows(evaluator, c, "core", name, None, n_resamples, mask)
        if hba1c_groups is not None and short in group_ids:
            for g, members in hba1c_groups.groupby(hba1c_groups, observed=True).groups.items():
                for r in contrast_rows(evaluator, c, "core", "all", set(members), n_resamples):
                    rows.append({**r, "group": str(g)})
    out = pd.DataFrame(rows)
    primary = out[out["family"].eq("primary") & (out["cohort"] == "core") & (out["stratum"] == "all") & (out["group"] == "all") & out["min_window_rule"].str.startswith("applied")]
    out["p_holm_primary_family"] = np.nan
    if len(primary):
        out.loc[primary.index, "p_holm_primary_family"] = holm(primary["p_bootstrap"].tolist())
    out["practical_label"] = [classify_effect(d, s, THRESHOLD) if pd.notna(d) else "" for d, s in zip(out.get("d_rmse", np.nan), out.get("improved_share", np.nan))]
    return add_inference_labels(out)


def config_summary(evaluator: Evaluator, configs: list) -> pd.DataFrame:
    """Subject-equal and pooled RMSE of every configuration on the core evaluation windows (and on all windows)."""
    rows = []
    for cohort in ("core", "full"):
        for cfg in configs:
            t = evaluator.subject_tables(cfg, cohort)
            kept = t[t["n_windows"] >= MIN_WINDOWS]
            rows.append({"config": cfg, "cohort": cohort, "subjects": len(t), "subjects_ge_20": len(kept), "windows": int(t["n_windows"].sum()), "rmse_subject_equal_rule_applied": float(kept["rmse"].mean()),
                         "rmse_subject_equal_all_subjects": float(t["rmse"].mean()), "rmse_pooled": float(np.sqrt(t["sum_squared_error"].sum() / t["n_windows"].sum())), "mae_subject_equal_all_subjects": float(t["mae"].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E8 columns
def anchor_columns(table: pd.DataFrame, minutes: dict, lags: list, eval_split: str, protocol_is_a: bool, bounds: pd.DataFrame | None) -> pd.DataFrame:
    """E8: anchor-causal base prediction, residual and reliability of every window (development-data analysis functions; the evaluation split is relabeled `validation` for them)."""
    aliased = table.assign(split=table["split"].replace({eval_split: "validation"}))
    rebuilt, info = E8.rebuild_lags(minutes, table[["subject", "origin_time"]])
    anchored = E8.anchor_residuals(aliased, rebuilt, info, lags, protocol_is_a, bounds)
    out = pd.DataFrame({"anchor_base": np.nan, "anchor_residual": np.nan, "anchor_reliable": info["reliable"].to_numpy()}, index=table.index)
    out.loc[anchored.index, "anchor_base"] = anchored["base_prediction"]
    out.loc[anchored.index, "anchor_residual"] = anchored["residual"]
    return out


# ------------------------------------------------------------------ assembling the windows of one protocol
def assemble_protocol(tag: str, lock: dict, splits=("train", "validation"), access=None) -> dict:
    """Window table with every column required by the configured models, built from the raw artifacts with the shared factor implementations.

    `splits` = ("train", eval split). With ("train", "validation") this is the validation reconstruction dry run; with ("train", "test") it is the authorized test evaluation call, which `window_data.check_access` refuses
    unless the lock is approved and `execute_test` is set.
    """
    from src import window_data as W
    from src.openfe_runner import evaluate_formula
    from src.stability import metabolic_group

    access = access or W.Access()
    name = W.PROTOCOL_A if tag == "A" else W.PROTOCOL_B
    splits = W.check_access(splits, access)
    eval_split = [s for s in splits if s != "train"]
    if len(eval_split) != 1:
        raise ValueError("exactly one evaluation split is required")
    eval_split = eval_split[0]
    reg = W.registries()
    table, lags = W.build_table(name, splits, access, registry=reg)
    keys = table[[*WINDOW_KEY, "split"]]
    minutes, events = W.load_inputs(keys, splits, access)
    bmi = W.bmi_of(table["subject"].unique())
    features = lock["features"]
    t_ids = features["T"][tag]["ids_in_selection_order"]
    recipes = pd.DataFrame(features["T"][tag]["recipes_including_inputs"])
    t_values = W.candidate_matrix(keys, minutes, events, recipes, t_ids, bmi, "logged")
    for c in t_ids:
        table[c] = t_values[c].to_numpy()
    primitives = W.primitives_table(keys, minutes, events)
    if not primitives[WINDOW_KEY + ["split"]].reset_index(drop=True).equals(keys.reset_index(drop=True)):
        raise ValueError("primitives and table rows differ")
    table["native_missing"] = primitives["native_missing"].to_numpy()
    inputs = table.drop(columns=[c for c in features["openfe_columns_from_primitives"] if c in table.columns])
    inputs = pd.concat([inputs, primitives[features["openfe_columns_from_primitives"]]], axis=1)
    for c, formula in features["O"][tag]["formulas"].items():
        table[c] = evaluate_formula(formula, inputs).to_numpy(dtype="float32")
    d_columns = reg["FEATURE_GROUPS"]["D"]
    variants = {}
    for version in ("delay_free", "delay_all"):
        diet = W.diet_matrix(keys, minutes, events, d_columns, version)
        factors = W.candidate_matrix(keys, minutes, events, recipes, t_ids, bmi, version)
        variants[version] = pd.concat([diet[d_columns], factors[t_ids]], axis=1)
    bounds = None
    if tag == "A":
        scope = pd.read_csv(W.DATA_QUALITY_DIR / "split_scope_v1.csv", parse_dates=["start_time", "protocol_a_train_end_time"]).set_index("subject_id")
        bounds = pd.DataFrame({"lo": scope["start_time"], "hi": scope["protocol_a_train_end_time"]})
    anchored = anchor_columns(table, minutes, lags, eval_split, tag == "A", bounds)
    covariates = pd.read_csv(W.FACTOR_DIR / "metabolic_covariates.csv")
    groups = metabolic_group(covariates[covariates["subject"].isin(set(table.loc[table["split"] == eval_split, "subject"]))].set_index("subject")["hba1c_percent"])
    last_minute = pd.Series({s: f["timestamp"].max() for s, f in minutes.items()})
    last_origin = table.groupby("subject")["origin_time"].max()
    return {"tag": tag, "name": name, "table": table, "lags": lags, "variants": variants, "anchored": anchored, "hba1c_groups": groups, "eval_split": eval_split, "manual": reg["MANUAL"],
            "groups": reg["FEATURE_GROUPS"], "inputs": inputs, "last_minute_loaded": last_minute, "last_origin": last_origin, "keys": keys, "minutes": minutes, "events": events}


def evaluate_protocol(tag: str, lock: dict, assembled: dict, n_resamples: int = N_RESAMPLES) -> tuple:
    """Results of one protocol: (paired contrasts, configuration summary)."""
    configs = lock["configs"][tag]
    evaluator = Evaluator(assembled["table"], configs, assembled["manual"], assembled["groups"], assembled["eval_split"], assembled["variants"], assembled["anchored"])
    comparisons = [c for c in lock["comparisons"] if c["protocol_tag"] == tag]
    contrasts = run_comparisons(evaluator, comparisons, assembled["hba1c_groups"], n_resamples)
    used = sorted({c["from_config"] for c in comparisons} | {c["to_config"] for c in comparisons})
    return contrasts, config_summary(evaluator, used)


def run_fixed_evaluation(lock: dict, eval_split: str, access=None, n_resamples: int = N_RESAMPLES) -> dict:
    """Evaluate the full comparison plan of both protocols on `eval_split`: `validation` for the validation reconstruction dry run, `test` for authorized test evaluation (refused by `window_data.check_access` unless the lock is
    approved and the caller has set `execute_test`)."""
    from src import window_data as W

    access = access or W.Access()
    W.check_access(("train", eval_split), access)
    out = {}
    for tag in ("A", "B"):
        assembled = assemble_protocol(tag, lock, ("train", eval_split), access)
        contrasts, summary = evaluate_protocol(tag, lock, assembled, n_resamples)
        out[tag] = {"assembled": assembled, "contrasts": contrasts, "summary": summary}
    return out

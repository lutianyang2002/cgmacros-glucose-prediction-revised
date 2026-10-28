"""Laboratory-state interaction inputs: read-only mixed-source isolation and laboratory interaction construction.

The caller writes results only to its isolated calculation directory.
Mixed objects are filtered by admitted window keys before feature statistics are examined.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.baselines import gather_lags
from src.factor_construction import conditional_lab_catalog, interact

KEY = ["subject", "origin_time", "task_id"]
FOLD_KEY = ["evaluation_protocol", *KEY]
SPLITS = {"train", "validation"}
MEALS = ("breakfast", "lunch", "dinner", "snack")
STATES = {"fasting_glucose": "fasting_glucose_mg_dl", "hba1c_percent": "hba1c_percent", "homa_ir": "homa_ir"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_trainval(table):
    if "split" not in table or not set(table["split"]).issubset(SPLITS):
        raise ValueError("Only explicitly isolated TRAIN/VALIDATION objects are accepted")
    if table.empty:
        raise ValueError("Empty train/validation input")


def isolate_keyed(raw, keys, columns, on=KEY, check_split=False):
    """No statistics of payload columns before key filtering; return a new object.

    Partition conflicts are checked after matching fixed keys, not hidden by a
    split filter. Raw keys/row counts are partition metadata, not performance.
    """
    assert_trainval(keys)
    wanted = pd.MultiIndex.from_frame(keys[on].drop_duplicates())
    selected = raw.loc[pd.MultiIndex.from_frame(raw[on]).isin(wanted)].copy()
    if selected.duplicated(on).any():
        raise ValueError("Duplicate source keys")
    if check_split:
        check = keys[on + ["split"]].merge(selected[on + ["split"]], on=on, how="left", suffixes=("", "_stored"), validate="one_to_one")
        if not check["split"].eq(check["split_stored"]).all():
            raise ValueError("Input split conflicts with the split registry")
    selected = selected[on + columns].copy()
    merged = keys[on].merge(selected, on=on, how="left", validate="many_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Missing source rows for admitted TRAIN/VALIDATION keys")
    return merged.drop(columns="_merge").reset_index(drop=True)


class Loader:
    def __init__(self, root):
        self.root = Path(root)

    def csv(self, rel):
        return pd.read_csv(self.root / rel)

    def keyed(self, rel, keys, columns, on=KEY, split=False):
        cols = list(dict.fromkeys(on + columns + (["split"] if split else [])))
        raw = pd.read_parquet(self.root / rel, columns=cols)
        isolated = isolate_keyed(raw, keys, columns, on, split)
        del raw
        return isolated

    def fixed_keys(self, protocols):
        rel = "results/forecast_windows/fold_assignments.parquet"
        cols = FOLD_KEY + ["split", "admitted_to_split", "history_start_time", "target_time"]
        raw = pd.read_parquet(self.root / rel, columns=cols)
        # Partition metadata only. No target values are loaded here.
        keys = raw.loc[raw["admitted_to_split"] & raw["split"].isin(SPLITS) & raw["evaluation_protocol"].isin(protocols.values())].copy()
        del raw
        keys = keys.drop(columns="admitted_to_split").sort_values(FOLD_KEY).reset_index(drop=True)
        if keys.duplicated(FOLD_KEY).any():
            raise ValueError("Duplicate window keys")
        scope = self.csv("results/data_quality/split_scope_v1.csv").set_index("subject_id")
        for c in ["start_time", "end_time", "protocol_a_train_end_time", "protocol_a_val_end_time"]:
            scope[c] = pd.to_datetime(scope[c])
        for tag, name in protocols.items():
            k = keys[keys.evaluation_protocol.eq(name)]
            for split in SPLITS:
                p = k[k.split.eq(split)]
                if tag == "B":
                    expected = set(scope.index[scope.protocol_b_group.eq("train" if split == "train" else "val")])
                    if set(p.subject) != expected:
                        raise ValueError("B registry membership conflict")
                else:
                    lo = p.subject.map(scope.start_time if split == "train" else scope.protocol_a_train_end_time)
                    hi = p.subject.map(scope.protocol_a_train_end_time if split == "train" else scope.protocol_a_val_end_time)
                    ok = (p.history_start_time >= lo) if split == "train" else (p.history_start_time > lo)
                    if not (ok & p.target_time.le(hi)).all():
                        raise ValueError("A window support conflicts with defined time boundaries")
        return keys

    def build(self, config):
        keys = self.fixed_keys(config["protocols"])
        manual = config["reference"]["manual_columns"]
        lab_old = config["reference"]["E6_original_added_columns"]
        ad = [c for c in manual if not c.startswith("bio_")]
        basecols = ["target_value", "base_prediction", "residual", "oof_fold"]
        base = self.keyed("results/factor_discovery/base_residuals.parquet", keys, basecols, FOLD_KEY, True)
        table = pd.concat([keys, base[basecols]], axis=1)
        m = self.keyed("results/manual_ablation/activity_diet_features.parquet", keys, ad)
        table = pd.concat([table, m[ad]], axis=1)
        cohort = self.keyed("results/manual_ablation/comparison_cohorts.parquet", keys, ["cohort_core"], FOLD_KEY, True)
        table["cohort_core"] = cohort["cohort_core"]
        # Static baseline values: restrict subjects before numerical checks/statistics.
        static_cols = [c for c in manual if c.startswith("bio_")] + lab_old + ["bio_values_available"]
        raw = pd.read_parquet(self.root / "results/manual_ablation/static_features.parquet", columns=["subject", *static_cols])
        static = raw.loc[raw.subject.isin(keys.subject.unique())].copy()
        del raw
        if static.subject.duplicated().any():
            raise ValueError("Nonunique static subject mapping")
        table = table.merge(static, on="subject", validate="many_to_one")
        cov = self.csv("results/factor_discovery/metabolic_covariates.csv")
        cov = cov.loc[cov.subject.isin(keys.subject.unique()), ["subject", *STATES.values(), "bmi_kg_m2"]].copy()
        if cov.subject.duplicated().any():
            raise ValueError("Nonunique laboratory subject mapping")
        table = table.merge(cov, on="subject", validate="many_to_one")
        # Independently verify saved core flags using exactly the original window rule.
        derived_core = (table.act_hr_missing_120m.eq(0) | table.act_calories_missing_120m.eq(0)) & table.bio_values_available.eq(1)
        if not derived_core.eq(table.cohort_core).all():
            raise ValueError("Saved core definition differs from original feature rule")
        kernels = sorted({r["kernel_column"] for r in config["kernel_source"]["parameters"]})
        values = self.keyed("results/factor_discovery/candidates_batch1.parquet", keys, kernels, FOLD_KEY, True)
        for c in kernels:
            table[c] = values[c].to_numpy()
        bmi_ids = [f"int_{m}_carbs_x_bmi" for m in MEALS]
        saved_bmi = self.keyed("results/factor_discovery/candidates_batch2.parquet", keys, bmi_ids, FOLD_KEY, True)
        for c in bmi_ids:
            table[c] = saved_bmi[c].to_numpy()
        for col in ["target_value", "base_prediction"]:
            if not np.isfinite(table[col]).all():
                raise ValueError("Missing allowed target/base value")
        if not np.allclose(table.bio_bmi, table.bmi_kg_m2, atol=0, rtol=0):
            raise ValueError("BMI sources conflict")
        assert_trainval(table)
        return table.sort_values(FOLD_KEY).reset_index(drop=True)

    def add_selection_lags(self, table, columns):
        """Only isolated minute copies reach gather_lags; no baseline fit is performed."""
        assert_trainval(table)
        raw = pd.read_parquet(self.root / "results/aligned_data/dynamic_aligned_1min.parquet", columns=["subject", "timestamp", "libre_gl"])
        bounds = table.groupby("subject").agg(lo=("history_start_time", "min"), hi=("origin_time", "max"))
        lo, hi = raw.subject.map(bounds.lo), raw.subject.map(bounds.hi)
        minutes = raw.loc[raw.timestamp.ge(lo) & raw.timestamp.le(hi)].copy()
        del raw, lo, hi
        offsets = [int(c[1:]) for c in columns]
        lags = gather_lags(minutes, table, offsets, "libre_gl")
        result = table.copy()
        for j, c in enumerate(columns):
            result[c] = lags[:, j]
        return result


def lab_state_parameters(table):
    assert_trainval(table)
    subjects = table.loc[table.split.eq("train"), ["subject", *STATES.values()]].drop_duplicates()
    if subjects.subject.duplicated().any():
        raise ValueError("Laboratory state varies within a subject")
    values = subjects[list(STATES.values())]
    if not np.isfinite(values).all().all():
        raise ValueError("Complete laboratory cases required")
    means, std = values.mean(), values.std(ddof=0)
    if (std <= 0).any():
        raise ValueError("Constant laboratory state")
    return {"training_subjects": int(len(subjects)), "mean": means.to_dict(), "std_ddof0": std.to_dict()}


def lab_family_catalog(kernel_map, scheme):
    cat = conditional_lab_catalog().copy()
    for i, row in cat.iterrows():
        meal, state = row.candidate_id[4:].split("_carbs_x_")
        cat.loc[i, "formula"] = f"{kernel_map[meal]} * " + (f"z_train_subject({state})" if scheme == "C-S1" else state)
        cat.loc[i, "inputs"] = json.dumps([kernel_map[meal], STATES[state]])
    cat["generation_status"] = "generated in independent exploratory train/validation Experiment C only"
    cat["standardization"] = scheme
    return cat


def lab_interaction_columns(table, kernel_map, parameters, scheme):
    assert_trainval(table)
    if scheme not in ("C-S1", "C-S2"):
        raise ValueError("Unknown laboratory-state standardization scheme")
    out = {}
    for meal in MEALS:
        left = table[kernel_map[meal]].to_numpy(float)
        for state, column in STATES.items():
            right = table[column].to_numpy(float)
            if scheme == "C-S1":
                right = (right - parameters["mean"][column]) / parameters["std_ddof0"][column]
            cid = f"lab_{meal}_carbs_x_{state}"
            out[cid] = interact(left, right, cid).values
    return pd.DataFrame(out, index=table.index)


def check_bmi(table, kernel_map, tolerance=1e-9):
    assert_trainval(table)
    mask = table.split.eq("train") & table.cohort_core & table.residual.notna()
    checks = []
    for meal in MEALS:
        cid = f"int_{meal}_carbs_x_bmi"
        generated = interact(table[kernel_map[meal]].to_numpy(float), table.bmi_kg_m2.to_numpy(float), cid).values
        saved = table[cid].to_numpy(float)
        raw_diff = float(np.max(np.abs(generated - saved)))
        standardized = []
        for values in (generated, saved):
            imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[mask, None])
            scaler = StandardScaler().fit(imp.transform(values[mask, None]))
            standardized.append(scaler.transform(imp.transform(values[:, None])))
        std_diff = float(np.max(np.abs(standardized[0] - standardized[1])))
        checks.append({"candidate_id": cid, "raw_max_abs_diff": raw_diff, "standardized_max_abs_diff": std_diff,
                       "values": len(table), "passed": bool(raw_diff <= tolerance and std_diff <= tolerance)})
    if not all(r["passed"] for r in checks):
        raise ValueError("BMI interaction reproduction failed")
    return checks
